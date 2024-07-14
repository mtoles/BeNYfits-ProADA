import logging
import random
import torch
import os
import pandas as pd
from transformers import set_seed, AutoModelForCausalLM
from datasets import DatasetDict, Dataset
from src.trainers.configs import (
    DataArguments, EfficientDPOConfig, LoggingArguments, ModelArguments,
    H4ArgumentParser
)
from src.dataloads.data import apply_chat_template, get_datasets
from src.dataloads.decontaminate import decontaminate_humaneval
from src.utils.data_utils import add_full_id
from src.utils.model_utils import (
    get_checkpoint,
    get_tokenizer,
)
from src.utils.utils import init_logger, is_main
from src.trainers.dpo_fixed_v2 import tokenize_row, EfficientDPOTrainer
from src.constants import DPO_DATA_COLUMNS_TO_REMOVE, DPO_DATA_MIX_COLUMNS


logger: logging.Logger


def prepare_datasets(
    data_args: DataArguments,
    training_args: EfficientDPOConfig,
    raw_datasets: DatasetDict,
    tokenizer,
):
    global logger
    column_names = DPO_DATA_COLUMNS_TO_REMOVE
    column_names.remove("prompt_id")
    column_names.remove("other_info")

    raw_datasets = raw_datasets.map(
        add_full_id,
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        desc="Adding full_id to dataset",
    )

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    ##########################
    # Decontaminate benchmarks
    ##########################
    num_raw_train_samples = len(raw_datasets["train"])
    raw_datasets = raw_datasets.filter(
        decontaminate_humaneval,
        fn_kwargs={"text_column": "text_chosen"},
        batched=True,
        batch_size=10_000,
        keep_in_memory=True,
        num_proc=1,
        desc="Decontaminating HumanEval samples",
    )
    num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    logger.info(
        f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    ######################################################
    # pre-tokenize data used originally inside DPO trainer
    ######################################################
    raw_datasets = raw_datasets.map(
        tokenize_row,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": training_args.max_length,
            "truncation_mode": "keep_end",
            "max_prompt_length": training_args.max_prompt_length,
            "label_pad_token_id": -100,
        },
        num_proc=data_args.preprocessing_num_workers,
        desc="(Map) Truncating prompt and responses"
    )
    return raw_datasets


def precompute_all_ref_probs(
    training_args: EfficientDPOConfig,
    ref_model: str,
    ref_model_kwargs: dict,
    datasets,
    tokenizer,
):
    if training_args.ref_update_steps != -1:
        return datasets['train'], datasets['test']

    save_path = os.path.join('data/precompute', training_args.output_dir.split('/')[-1])
    train_save_path = os.path.join(save_path, 'train_ref_logps.csv')
    test_save_path = os.path.join(save_path, 'test_ref_logps.csv')
    if training_args.precompute_train_ref_file_path != '':
        train_save_path = training_args.precompute_train_ref_file_path
    if training_args.precompute_test_ref_file_path != '':
        test_save_path = training_args.precompute_test_ref_file_path

    model = None
    trainer = None
    if os.path.exists(train_save_path):
        # load csv
        logger.info(f"Loading precomputed reference log probabilities from {train_save_path}")
        train_logps_df = pd.read_csv(train_save_path)
    else:
        # precompute all when ref_update_steps == -1
        # doing it here allows using a different ref model than the main model
        logger.info(f"Precomputing reference log probabilities using {ref_model}")
        model = model or AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_kwargs)

        trainer = trainer or EfficientDPOTrainer(
            model,
            args=training_args,
            train_dataset=datasets["train"],
            tokenizer=tokenizer,
        )
        print("returning average logps")
        trainer.loss_type = 'ipo' # trl uses average log prob when loss type is ipo
        
        train_logps = trainer.compute_all_reference_log_probs(datasets['train'])
        train_logps_df = pd.DataFrame(train_logps).drop_duplicates(subset='full_id')
        train_logps_df.to_csv(train_save_path, index=False)

    # same for test
    if os.path.exists(test_save_path):
        # load csv
        logger.info(f"Loading precomputed test reference log probabilities from {test_save_path}")
        test_logps_df = pd.read_csv(test_save_path)
    else:
        logger.info(f"Precomputing test reference log probabilities using {ref_model}")
        model = model or AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_kwargs)

        trainer = trainer or EfficientDPOTrainer(
            model,
            args=training_args,
            train_dataset=datasets["train"],
            tokenizer=tokenizer,
        )
        print("returning average logps")
        trainer.loss_type = 'ipo' # trl uses average log prob when loss type is ipo

        test_logps = trainer.compute_all_reference_log_probs(datasets['test'])
        test_logps_df = pd.DataFrame(test_logps).drop_duplicates(subset='full_id')
        test_logps_df.to_csv(test_save_path, index=False)
    
    # update the datasets
    train_df: pd.DataFrame = datasets['train'].to_pandas()
    train_df.index = train_df['full_id'].values
    train_logps_df.index = train_logps_df['full_id'].values
    train_logps_df = train_logps_df.drop(columns=['full_id'])
    train_df = train_df.join(
        train_logps_df,
        on='full_id',
        how='inner'
    )

    # update the datasets
    test_df: pd.DataFrame = datasets['test'].to_pandas()
    test_df.index = test_df['full_id'].values
    test_logps_df.index = test_logps_df['full_id'].values
    test_logps_df = test_logps_df.drop(columns=['full_id'])
    test_df = test_df.join(
        test_logps_df,
        on='full_id',
        how='inner'
    )
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


def main():
    global logger
    parser = H4ArgumentParser((ModelArguments, DataArguments, LoggingArguments, EfficientDPOConfig))
    model_args, data_args, logging_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    #######
    # Setup
    #######
    log_level = training_args.get_process_log_level()
    logger = init_logger(is_main=is_main(), log_level=log_level, is_distributed=True)

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    column_to_keep = DPO_DATA_MIX_COLUMNS
    
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits, col_to_mix=column_to_keep)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    if 'stablelm' in tokenizer.name_or_path:
        logger.warning("Setting pad token id to 100288 assuming you are using StableLM tokenizer")
        tokenizer.pad_token_id = 100288

    raw_datasets = prepare_datasets(
        data_args=data_args,
        training_args=training_args,
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
    )


    ############################
    # prepare model loading args
    ############################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=None
    )

    ####################################################################################
    # pre-compute all reference. Used when you want to do DPO with a different ref model
    ####################################################################################
    precompute_all_ref_probs(
        training_args=training_args,
        ref_model=model_args.ref_model_name_or_path,
        ref_model_kwargs=model_kwargs,
        datasets=raw_datasets,
        tokenizer=tokenizer,
    )
    return


if __name__ == "__main__":
    main()