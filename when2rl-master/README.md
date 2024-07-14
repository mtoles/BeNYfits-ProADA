# When2RL

## Dependencies

- If you are using docker pulled from `jasonyux/liquid:latest`, you should only need to install `docker_requirements.txt` with `--no-deps`
- If not, use `full_requirements.txt` to install all dependencies.

Afterwards, run `export PYTHONPATH=$(pwd)` so all the relative imports would work.

## Evaluation

Before running any evaluation, make sure the following directories exist:
```bash
data/
├── alpaca_eval_results  # copied from `results` folder in the official alpaca_eval repo
│   ├── Conifer-7B-DPO
│   ├── Contextual-KTO-Mistral-PairRM
│   ├── Ein-70B-v0.1
│   └── ... (other model answers)
├── arena-hard-v0.1      # copied from `data` folder in the official Arena Hard Auto repo
│   ├── model_answer
│   ├── model_judgment
│   └── question.jsonl
├── mt_bench             # copied from `data` folder in the official MT-Bench repo
│   ├── model_answer
│   ├── model_judgment
│   ├── question.jsonl
│   └── reference_answer
└── openllm              # empty folder for storing openllm results
```

For more details on what should and will go into each directory, see following sections.

> Note: all the following utilizes `sglang` to generate responses, which is much faster but relies on manually passing in `chat_template`. Currently built-in ones are:
> ```bash
> llama-2, chatml, vicuna_v1.1  # see https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/conversation.py for more details
> ```
> This means if you are using any of the above, you can do ``--chat_template=llama-2`` to use the corresponding template. Otherwise, you need to manully write a chat template file and pass it in. Sglang reads it in by doing:
> ```python
> Conversation(
>   name=template["name"],
>   system_template=template["system"] + "\n{system_message}",
>   system_message=template.get("system_message", ""),
>   roles=(template["user"], template["assistant"]),
>   sep_style=sep_style,
>   sep=template.get("sep", "\n"),
>   stop_str=template["stop_str"],
> )
> ```
> where `template` will be your chat template `.json` file.


### Alpaca Eval 2.0

Running evaluation on a model using `weighted_alpaca_eval_gpt4_turbo` as metric (i.e., runs `gpt-4-turbo` as judge and computes length controlled win rate). Similar to Arena Hard Auto, all respones are computed against a baseline answer (defaults to `tatsu-lab/alpaca_eval` in huggingface datasets).

To run this using a single command line, you need to **first install `alpaca_eval`**. To do this:
- if you have python >= 3.10, you can install from the official github:
    ```bash
    pip install git+https://github.com/tatsu-lab/alpaca_eval
    ```
- otherwise, we modified the `alpaca_eval` to work with lower python versions (tested 3.8). You can install it by:
    ```bash
    pip install git+https://github.com/When2RL/alpaca_eval.git
    ```

Note that this will also download a `results` folder, which contains all precomputed model results used for computing the leaderboard. To make the following scripts work, you need to **copy/move that folder to `data/alpaca_eval_results`**.


Then, to run the evaluation:
```bash
CKPT_FOLDER=model_checkpoints
MODEL_NAME=gemma-2b-lion-v0.7-full-264k-beta0.05-epoch2-bsz64-zero1
EVAL_GPU_IDX=3

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python scripts/test/run_alpaca_eval.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--tokenizer_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME} \
--gen_temperature=0.7 \
--use_sglang \
--gen_parallel=16 \
--chat_template=scripts/configs/chat_templates/hf_gemma_zephyr.json \
--judge_only=false \
--judge_parallel=8 \
--to_wandb=true \
--num_runs=1
```

Note that `gen_temperature=1.0` is assumed for most other models in this benchmark.

To display the results manually, run:
```bash
RUN_NAME=gemma-2b-lion-v0.7-full-264k-beta0.05-epoch2-bsz64-zero1_run0

python src/evaluation/show_alpaca_eval_result.py \
--output_path=data/alpaca_eval_results/${RUN_NAME}
```


### Arena Hard Auto

Running evaluation on a model using `gpt-4o-2024-05-13` as judge.

```bash
CKPT_FOLDER=model_checkpoints/oil/gemma-2b-dpo-hpsweep
MODEL_NAME=gemma-2b-lion-v0.5-mix-10k-beta0.1-epoch26-from20
EVAL_GPU_IDX=7

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python scripts/test/run_arena_hard_auto.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--tokenizer_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME} \
--gen_temperature=0.0 \
--use_sglang \
--gen_parallel=16 \
--chat_template=scripts/configs/chat_templates/hf_gemma_zephyr.json \
--judge_only=false \  # if you use judge_only=true, then all params before has no effect
--judge_parallel=8 \
--judge_model=gpt-4o-2024-05-13 \
--baseline_model=gpt-4-0314 \
--to_wandb=true \
--num_runs=1
```

note that this will compute battles *against the baseline model*, and all performance is computed based on if the model wins/loses/ties against the baseline model.

To display all the results under a judge (assuming baseline is `gpt-4-0314`):

```bash
python src/evaluation/show_arena_hard_result.py --judge-name=gpt-3.5-turbo-0125 --baseline=gpt-4-0314
```

> For measuring scaling performance of small models such as gemma-2b, you may want to use `gpt-3.5-turbo-0125` as baseline model instead


### MT-Bench

There are currently three ways to do this. The second and third should give the same results. The second methods is perferred if you are working with THIS repository mainly.


**1. Using the native scripts from `fastchat`**:

Say we are interested in the performance of `alignment-handbook/zephyr-7b-sft-full`

1. make sure you already symlinked the `data/mt_bench` to the correct location (TODO)
2. generate answers:
    ```bash
    python -m fastchat.llm_judge.gen_model_answer --model-path alignment-handbook/zephyr-7b-sft-full --model-id zephyr-7b-sft-full
    ```
    where after this we will refer to this model's performance using `zephyr-7b-sft-full`
3. score the results (assumes `OPENAI_API_KEY` and `OPENAI_ORGANIZATION` set)
    ```bash
    python -m fastchat.llm_judge.gen_judgment --model-list zephyr-7b-sft-full --parallel 4 --judge-model gpt-4
    ```
    this by default appends all results to `data/mt_bench/model_judgment/gpt-4_single.jsonl`. To switch to using `gpt-3.5-turbo`, add the flag `"--judge-model gpt-3.5-turbo`.
4. show result (our own script, which shows more details):
    ```bash
    python src/evaluation/show_mt_bench_result.py --model-list zephyr-7b-sft-full --judge-model gpt-4
    ```

You can also browse through the results in a web-brower with:
```bash
python src/evaluation/qa_browser.py --judge-model=gpt-4-0125-preview
```
this browser is modified from the original browser so that you can 1) choose judge model, and 2) show model scores in the pairwise comparison panel.

**2. Using a wrapper script from this repo**:

The main difference is that here we support 1) running multiple iterations + average, 2) report the results to `wandb` automatically, and 3) using sglang to speed up inference.

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/test/run_mt_bench.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME} \
--use_sglang \ # whether to use sglang to speed up inference
--gen_parallel 16 \ # used only when you use sglang
--chat_template scripts/configs/chat_templates/hf_gemma_zephyr.json \ # used only when using sglang
--judge_parallel=16 \
--judge_model=gpt-4-0125-preview \
--to_wandb=true \  # log to wandb
--num_runs=2 \  # number of runs to average. TODO: num=3 often hangs
--y # no confirmation prompt
```
where:
- this will log the results to `wandb` by finding the `wandb_id` from the `{model_path}/run_args.yaml` file
- the final performance will also be logged to `{model_path}`. Otherwise, you can specify another directory by, e.g., adding the flag `--result_save_path=model_checkpoints/debug`
- the main implementations for generating model responses/judgments are basically copied from the `fastchat` scripts


If you do not want to use `sglang` or upload to `wandb`, then the equivalent of the above would be:
```bash
CUDA_VISIBLE_DEVICES=7 python scripts/test/run_mt_bench.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME}_templ_chatml \  # use chatml template
--judge_parallel=16 \
--judge_model=gpt-4-0125-preview \
--num_runs=2 \ 
--y # no confirmation prompt
```

**3. Using `run_llm_judge.py` from the `llm_judge_plus` repo**

Essentially the same functionality except for loop runs, but added support for using `sglang` to speed up inference. This requires you to have setup `sglang` correctly. If so, then you can do:

```bash
python run_llm_judge.py \
--model-path ${CKPT_FOLDER}/${MODEL_NAME} \
--model-id ${MODEL_NAME} \
--chat-template chat_templates/HuggingFaceH4_zephyr-7b-beta.json \
--overwrite true
```

When reporting results from this method, make sure you report averages of at least 2 runs. See more details at: https://github.com/When2RL/llm_judge_plus.


### OpenLLM

This assumes you have installed the 0.4.1 version of `lm_eval`.

```bash
python scripts/test/run_lm_eval.py \
--model_name_or_path=HuggingFaceH4/zephyr-7b-beta \
--torch_dtype=bfloat16 \
--attn_implementation="flash_attention_2" \
--batch_size=16 \
--output_path=data/openllm/zephyr-7b-beta
```
With the latest version of `lm_eval`, the above can reproduce most results on the official OpenLLM leaderboard.


For stable lm, there seems to be a bug when loading a saved tokenizer, so you may need to do this:
```bash
python scripts/test/run_lm_eval.py \
--model_name_or_path=model_checkpoints_coffee/stablelm-sft-full_bsz32_lr2e-5/checkpoint-11472 \
--tokenizer_name_or_path=stabilityai/stablelm-2-1_6b \
--tokenizer_revision=39d1453f64ffe2a97df9b2f1e6d007eb28248245 \
--torch_dtype=bfloat16 \
--batch_size=16 \
--output_path=data/openllm/stablelm-sft-full_bsz32_lr2e-5
```

## Data Analysis

### Dataset Prediction

Is there a difference between datset x and dataset y? A simple and, as it turns out, quite effective way to do this is to consider the dataset prediction task:
- if the two datasets are easily distinguishable, then the model should be able to predict which dataset a given sample comes from
- if the two datasets are not easily distinguishable, we should then expect accuracy near 50%

Training and testing a `jinaai/jina-embeddings-v2-base-en` to **distinguish between UltraFeedback and UltraChat**. This gives around 86.7% accuracy!
```bash
python scripts/analysis/dset_prediction.py scripts/configs/dset_pred_ultra.yaml \
--output_dir=model_checkpoints/dset_analysis/dset_pred_ultrafeedback_v_ultrachat \
--seed=42
```

Training and testing to **distinguish between UltraChat and UltraChat (dummy test)**. This gives only around 46.7% accuracy.
```bash
python scripts/analysis/dset_prediction_dummy.py scripts/configs/dset_pred_dummy_ultra.yaml \
--output_dir=model_checkpoints/dset_analysis/dset_pred_ultrachat_v_ultrachat \
--seed=42
```

You can test subsets of the dataseta as well by modifying the config file. For example, training and testing to **distinguish between `evo_instruct` subset of UltraFeedback and the UltraChat dataset**:
```yaml
# scripts/configs/dset_pred_ultra.yaml
dataset_to_test:
  when2rl/UltraFeedback_binarized_cleaned_annotated: train_prefs
  HuggingFaceH4/ultrachat_200k: train_sft
# a dict for each dataset
filtering:
  when2rl/UltraFeedback_binarized_cleaned_annotated:
    source: evol_instruct
  HuggingFaceH4/ultrachat_200k:
per_dataset_size:
  train: 1000
  validation: 500
  test: 500
content_to_predict: prompt
```
Then run (which gives a `94.7` accuracy):
```bash
python scripts/analysis/dset_prediction.py scripts/configs/dset_pred_ultra.yaml \
--output_dir=model_checkpoints/dset_analysis/dset_pred_ultrachat_v_evoinstruct \
--seed=42
```


### Dataset Prediction V2

The idea is to check if there are any distinguishing data by comparing
- sub-dataset A directly against sub-dataset B
- data unique in A against data unique in B

This is essentially achieved by reading from `.csv` file that contains the data `full_id` that you want to predict.

```bash
python scripts/analysis/dset_prediction_idfile.py scripts/configs/analysis/dset_pred_idfile.yaml \
--output_dir=data/analysis/ultrafbk/rm-importance-ge0.9_v_score-diff-ge3_confidence \
--seed=42
```

To get those `.csv` file, an example would look like:

```python
## a properly formatted dataset where we added rm_weight and score_diff columns
real_datasets_w_weight_df[real_datasets_w_weight_df['rm_weight'] >= 0.9].join(
    real_datasets_w_weight_df[real_datasets_w_weight_df['score_diff'] < 3.0],
    how='inner',
    lsuffix='_caller',
)['prompt_id'].to_csv(
    "../../data/analysis/ultrafbk/rm-importance-ge0.9_v_score-diff-ge3_confidence/rm-importance-only_train_data_ids.csv",
    index=False
)
```

### Compute LM Reward


(multigpu not yet supported)

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/analysis/compute_lm_reward.py scripts/configs/analysis/lm_reward.yaml \
--model_name_or_path=alignment-handbook/zephyr-7b-sft-full \
--ref_model_name_or_path=model_checkpoints/oil/reprod/zephyr-7b-dpo-full-orca_2epoch \
--output_dir=data/analysis/orca_pairs/dpo_v_sft_importance
```

to change what dataset its evaluated on, modify the `lm_reward.yaml` file.


### Compute (Partial) Importance Weight


(multigpu not yet supported)

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/analysis/compute_importance_weight.py \
scripts/configs/analysis/importance_weight.yaml \
--model_name_or_path=WizardLM/WizardLM-7B-V1.0 \
--output_dir=data/analysis/orca_pairs/WizardLM-7B-V1.0_weights \
--torch_dtype=bfloat16 \
--per_device_train_batch_size=4 \
--use_flash_attention_2=true
```

to change what dataset its evaluated on, modify the `lm_reward.yaml` file.

To load models upto 70B size, the following config works:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/analysis/compute_importance_weight.py \
scripts/configs/analysis/importance_weight.yaml \
--model_name_or_path=WizardLM/WizardLM-70B-V1.0 \
--output_dir=data/analysis/orca_pairs/WizardLM-70B-V1.0_8bit_weights \
--per_device_train_batch_size=1 \
--torch_dtype=bfloat16 \
--use_flash_attention_2=true \
--llm_int8_enable_fp32_cpu_offload=True \
--load_in_8bit=True
```


### Reward Augmentation

First we can recover using gpt-4 as judge to annotate datasets:
```bash
python scripts/analysis/judge_dset.py \
--eval_mode=single \  # {single, all}
--dset_name=when2rl/UltraFeedback_binarized_cleaned_annotated \
--dset_split=train_prefs \
--num_to_judge=150 \  # number of samples to use for annotation
--judge_model=gpt-4-0125-preview \
--judge_parallel=8 \
--output_path=data/analysis/ultrafbk/gpt-4-0125-preview_150-single.csv
```

NOTE: It seems that using `all` gives a different result than `single` mode! With single this is consistent with the original annotation for 92.61% of the time, but with `all` it is only 80.54% of the time.


TODO: Can we use a RM to filter out and only keep the "high-quality" samples?

prelimnary analysis: measure the performance of existing RM models:
```bash
python scripts/analysis/predict_preference.py \
--output_dir model_checkpoints_coffee/dset_analysis/reward_preds/ultrafbk_500_starlingrm \
--model_name_or_path berkeley-nest/Starling-RM-7B-alpha
```

## Generating More Data

Methods to get more data

### Generate Online DPO data

The following will take prompts from `when2rl/UltraFeedback_binarized_cleaned_annotated` and generate responses using the model `gemma-2b-lion-v0.6-full-165k-beta0.05-epoch4-from2-bsz64-zero1`. The responses are then judged using `pair-rm` and saved to `data/lion-dpo-online/UltraFeedback-gemma-2b-lion-v0.6-165k-epoch4`.

```bash
export PYTHONPATH=$(pwd)

CKPT_FOLDER=model_checkpoints/oil/lion-gemma-v0.1
MODEL_NAME=gemma-2b-lion-v0.6-full-165k-beta0.05-epoch4-from2-bsz64-zero1
EVAL_GPU_IDX=5,6
NUM_GPUS=2
PORT=29572

# manually host the sglang server
# echo "python -m sglang.launch_server --model-path ${CKPT_FOLDER}/${MODEL_NAME} --enable-flashinfer --attention-reduce-in-fp32 --chat-template ${CHAT_TEMPLATE} --port 60001"
# echo "python -m sglang.launch_server --model-path ${CKPT_FOLDER}/${MODEL_NAME} --enable-flashinfer --attention-reduce-in-fp32 --chat-template ${CHAT_TEMPLATE} --port 60002"

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--num_processes=${NUM_GPUS} \
--main_process_port=${PORT} \
scripts/gen_data/gen_preference_pairs.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME} \
--sglang_ports='60001,60002' \
--prompt_dataset=when2rl/UltraFeedback_binarized_cleaned_annotated \
--prompt_dataset_split=train_prefs \
--max_samples=-1 \
--n_to_rank=5 \
--gen_temperature=0.8 \
--gen_parallel=16 \
--chat_template=scripts/configs/chat_templates/hf_gemma_zephyr.json \
--judge_only=false \
--judge_batch_size=8 \
--dset_save_name=UltraFeedback-gemma-2b-lion-v0.6-165k-epoch4
```

### Generate and Judge

To generate a model's response given the prompts of a given dataset and then judge it:

1. launch with sglang to speed up inference:
    ```bash
    python -m sglang.launch_server \
    --model-path Columbia-NLP/gemma-2b-lion-sft-v0.1 \
    --port 41911 \
    --enable-flashinfer \
    --attention-reduce-in-fp32 \
    --chat-template scripts/configs/chat_templates/hf_gemma_zephyr.json
    ```

2. genereate model response (TODO: hardcoded sglang url and dset)
    ```bash
    python scripts/gen_data/gen_response.py
    ```

3. generate judgement score
    ```bash
    python scripts/gen_data/judge_gen_data.py \
    --dset_name=when2rl/dpo-mix-7k-rescaled_reformatted \
    --dset_split=train \
    --gen_data_path=data/dpo-mix-7k/gemma_2b_sft.csv \
    --output_path=data/dpo-mix-7k/gemma_2b_sft_gpt4-turbo-scored.csv \
    --num_to_judge=500 # test before going on the full dset (specify -1)
    ```

## Training

To do both training AND evaluation, do (or a better approach - just write a shell script):

```bash
CKPT_FOLDER=model_checkpoints_coffee
MODEL_NAME=gemma-2b-dpo-mix-beta0.005
CONFIG_FILE=scripts/configs/gemma/dpo-full-2b.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file scripts/configs/deepspeed_zero3.yaml \
--main_process_port=29500 \
scripts/train/whatever_training_script.py ${CONFIG_FILE} \
--output_dir=${CKPT_FOLDER}/${MODEL_NAME} \
--other_hyperparmers=x
&& CUDA_VISIBLE_DEVICES=4 python scripts/test/run_mt_bench.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME} \
--use_sglang \
--gen_parallel=16 \
--chat_template scripts/configs/chat_templates/hf_gemma_zephyr.json \
--judge_parallel=8 \
--judge_model=gpt-4-0125-preview \
--to_wandb=true \
--num_runs=2 \
--y
&& CUDA_VISIBLE_DEVICES=4 python scripts/test/run_lm_eval.py \
--model_name_or_path=${CKPT_FOLDER}/${MODEL_NAME} \
--torch_dtype=bfloat16 \
--batch_size=16 \
--output_path=data/openllm/${MODEL_NAME} \
--to_wandb=true
```

where the first part of the command will use 4 GPUs to do training, and the latter `lm_eval` evaluation script will:
1. read the `wandb` informatino (e.g., id and project name) from the `model_checkpoints/some_folder_to_save_model/run_args.yaml`
2. load the model from `model_checkpoints/some_folder_to_save_model`
3. evaluate the model using `lm_eval` and save the results to `data/openllm/result_folder`
4. log the results to `wandb` using the run id obtained from step 1.


### SFT trainig

#### Reproducing Zephyr-SFT-2
Below is an example to train from `alignment-handbook/zephyr-7b-sft-full` using "top-ranked" data from the UltraFeedback dataset.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file scripts/configs/deepspeed_zero3.yaml \
--main_process_port=29500 \
--num_processes {NUM_GPUS_USED, 4 in this case} \
scripts/train/zephyr_sft_2.py scripts/configs/zephyr/sft-2.yaml \
--wandb_group=zephyr-sft-2-reprod
```

which will use four GPUs to train the model, and it should take about 20-30 minutes since the dataset is small after filtering.

See `scripts/configs/sft-2.yaml` for what hyperparameters/datasets are used. Note that this setup assumes:
- model will be saved to `model_checkpoints/`
- you have already setup your `wandb` account (if not, do `wandb login` in the terminal)

#### Training Zephyr-SFT by mixing UltraChat and UltraFeedback

We can instead directly mix `UltraChat` and `UltraFeedback` datasets altogether and train from Mistral-7B. This is done by:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file scripts/configs/deepspeed_zero3.yaml \
--main_process_port=29500 \
--num_processes 4 \
scripts/train/zephyr_sft_2.py scripts/configs/zephyr/sft-2-mixed.yaml \
--wandb_group=zephyr-sft-2-mixed \
--output_dir=model_checkpoints_coffee/zephyr-7b-sft-2-mixed
```

### Training SFT-1 with StableLM

Since we will be mostly doing in a single-GPU setting, no need to do `accelerate`:

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=20002 \
scripts/train/zephyr_sft.py scripts/configs/stablelm/sft-1.yaml \
--output_dir=model_checkpoints_coffee/stablelm-sft-full_bsz128_lr2e-5 \
--learning_rate=2e-5 \
--gradient_accumulation_step=8 \
--per_device_train_batch_size=16
```


### DPO training

There are two ways to do it currently. Either you can go for the native implementation `scripts/train/zephyr_dpo.py`, or a modified `scripts/train/zephyr_dpo_precompute.py` that uses precomputed logprobs to save GPU memory.

**1. Native DPO training**

(preferred)
Very similar to SFT training, but using `scripts/train/zephyr_dpo.py` instead. For example:

```bash
CKPT_FOLDER=model_checkpoints_coffee
MODEL_NAME=gemma-2b-dpo-HermesReason-beta0.01
CONFIG_FILE=scripts/configs/gemma/dpo-full-2b.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file scripts/configs/deepspeed_zero3.yaml \
--main_process_port=29500 \
--num_processes 4 \
scripts/train/zephyr_dpo.py ${CONFIG_FILE} \
--output_dir=${CKPT_FOLDER}/${MODEL_NAME} \
--max_length=2048 \
--beta=0.01 \
--num_train_epochs=1 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps=32 \
--wandb_group=zephyr-gemma-2b \
--save_strategy=no \
--save_total_limit=-1
```

**2. DPO training with precomputed logprobs**


You will first need to compute the logprobs with `scripts/train/precompute_ref_logprobs.py`, and then load the saved log probs with `scripts/train/zephyr_dpo_precompute.py`.

1. consider the following setting:
    ```bash
    CKPT_FOLDER=model_checkpoints
    MODEL_NAME=debug
    PRECOMPUTE_FILE=ultrafbk/gemma-2b-zephyr-sft_2048.csv
    CONFIG_FILE=scripts/configs/gemma/dpo-full-2b.yaml
    ```
    where inside `scripts/configs/gemma/dpo-full-2b.yaml` we set:
    ```yaml
    dataset_splits:
      when2rl/UltraFeedback_binarized_cleaned_annotated: ["train_prefs", "test_prefs"]
    dataset_mixer:
      when2rl/UltraFeedback_binarized_cleaned_annotated: 1.0
    ```
2. then, first compute the logprobs:
    ```bash
    CUDA_VISIBLE_DEVICES=7 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file scripts/configs/deepspeed_zero2.yaml \
    --main_process_port=29500 \
    --num_processes=1 \
    scripts/train/precompute_ref_logprobs.py ${CONFIG_FILE} \
    --output_dir=${CKPT_FOLDER}/${MODEL_NAME} \
    --precompute_file_path=data/precompute/${PRECOMPUTE_FILE} \
    --max_length=2048 \
    --per_device_train_batch_size=4
    ```
3. then do training with this precomputed logprobs:
    ```bash
    CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file scripts/configs/deepspeed_zero3.yaml \
    --main_process_port=29500 \
    --num_processes=4 \
    scripts/train/zephyr_dpo_precompute.py ${CONFIG_FILE} \
    --precompute_file_path=data/precompute/${PRECOMPUTE_FILE} \
    --output_dir=${CKPT_FOLDER}/${MODEL_NAME} \
    --max_length=2048 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --wandb_group=debug \
    --num_train_epochs=1 \
    --save_strategy=no \
    --save_total_limit=-1
    ```

### DPO with random sampling (baseline)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file scripts/configs/deepspeed_zero3.yaml \
--main_process_port=29501 \
scripts/train/zephyr_dpo_subsample.py scripts/configs/zephyr/dpo-subsample.yaml \
--wandb_group=zephyr-subsample \
--output_dir=model_checkpoints/zephyr-7b-random5k_3epoch \
--max_data_size=5000 \
--save_strategy=no \
--save_total_limit=-1 \
--num_train_epochs=3
```

where the `--save_strategy=no` and `--save_total_limit=-1` is hardcoded for the script to save the model only at the end of training.


### DPO with Preference Difference

Train DPO while controlling for the data budget and preference difference:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file scripts/configs/deepspeed_zero3.yaml \
--main_process_port=29500 \
--num_processes {NUM_GPUS_USED, 4 in this case} \
scripts/train/zephyr_dpo_pref_strength.py scripts/configs/zephyr/dpo-pref-strength.yaml \
--wandb_group=zephyr-pref-strength \
--num_train_epochs=5 \
--save_total_limit=1
```

and the preference strength is controlled in the `.yaml` config file:
```yaml
# other config omitted
preference_config:
  -100: 1.0  # uniform
max_data_size: 2000
```

where `preference_config` specifies the weight for each preference difference. For example, `{-100:1.0, 1.0:2.0, 5.0:0.5}` would mean:
```python
{
  0.0: 1.0,
  1.0: 2.0,
  2.0: 1.0
  3.0: 1.0
  4.0: 1.0
  5.0: 0.5
  6.0: 1.0
  ...
  9.0: 1.0
}
```


### DPO with LM Reward Weight

```bash
scripts/train/zephyr_dpo_rm_importance.py scripts/configs/zephyr/dpo-rm-importance.yaml \
```

### Dataset Mixing

This is all handled internally by `mix_datasets` function in `src/dataloads/data.py`. Under the hood, given the following config as an example:
```yaml
dataset_splits:
  HuggingFaceH4/ultrafeedback_binarized: ["train_prefs", "test_prefs"]
  argilla/ultrafeedback-binarized-preferences-cleaned: ["train"]
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0
  argilla/ultrafeedback-binarized-preferences-cleaned: 1.0
```
it will do:
1. load all the datasets and the corresponding splits specified in `dataset_splits`
2. transform the data into a common format (see `src/dataloads/formatting.py`)
3. put all `train*` splits into a training raw dataset, and all `test*` splits into a testing raw dataset
4. mix the training raw dataset using the weights specified in `dataset_mixer`. In this case, `1.0` for all training datasets from `HuggingFaceH4/ultrafeedback_binarized`, and `1.0` for all training datasets from `argilla/ultrafeedback-binarized-preferences-cleaned`.

For real examples, see `scripts/configs/sft-2.yaml` and `scripts/configs/dpo-full.yaml`.