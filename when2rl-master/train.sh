export PYTHONPATH="/home/mt3639/miniconda3/envs/dpo/bin/python"

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=3
TRAIN_GPU_IDX=0,1,2,3  # we use task scheduler
# TRAIN_GPU_IDX=0
NUM_GPUS=4
# NUM_GPUS=1
EVAL_GPU_IDX=0

CKPT_FOLDER=model_checkpoints
MODEL_NAME=llama-8b-lion-v0.2-full-180k-beta0.025-epoch1-bsz128-zero3
LOGP_TRAIN_FILE=data/precompute/lion-dpo-v0.5/llama-8b-lion-v0.2-train-full-180k.csv
LOGP_TEST_FILE=data/precompute/lion-dpo-v0.5/llama-8b-lion-v0.2-test.csv
CONFIG_FILE=scripts/configs/llama/dpo-full-8b-subsample-extended.yaml
MAIN_PORT=29507
MT_JUDGE_MODEL=gpt-4-0125-preview
AHA_JUDGE_MODEL=gpt-4o-2024-05-13
AHA_BASELINE_MODEL=gpt-3.5-turbo-0125


ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file scripts/configs/deepspeed_zero3.yaml \
--main_process_port=${MAIN_PORT} \
--num_processes=${NUM_GPUS} \
scripts/train/zephyr_dpo_subsample_v2.py ${CONFIG_FILE} \
--model_name_or_path=Columbia-NLP/lion-llama-3-8b-sft-dev-v0.2 \
--ref_model_name_or_path=Columbia-NLP/lion-llama-3-8b-sft-dev-v0.2 \
--precompute_train_ref_file_path=${LOGP_TRAIN_FILE} \
--precompute_test_ref_file_path=${LOGP_TEST_FILE} \
--dataset_splits='{"Columbia-NLP/lion-dpo-mix-v0.5": ["train_180k", "test"]}' \
--dataset_mixer='{"Columbia-NLP/lion-dpo-mix-v0.5": 1.0}' \
--output_dir=${CKPT_FOLDER}/${MODEL_NAME} \
--max_data_size=-1 \
--learning_rate=0.0000005 \
--ref_update_steps=-1 \
--do_eval=true \
--evaluation_strategy=epoch \
--max_length=2048 \
--beta=0.025 \
--num_train_epochs=1 \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--gradient_accumulation_steps=16 \
--wandb_group=llama-8b-lion \
--save_strategy=no \
--save_total_limit=-1