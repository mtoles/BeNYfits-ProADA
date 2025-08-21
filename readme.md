# 🗽 BeNYfits: Determine User Eligibility for Public Benefits Through Dialog

Welcome to the BeNYfits/ProADA repository. This repository contains

- Dataset and evaluation for the BeNYfits benchmark
- ProADA Prgram Synthesis Adaptive Decision Agent implementation

This is the companion repository to [Program Synthesis Dialog Agents for Interactive Decision-Making](https://arxiv.org/abs/2502.19610)

## 🗂️ Raw Data

The raw datasets can be found in jsonl form in the `./dataset` directory. This directory also contains natural language descriptions of eligibility requirements in `./dataset/benefits_clean.jsonl`

## 📈 Evaluation

You can replicate our evaluation using the Dockerfile. First, build the Docker image:

```bash
docker build -t benefitsbot .
```

Then run the container:

```bash
docker run \
  --gpus all \
  --rm \
  -p 55221:55221 \
  -e HF_TOKEN=$HF_TOKEN \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e CHAT_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e CHATBOT_STRATEGY=backbone \
  -e SYNTHETIC_USER_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  -e ESTRING=output_dir_name \
  -e DATASET_PATH=dataset/representative_dataset.jsonl \
  benefitsbot
```

You can pass parameters for `analysis/benefitsbot.py` to `docker run` using the -e flag.
Key parameters include:

- `--chat_model_id` - The full HuggingFace, OpenAI, or Anthropic model ID to be used for dialog and prediction turns
- `--code_model_id` - The full HuggingFace, OpenAI, or Anthropic model ID to be used for code generation
- `--chatbot_strategy` - One of:
  - `backbone` - Directly prompt the model
  - `cot` - Use chain-of-thought
  - `codebot` - Use ProADA (ours). This requires additionally `code_model_id`
- `--dataset_path` - `dataset/diverse_dataset.jsonl` or `dataset/representative_dataset.jsonl`

## 🧑‍🔬 Development

If you wish to test a locally hosted backbone model, the easiest strategy is to host it on `HuggingFace` and choose it with the `chat_model_id` parameter.

If you wish to test an agent with custom logic, you can edit `datamodels/chatbot.py`. 

## 📚 Cite
```
@misc{toles2025programsynthesisdialogagents,
      title={Program Synthesis Dialog Agents for Interactive Decision-Making}, 
      author={Matthew Toles and Nikhil Balwani and Rattandeep Singh and Valentina Giulia Sartori Rodriguez and Zhou Yu},
      year={2025},
      eprint={2502.19610},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.19610}, 
}
```