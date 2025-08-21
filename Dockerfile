# Use a more recent CUDA image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System & Python deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python requirements
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source & .env, install your package
COPY . .
RUN pip3 install --no-cache-dir -e .

# Pass through your env‑vars and make all benefitsbot.py arguments configurable
ENV OPENAI_API_KEY=${OPENAI_API_KEY} \
    HF_TOKEN=${HF_TOKEN} \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    LM_PORT_NO=${LM_PORT_NO:-55221} \
    LM_SERVER_URL=${LM_SERVER_URL:-"http://127.0.0.1"} \
    CHAT_MODEL_ID=${CHAT_MODEL_ID:-"meta-llama/Llama-3.1-8B-Instruct"} \
    CODE_MODEL_ID=${CODE_MODEL_ID:-"gpt-4o-2024-08-06"} \
    CHATBOT_STRATEGY=${CHATBOT_STRATEGY:-"backbone"} \
    MAX_CODE_GEN_ATTEMPTS=${MAX_CODE_GEN_ATTEMPTS:-1} \
    MAX_CODE_REWRITE_ATTEMPTS=${MAX_CODE_REWRITE_ATTEMPTS:-0} \
    SYNTHETIC_USER_MODEL_NAME=${SYNTHETIC_USER_MODEL_NAME:-"meta-llama/Llama-3.1-8B-Instruct"} \
    MAX_DIALOG_TURNS=${MAX_DIALOG_TURNS:-100} \
    ELIGIBILITY_REQUIREMENTS=${ELIGIBILITY_REQUIREMENTS:-"./dataset/benefits_clean.jsonl"} \
    DATASET_PATH=${DATASET_PATH:-"dataset/representative_dataset.jsonl"} \
    DOWNSAMPLE_SIZE=${DOWNSAMPLE_SIZE:-""} \
    DS_SHIFT=${DS_SHIFT:-0} \
    TOP_K=${TOP_K:-20} \
    PROGRAMS=${PROGRAMS:-"EarlyHeadStart InfantToddlerPrograms HeadStart ComprehensiveAfterSchoolSystemOfNYC PreKForAll"} \
    NUM_PROGRAMS=${NUM_PROGRAMS:-""} \
    ESTRING=${ESTRING:-"eval"} \
    USE_CACHE=${USE_CACHE:-"true"} \
    RANDOM_SEED=${RANDOM_SEED:-0}

# Startup script with better error handling and configurable arguments
RUN echo '#!/usr/bin/env bash\n\
set -e\n\
\n\
# Check required environment variables\n\
if [ -z "$HF_TOKEN" ]; then\n\
    echo "ERROR: HF_TOKEN environment variable is required"\n\
    exit 1\n\
fi\n\
\n\
echo "Starting model server..."\n\
# login & launch model server\n\
huggingface-cli login --token "${HF_TOKEN}"\n\
\n\
echo "Launching uvicorn server..."\n\
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \\\n\
  uvicorn server.concurrent_multiple_model_server:app --port ${LM_PORT_NO} &\n\
\n\
echo "Waiting for server to start..."\n\
sleep 10\n\
\n\
echo "Starting benefits bot..."\n\
# Build the command with all configurable arguments\n\
CMD_ARGS="--chat_model_id ${CHAT_MODEL_ID} \\\n\
  --chatbot_strategy ${CHATBOT_STRATEGY} \\\n\
  --max_code_gen_attempts ${MAX_CODE_GEN_ATTEMPTS} \\\n\
  --max_code_rewrite_attempts ${MAX_CODE_REWRITE_ATTEMPTS} \\\n\
  --synthetic_user_model_name ${SYNTHETIC_USER_MODEL_NAME} \\\n\
  --max_dialog_turns ${MAX_DIALOG_TURNS} \\\n\
  --eligibility_requirements ${ELIGIBILITY_REQUIREMENTS} \\\n\
  --dataset_path ${DATASET_PATH} \\\n\
  --ds_shift ${DS_SHIFT} \\\n\
  --top_k ${TOP_K} \\\n\
  --estring ${ESTRING} \\\n\
  --use_cache ${USE_CACHE} \\\n\
  --random_seed ${RANDOM_SEED}"\n\
\n\
# Add optional arguments if they are set\n\
if [ ! -z "${CODE_MODEL_ID}" ]; then\n\
    CMD_ARGS="${CMD_ARGS} --code_model_id ${CODE_MODEL_ID}"\n\
fi\n\
\n\
if [ ! -z "${DOWNSAMPLE_SIZE}" ]; then\n\
    CMD_ARGS="${CMD_ARGS} --downsample_size ${DOWNSAMPLE_SIZE}"\n\
fi\n\
\n\
if [ ! -z "${NUM_PROGRAMS}" ]; then\n\
    CMD_ARGS="${CMD_ARGS} --num_programs ${NUM_PROGRAMS}"\n\
fi\n\
\n\
# Handle programs argument (convert space-separated string to array)\n\
if [ ! -z "${PROGRAMS}" ]; then\n\
    CMD_ARGS="${CMD_ARGS} --programs ${PROGRAMS}"\n\
fi\n\
\n\
echo "Running: python3 analysis/benefitsbot.py ${CMD_ARGS}"\n\
python3 analysis/benefitsbot.py ${CMD_ARGS}\n\
' > /app/start.sh && \
chmod +x /app/start.sh

EXPOSE 55221
CMD ["/app/start.sh"]
