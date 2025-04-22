# FROM nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu22.04
# WORKDIR /app

# # Install Python and other dependencies
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first and install dependencies
# COPY requirements.txt .
# RUN pip3 install -r requirements.txt

# # Copy the rest of the application
# COPY . .

# # Copy .env file
# COPY .env .

# # Install the package
# RUN pip3 install -e .

# # Set environment variables from .env file
# ENV OPENAI_API_KEY=${OPENAI_API_KEY}
# ENV HF_TOKEN=${HF_TOKEN}
# ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# ENV LM_PORT_NO=${LM_PORT_NO}
# ENV LM_SERVER_URL=${LM_SERVER_URL}

# # Create startup script
# RUN echo '#!/bin/bash\n\
# huggingface-cli login --token "${HF_TOKEN}"\n\
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uvicorn server.concurrent_multiple_model_server:app --port ${LM_PORT_NO} &\n\
# sleep 5\n\
# python3 analysis/benefitsbot.py --chat_model_id meta-llama/Meta-Llama-3.1-70B-Instruct --code_model_id gpt-4o-2024-08-06 --chatbot_strategy cot --max_code_gen_attempts 3 --synthetic_user_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --programs EarlyHeadStart InfantToddlerPrograms HeadStart ComprehensiveAfterSchoolSystemOfNYC PreKForAll --estring debug --dataset_path dataset/user_study_dataset.jsonl\n\
# ' > /app/start.sh && chmod +x /app/start.sh

# # Expose the port
# EXPOSE 55221

# # Run the startup script
# CMD ["/app/start.sh"]

# docker build -t benefitsbot .
# docker run --gpus all -p 55221:55221 --env-file .env benefitsbot

# 1) CUDA runtime + drivers included
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2) System & Python deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# 3) Install Python requirements
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 4) Copy source & .env, install your package
COPY . .
RUN pip3 install --no-cache-dir -e .

# 5) Pass through your env‑vars
ENV OPENAI_API_KEY=${OPENAI_API_KEY} \
    HF_TOKEN=${HF_TOKEN} \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    LM_PORT_NO=${LM_PORT_NO} \
    LM_SERVER_URL=${LM_SERVER_URL}

# 6) Startup script
RUN echo '#!/usr/bin/env bash\n\
set -e\n\
# login & launch model server\n\
huggingface-cli login --token "${HF_TOKEN}"\n\
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \\\n\
  uvicorn server.concurrent_multiple_model_server:app --port ${LM_PORT_NO} &\n\
sleep 5\n\
# then run your bot\n\
python3 analysis/benefitsbot.py \\\n\
  --chat_model_id meta-llama/Meta-Llama-3.1-70B-Instruct \\\n\
  --code_model_id gpt-4o-2024-08-06 \\\n\
  --chatbot_strategy cot \\\n\
  --max_code_gen_attempts 3 \\\n\
  --synthetic_user_model_name meta-llama/Meta-Llama-3.1-70B-Instruct \\\n\
  --programs EarlyHeadStart InfantToddlerPrograms HeadStart ComprehensiveAfterSchoolSystemOfNYC PreKForAll \\\n\
  --estring debug \\\n\
  --dataset_path dataset/user_study_dataset.jsonl' > /app/start.sh && \
chmod +x /app/start.sh

EXPOSE 55221
CMD ["/app/start.sh"]
