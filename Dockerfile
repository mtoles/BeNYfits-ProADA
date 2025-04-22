FROM nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy .env file
COPY .env .

# Install the package
RUN pip3 install -e .

# Set environment variables from .env file
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV HF_TOKEN=${HF_TOKEN}
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
ENV LM_PORT_NO=${LM_PORT_NO}
ENV LM_SERVER_URL=${LM_SERVER_URL}

# Create startup script
RUN echo '#!/bin/bash\n\
huggingface-cli login --token "${HF_TOKEN}"\n\
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uvicorn server.concurrent_multiple_model_server:app --port ${LM_PORT_NO} &\n\
sleep 5\n\
python3 analysis/benefitsbot.py --chat_model_id meta-llama/Meta-Llama-3.1-70B-Instruct --code_model_id gpt-4o-2024-08-06 --chatbot_strategy cot --max_code_gen_attempts 3 --synthetic_user_model_name meta-llama/Meta-Llama-3.1-70B-Instruct --programs EarlyHeadStart InfantToddlerPrograms HeadStart ComprehensiveAfterSchoolSystemOfNYC PreKForAll --estring debug --dataset_path dataset/user_study_dataset.jsonl\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose the port
EXPOSE 55221

# Run the startup script
CMD ["/app/start.sh"]