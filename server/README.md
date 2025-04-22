# Model Hosting Server

Steps to Start the Model Hosting:
1. `cd server`
2. Activate the required python environment with dependencies.
3. Start the service: `uvicorn model_server:app --host 0.0.0.0 --port 8000`


# Run OUR model (Group A)

replace programs with your programs
```
LM_SERVER_URL=http://127.0.0.1 \
LM_PORT_NO=55221 \
python3 analysis/benefitsbot.py \
  --chat_model_id meta-llama/Meta-Llama-3.1-70B-Instruct \
  --code_model_id gpt-4o-2024-08-06 \
  --chatbot_strategy codebot \
  --max_code_gen_attempts 3 \
  --synthetic_user_model_name human \
  --programs EarlyHeadStart InfantToddlerPrograms HeadStart ComprehensiveAfterSchoolSystemOfNYC PreKForAll \ # YOUR PROGRAMS GO HERE
  --estring human-evals \
  --dataset_path dataset/user_study_dataset.jsonl
```

# Run the BASELINE model (Group B)
```
LM_SERVER_URL=http://127.0.0.1 \
LM_PORT_NO=55221 \
python3 analysis/benefitsbot.py \
  --chat_model_id gpt-4o-2024-08-06 \
  --chatbot_strategy cot \
  --synthetic_user_model_name human \
  --programs EarlyHeadStart InfantToddlerPrograms HeadStart ComprehensiveAfterSchoolSystemOfNYC PreKForAll \ # YOUR PROGRAMS GO HERE
  --estring human-evals \
  --dataset_path dataset/user_study_dataset.jsonl
```