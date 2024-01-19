# Setup Instructions

## Python Environment Setup
Install Python libraries with

```
pip install -r requirements.txt
```

Install local modules with 
```
pip install -e .
```

## OpenAI Credential Setup
Create a file called `.env` containing

```
OPENAI_API_KEY=abc123
HUGGINGFACE_API_KEY=def456
```
## Set up Google BigQuery
Follow instructions here https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to
When installing the gcloud CLI, https://cloud.google.com/sdk/docs/install#linux , hen prompted, select `[3] Enter a project ID` and enter `lunar-carving-267205` 

## Download Data
`python3 get_data.py` for normal reddit data or `python3 get_data.py --tldr` for the tldr dataset