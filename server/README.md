# Model Hosting Server

Steps to Start the Model Hosting:
1. `cd server`
2. Activate the required python environment with dependencies.
3. Start the service: `uvicorn model_server:app --host 0.0.0.0 --port 8000`