export CUDA_VISIBLE_DEVICES=0
LM_PORT_NO=55221
uvicorn server.concurrent_multiple_model_server:app --port ${LM_PORT_NO}