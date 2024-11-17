from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.openai_wrapper import get_open_ai_lm
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
from fastapi.encoders import jsonable_encoder

app = FastAPI()

# Model container
class ModelServer:
    def __init__(self):
        self.models = {}

    def load_model(self, family, hf_name):
        if hf_name not in self.models:
            if family in ["llama", "gemma"]:
                model = get_huggingface_lm(hf_name)
                model._tokenizer.pad_token_id = model._tokenizer.eos_token_id
                model._tokenizer.padding_side = "left"
            else:
                model = get_open_ai_lm(hf_name)
            self.models[hf_name] = model
            print(f"Model Pipeline Instantiated: {family} {hf_name}")
        return hf_name

    def get_model(self, model_id):
        if model_id in self.models:
            return self.models[model_id]
        else:
            raise ValueError(f"Model with ID '{model_id}' not found!")

model_server = ModelServer()

class LoadModelRequest(BaseModel):
    family: str
    hf_name: str

@app.post("/load_model")
def load_model(request: LoadModelRequest):
    print(f"Received request: {request}")  # Debugging step
    try:
        result = model_server.load_model(request.family, request.hf_name)
        return {"message": "Model loaded successfully!", "model": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define request schema for predict_many
class PromptInput(BaseModel):
    text: str
    cache: bool = True
    logprobs: int = 0
    max_tokens: int = 50

class PredictManyRequest(BaseModel):
    prompts: List[PromptInput]
    model_id: str

@app.post("/predict_many")
def predict_many(request: PredictManyRequest):
    print(f"Request Received: {request}")

    try:
        model = model_server.get_model(request.model_id)
        lm_prompts = [
            LmPrompt(
                prompt.text,
                cache=prompt.cache,
                logprobs=prompt.logprobs,
                max_tokens=prompt.max_tokens,
            )
            for prompt in request.prompts
        ]
        sequences = list(model.predict_many(lm_prompts, completion_window=CompletionWindow.ASAP))

        sequence = sequences[0]
        output = sequence.completion_text

        print(f"Results of Predict Many: {output}")

        # Convert results to JSON-compatible format if needed
        # responses = [
        #     jsonable_encoder(result) for result in results
        # ]
        return {"model_id": request.model_id, "responses": output}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))