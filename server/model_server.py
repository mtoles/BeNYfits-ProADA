from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.openai_wrapper import get_open_ai_lm
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
from fastapi.encoders import jsonable_encoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import traceback
from transformers import GPT2TokenizerFast

class PromptInput(BaseModel):
    text: str
    cache: bool = True
    logprobs: int = 0
    max_tokens: int = 50


class PredictManyRequest(BaseModel):
    prompts: List[PromptInput]
    model_id: str


app = FastAPI()


class ModelUnwrapped:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # self.model._tokenizer.pad_token_id = self.model._tokenizer.eos_token_id
        # self.model._tokenizer.padding_side = "left"

    def predict_many(self, lm_prompts: list[LmPrompt], completion_window):
        print(lm_prompts)
        messages = [{"role": "user", "content": prompt.text} for prompt in lm_prompts]

        input = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(input, max_new_tokens=512, do_sample=False)

        result = self.tokenizer.decode(
            output[0][len(input[0]) :], skip_special_tokens=True
        )
        return [result]


# Model container
class ModelServer:
    def __init__(self):
        self.models = {}

    def load_model(self, family, hf_name, wrapped=True):
        if wrapped:
            if hf_name not in self.models:
                if family in ["llama", "gemma"]:
                    model = get_huggingface_lm(hf_name)
                    model._tokenizer.pad_token_id = model._tokenizer.eos_token_id
                    model._tokenizer.padding_side = "left"
                else:
                    model = get_open_ai_lm(hf_name)
                    model._tokenizer = (GPT2TokenizerFast.from_pretrained("Xenova/gpt-3.5-turbo"))
                self.models[hf_name] = model
                print(f"Model Pipeline Instantiated: {family} {hf_name}")
        else:
            if hf_name not in self.models:
                print("loading unwrapped model")
                self.models[hf_name] = ModelUnwrapped(hf_name)
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
    wrapped: bool


@app.post("/load_model")
def load_model(request: LoadModelRequest):
    print(f"Received request: {request}")  # Debugging step
    try:
        result = model_server.load_model(
            request.family, request.hf_name, request.wrapped
        )

        print(f"Model loaded successfully: {result}")
        
        return {"message": "Model loaded successfully!", "model": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define request schema for predict_many


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
        sequences = list(
            model.predict_many(lm_prompts, completion_window=CompletionWindow.ASAP)
        )
        print(sequences)
        sequence = sequences[0]
        if type(sequence) != str:
            output = sequence.completion_text
        else:
            output = sequence

        print(f"Results of Predict Many: {output}")

        # Convert results to JSON-compatible format if needed
        # responses = [
        #     jsonable_encoder(result) for result in results
        # ]
        return {"model_id": request.model_id, "responses": output}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        print(traceback.format_exc())  # Print the entire stack trace
        raise HTTPException(status_code=500, detail=str(e))


class ChatHistoryInput(BaseModel):
    history: list[dict]
    model_id: str


@app.post("/apply_chat_template")
def apply_chat_template(request: ChatHistoryInput):
    print(f"Request Received: {request}")

    try:
        model = model_server.get_model(request.model_id)

        history = request.history

        # convert the first n-1 dictionaries in the history to a single string
        if len(history) > 1:
            history_str = "\n".join(
                [f"{turn['role']}:{turn['content']}" for turn in history[:-1]]
            )
            history = [
                {"role": "system", "content": history_str},
                history[-1],
            ]
        history[-1]["role"] = "user"

        output = model._tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )

        print(f"Chat Template Applied: {output}")

        return {"formatted_chat": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
