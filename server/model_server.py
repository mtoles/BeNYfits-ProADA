from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import openai
import os
from fastapi import FastAPI, HTTPException
from joblib import Memory
from typing import Union, Optional, Any
import outlines
import traceback
import uvicorn
from dotenv import load_dotenv
import time
import threading
import gc

load_dotenv(override=False)

"""
run with:
uvicorn server.model_server:app --reload
"""

memory = Memory(".joblib_cache", verbose=0)

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

sampler = outlines.samplers.multinomial(temperature=0.7)

last_request_time = time.time()
request_in_progress = False


### GLOBALS ###
current_name_of_model = None
model = None
raw_model = None
tk = None

def watch_inactivity():
    T = 60*60 # 1 hour
    # T = 30
    global model, tk, current_name_of_model, raw_model
    while True:
        time.sleep(T/4)  
        if time.time() - last_request_time > T and not request_in_progress:
            print("flushing model")
            # flush model
            if model is not None:
                del model
                model = None
            if tk is not None:
                del tk
                tk = None
            if raw_model is not None:
                del raw_model
                raw_model = None
            gc.collect()
            torch.cuda.empty_cache()
            current_name_of_model = None
        else:
            print("model preserved")

threading.Thread(target=watch_inactivity, daemon=True).start()

class ForwardRequest(BaseModel):
    name_of_model: str
    history: list[dict]
    use_cache: bool
    constraints: Optional[Union[BaseModel, list[str], str]]
    # constraints: Optional[Union[list[str], str]]
    constraint_type: Optional[str]
    response_format: Any
    random_seed: int
    # prefix: Optional[list[dict]]





def _str_to_type(s):
    if s == "int":
        return int
    elif s == "float":
        return float
    else:
        raise NotImplementedError


@memory.cache
def forward_hf(request: ForwardRequest):
    global current_name_of_model, model, tk, raw_model
    name_of_model = request.name_of_model
    history = request.history
    assert history[-1]["role"] == "user"
    print(f"hf Received: {history}")
    if request.constraint_type == "types":
        constraints = [_str_to_type(x) for x in request.constraints]
    elif request.constraint_type == "choice":
        constraints = request.constraints
    elif request.constraint_type == "regex":
        constraints = request.constraints
    elif request.constraint_type == "none":
        constraints = None
    else:
        raise NotImplementedError
    print(f"constraints (input): {request.constraints}")
    print(f"constraint_type: {request.constraint_type}")
    print(f"constraints: {constraints}")

    if name_of_model != current_name_of_model:
        if model is not None:
            model = None
            tk = None
            torch.cuda.empty_cache()
        try:
            tk = AutoTokenizer.from_pretrained(name_of_model)
            print("CUDA devices available:")
            for i in range(torch.cuda.device_count()):
                print(f"- {torch.cuda.get_device_name(i)}")
            # model = outlines.models.transformers(name_of_model, device="cuda", kwargs={"torch_dtype": torch.bfloat16})
            raw_model = AutoModelForCausalLM.from_pretrained(
                name_of_model,
                torch_dtype=torch.bfloat16,
                load_in_4bit=True,
                device_map={"": "cuda:0"},
            )
            # if torch.cuda.device_count() > 1:
            # raw_model = torch.nn.DataParallel(raw_model)
            model = outlines.models.Transformers(raw_model, tk)
            current_name_of_model = name_of_model
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            raise HTTPException(
                status_code=500, detail=f"Error loading model '{name_of_model}': {e}"
            )
    try:
        prompt = tk.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"prompt: {prompt}")
        if request.constraint_type == "none":
            generator = outlines.generate.text(model, sampler=sampler)
        elif request.constraint_type == "choice":
            generator = outlines.generate.choice(
                model, request.constraints, sampler=sampler
            )
        elif request.constraint_type == "types":
            assert len(constraints) == 1
            generator = outlines.generate.format(model, constraints[0], sampler=sampler)
        elif request.constraint_type == "regex":
            generator = outlines.generate.regex(
                model, request.constraints, sampler=sampler
            )
        else:
            print(f"Unknown constraints: {request.constraints}")
            raise NotImplementedError
        generated_text = str(generator(prompt)).strip()
        print(f"hf Generated: {generated_text}")
        return {"generated_text": generated_text}
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text: {e} in line {traceback.format_exc()}",
        )


@app.post("/forward")
def forward(request: ForwardRequest):
    global last_request_time, request_in_progress
    request_in_progress = True
    last_request_time = time.time()  # update on every request
    try:
        assert not request.name_of_model.startswith("gpt"), "gpt moved to client side"
        output = forward_hf(request)
        return output
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"traceback:\n{traceback.format_exc()}"
        )
    finally:
        request_in_progress = False


if __name__ == "__main__":
    load_dotenv()
    port = int(os.getenv("LM_PORT_NO"))
    url = os.getenv("LM_SERVER_URL")
    uvicorn.run(app, host=url, port=port)