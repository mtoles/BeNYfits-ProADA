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
import os

load_dotenv(override=False)
os.environ['HF_HOME'] = '/local/data/rds_hf_cache'

"""
run with:
uvicorn server.model_server:app --reload
"""

memory = Memory(".joblib_cache", verbose=0)
app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
sampler = outlines.samplers.multinomial(temperature=0.7)

MODEL_STORE = {}
"""
MODEL_STORE structure:
{
    "<model_name>": {
        "model": <Transformers(...) object>,
        "tokenizer": <AutoTokenizer>,
        "device": 0,              # e.g. 0 means "cuda:0"
        "last_used": <timestamp>,
    },
    ...
}
"""

# Track GPU occupancy: which models are on each GPU
GPU_OCCUPANCY = {
    gpu_id: set() for gpu_id in range(torch.cuda.device_count())
}
"""
GPU_OCCUPANCY = {
  0: set(["modelA", "modelB"]),
  1: set(["modelC"]),
  ...
}
"""

INACTIVITY_TIMEOUT = 60 * 60  # e.g. 1 hour
REQUEST_IN_PROGRESS = False

model_lock = threading.Lock()

def watch_inactivity():
    """
    Unload models if they haven't been used in INACTIVITY_TIMEOUT seconds.
    """
    global MODEL_STORE, GPU_OCCUPANCY, REQUEST_IN_PROGRESS
    while True:
        time.sleep(INACTIVITY_TIMEOUT // 4)  # Check periodically
        now = time.time()

        with model_lock:
            if REQUEST_IN_PROGRESS:
                # If there's an active request, skip immediate unloading cycle
                continue

            to_remove = []
            for model_name, info in MODEL_STORE.items():
                if (now - info["last_used"]) > INACTIVITY_TIMEOUT:
                    to_remove.append(model_name)

            for model_name in to_remove:
                print(f"[Inactivity Watcher] Unloading model {model_name} (inactive).")
                try:
                    # Remove from memory
                    device_id = MODEL_STORE[model_name]["device"]
                    del MODEL_STORE[model_name]["model"]
                    del MODEL_STORE[model_name]["tokenizer"]
                    del MODEL_STORE[model_name]
                    # Remove from GPU occupancy
                    if model_name in GPU_OCCUPANCY[device_id]:
                        GPU_OCCUPANCY[device_id].remove(model_name)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error unloading {model_name}: {e}")

        print("[Inactivity Watcher] Cycle complete.")


threading.Thread(target=watch_inactivity, daemon=True).start()

class ForwardRequest(BaseModel):
    name_of_model: str
    history: list[dict]
    use_cache: bool
    constraints: Optional[Union[BaseModel, list[str], str]]
    constraint_type: Optional[str]
    response_format: Any
    random_seed: int

def _str_to_type(s):
    if s == "int":
        return int
    elif s == "float":
        return float
    else:
        raise NotImplementedError(f"Type {s} not supported.")

@memory.cache
def forward_hf(request: ForwardRequest):
    """
    The main text-generation function using Outlines & HuggingFace models.
    """
    
    name_of_model = request.name_of_model
    history = request.history

    # Determine constraints
    if request.constraint_type == "types":
        constraints = [_str_to_type(x) for x in request.constraints]
    elif request.constraint_type == "choice":
        constraints = request.constraints
    elif request.constraint_type == "regex":
        constraints = request.constraints
    elif request.constraint_type == "none":
        constraints = None
    else:
        raise NotImplementedError(f"Unknown constraint type: {request.constraint_type}")

    print(f"[{name_of_model}] History: {history}")
    print(f"[{name_of_model}] Constraints: {constraints} (type={request.constraint_type})")
    print(f"GPU Occupancy: {GPU_OCCUPANCY}")
    
    # --- Acquire lock to safely load or fetch model ---
    with model_lock:
        if name_of_model in MODEL_STORE:
            # Model is already loaded
            model_info = MODEL_STORE[name_of_model]
            print(f"[{name_of_model}] Already loaded on GPU {model_info['device']}")
            model_info["last_used"] = time.time()
            model_obj = model_info["model"]
            tokenizer = model_info["tokenizer"]
        else:
            # Model not yet loaded -> choose a GPU according to the logic:
            # "If there is an available GPU with no model loaded, then load
            #  current model on that GPU, else load model on any available GPU."
            if torch.cuda.device_count() == 0:
                # If no GPU, fallback to CPU
                chosen_gpu = None
                device_map = {"": "cpu"}
            else:
                # Find GPUs with zero models
                free_gpus = [g for g, occupant in GPU_OCCUPANCY.items() if len(occupant) == 0]
                if free_gpus:
                    chosen_gpu = free_gpus[0]
                    print(f"[{name_of_model}] Found free GPU {chosen_gpu} (no models).")
                else:
                    # No GPU is empty, pick any GPU. For example:
                    # chosen_gpu = 0
                    # or pick the GPU with the fewest models:
                    chosen_gpu = min(GPU_OCCUPANCY.keys(), key=lambda g: len(GPU_OCCUPANCY[g]))
                    print(f"[{name_of_model}] All GPUs occupied. Using GPU {chosen_gpu}.")

                device_map = {"": f"cuda:{chosen_gpu}"}

            try:
                print(f"[{name_of_model}] Loading model onto device_map={device_map} ...")
                tokenizer = AutoTokenizer.from_pretrained(name_of_model, use_fast=False)
                raw_model = AutoModelForCausalLM.from_pretrained(
                    name_of_model,
                    torch_dtype=torch.bfloat16,
                    load_in_4bit=True,
                    device_map=device_map
                )
                model_obj = outlines.models.Transformers(raw_model, tokenizer)

                # Update global structures
                MODEL_STORE[name_of_model] = {
                    "model": model_obj,
                    "tokenizer": tokenizer,
                    "device": chosen_gpu if chosen_gpu is not None else -1,
                    "last_used": time.time()
                }
                if chosen_gpu is not None:
                    GPU_OCCUPANCY[chosen_gpu].add(name_of_model)

            except Exception as e:
                print(traceback.format_exc())
                raise HTTPException(
                    status_code=500,
                    detail=f"Error loading model '{name_of_model}': {e}"
                )

    # --- Build prompt and generate ---
    try:
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Build generator according to constraints
        if not constraints or request.constraint_type == "none":
            generator = outlines.generate.text(model_obj, sampler=sampler)
        elif request.constraint_type == "choice":
            generator = outlines.generate.choice(model_obj, constraints, sampler=sampler)
        elif request.constraint_type == "types":
            # Typically expect a single Python type in constraints for "types"
            assert len(constraints) == 1, "Types constraint must have exactly one type."
            generator = outlines.generate.format(model_obj, constraints[0], sampler=sampler)
        elif request.constraint_type == "regex":
            generator = outlines.generate.regex(model_obj, constraints, sampler=sampler)
        else:
            raise NotImplementedError(f"Constraint type {request.constraint_type} not supported.")

        generated_text = str(generator(prompt)).strip()
        print(f"[{name_of_model}] Generated text: {generated_text}")

        return {"generated_text": generated_text}

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text: {e}"
        )
    
@app.post("/forward")
def forward(request: ForwardRequest):
    """
    Endpoint that handles generation requests.
    """
    global REQUEST_IN_PROGRESS
    REQUEST_IN_PROGRESS = True
    try:
        # Example policy to forbid GPT models server-side
        # (Remove if not needed)
        assert not request.name_of_model.startswith("gpt"), "GPT models are client side only."

        return forward_hf(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"traceback:\n{traceback.format_exc()}"
        )
    finally:
        REQUEST_IN_PROGRESS = False


if __name__ == "__main__":
    load_dotenv()
    print(f"GPU Occupancy: {GPU_OCCUPANCY}")
    # port = int(os.getenv("LM_PORT_NO"))
    # url = os.getenv("LM_SERVER_URL")
    port = 8000
    url = "localhost"
    uvicorn.run(app, host=url, port=port)
