from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
import queue  # <--- For the per-model queues

load_dotenv(override=False)

"""
Run with:
    CUDA_VISIBLE_DEVICES=0 uvicorn server.concurrent_multiple_model_server:app --port XXXXX
"""

memory = Memory(".joblib_cache", verbose=0)
app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

sampler = outlines.samplers.multinomial(temperature=0.7)

# ---------------------------
# Global storages
# ---------------------------

MODEL_STORE = {}
"""
MODEL_STORE structure:
{
    "<model_name>": {
        "model": <outlines.models.Transformers(...)>,
        "tokenizer": <AutoTokenizer>,
        "device": 0,        # e.g. 0 means "cuda:0"
        "last_used": time.time(),
    },
    ...
}
"""

GPU_OCCUPANCY = {
    gpu_id: set() for gpu_id in range(torch.cuda.device_count())
}

INACTIVITY_TIMEOUT = 60 * 60  # 1 hour

# One lock to protect the actual model_store loading/unloading
model_store_lock = threading.Lock()

# Create a queue per model name when needed
MODEL_QUEUES = {}   # model_name -> queue.Queue
MODEL_WORKERS = {}  # model_name -> threading.Thread

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

    # Handle constraints
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

    model_obj, tokenizer = load_model_if_needed(name_of_model)

    # Build prompt and generate
    try:
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        if not constraints or request.constraint_type == "none":
            generator = outlines.generate.text(model_obj, sampler=sampler)
        elif request.constraint_type == "choice":
            generator = outlines.generate.choice(model_obj, constraints, sampler=sampler)
        elif request.constraint_type == "types":
            # Typically expect single type
            assert len(constraints) == 1, "For 'types' constraint, provide exactly one type."
            generator = outlines.generate.format(model_obj, constraints[0], sampler=sampler)
        elif request.constraint_type == "regex":
            generator = outlines.generate.regex(model_obj, constraints, sampler=sampler)
        else:
            raise NotImplementedError(f"Constraint type {request.constraint_type} not supported.")

        generated_text = str(generator(prompt)).strip()
        print(f"[{name_of_model}] Generated text: {generated_text}")

        # Update last_used
        with model_store_lock:
            MODEL_STORE[name_of_model]["last_used"] = time.time()

        return {"generated_text": generated_text}

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text: {e}"
        )

def load_model_if_needed(model_name: str):
    """
    Ensure the model is loaded into MODEL_STORE. Return (model_obj, tokenizer).
    This function is concurrency-safe by using `model_store_lock`.
    """
    with model_store_lock:
        if model_name in MODEL_STORE:
            # Already loaded
            model_info = MODEL_STORE[model_name]
            return model_info["model"], model_info["tokenizer"]
        else:
            # Load the model
            if torch.cuda.device_count() == 0:
                # If no GPU, fallback to CPU
                chosen_gpu = None
                device_map = {"": "cpu"}
            else:
                # Find free GPU (0 models). If none, pick GPU with fewest models.
                free_gpus = [g for g, occupant in GPU_OCCUPANCY.items() if len(occupant) == 0]
                if free_gpus:
                    chosen_gpu = free_gpus[0]
                    print(f"[{model_name}] Found free GPU {chosen_gpu} (no models).")
                else:
                    chosen_gpu = min(GPU_OCCUPANCY.keys(), key=lambda g: len(GPU_OCCUPANCY[g]))
                    print(f"[{model_name}] All GPUs occupied. Using GPU {chosen_gpu}.")

                device_map = {"": f"cuda:{chosen_gpu}"}

            try:
                print(f"[{model_name}] Loading model onto device_map={device_map} ...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                raw_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    load_in_4bit=True,
                    device_map=device_map
                )
                model_obj = outlines.models.Transformers(raw_model, tokenizer)

                # Update global structures
                MODEL_STORE[model_name] = {
                    "model": model_obj,
                    "tokenizer": tokenizer,
                    "device": chosen_gpu if chosen_gpu is not None else -1,
                    "last_used": time.time()
                }
                if chosen_gpu is not None:
                    GPU_OCCUPANCY[chosen_gpu].add(model_name)

                return model_obj, tokenizer

            except Exception as e:
                print(traceback.format_exc())
                raise HTTPException(
                    status_code=500,
                    detail=f"Error loading model '{model_name}': {e}"
                )

#
# Worker / queue design
#

def model_worker(model_name: str):
    """
    A dedicated worker that processes requests from the queue for `model_name`.
    Only one request is processed at a time per model.
    """
    q = MODEL_QUEUES[model_name]
    while True:
        job = q.get()
        if job is None:
            # If we receive `None`, it’s a signal to shut down
            q.task_done()
            break

        (request_obj, done_event, result_dict) = job
        
        try:
            output = forward_hf(request_obj)
            result_dict["result"] = output
        except Exception as ex:
            # Store the exception text so the main thread can raise it
            result_dict["exception"] = ex

        # Signal that we’re done
        done_event.set()
        q.task_done()

def start_model_worker(model_name: str):
    """
    If there's no worker thread for this model yet, create one.
    """
    if model_name not in MODEL_QUEUES:
        MODEL_QUEUES[model_name] = queue.Queue()

    if model_name not in MODEL_WORKERS:
        th = threading.Thread(target=model_worker, args=(model_name,), daemon=True)
        MODEL_WORKERS[model_name] = th
        th.start()

#
# The inactivity watcher
#

def watch_inactivity():
    global MODEL_STORE, GPU_OCCUPANCY
    while True:
        time.sleep(INACTIVITY_TIMEOUT // 4)
        now = time.time()

        # Attempt to unload inactive models
        with model_store_lock:
            to_remove = []
            for model_name, info in MODEL_STORE.items():
                if (now - info["last_used"]) > INACTIVITY_TIMEOUT:
                    to_remove.append(model_name)

            for model_name in to_remove:
                print(f"[Inactivity Watcher] Unloading model {model_name} (inactive).")
                try:
                    device_id = MODEL_STORE[model_name]["device"]
                    # Remove from GPU occupancy
                    if device_id >= 0 and model_name in GPU_OCCUPANCY[device_id]:
                        GPU_OCCUPANCY[device_id].remove(model_name)

                    # Free memory
                    del MODEL_STORE[model_name]["model"]
                    del MODEL_STORE[model_name]["tokenizer"]
                    del MODEL_STORE[model_name]
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error unloading {model_name}: {e}")

        print("[Inactivity Watcher] Cycle complete.")

threading.Thread(target=watch_inactivity, daemon=True).start()

#
# FastAPI endpoint
#

@app.post("/forward")
def forward(request: ForwardRequest):
    """
    Endpoint that handles generation requests via queue-based model concurrency.
    """
    # Example policy: forbid GPT names
    if request.name_of_model.startswith("gpt"):
        raise HTTPException(status_code=400, detail="GPT models are client side only.")

    model_name = request.name_of_model

    # 1) Ensure we have a worker thread and a queue for this model
    start_model_worker(model_name)

    # 2) Create a job with an Event to wait for completion
    done_event = threading.Event()
    result_holder = {}
    job = (request, done_event, result_holder)

    # 3) Put the job in the model's queue
    MODEL_QUEUES[model_name].put(job)

    # 4) Block until job is done
    done_event.wait()

    # 5) Check for exceptions
    if "exception" in result_holder:
        ex = result_holder["exception"]
        raise HTTPException(
            status_code=500,
            detail=f"Error during generation: {ex}"
        )
    
    # 6) Return result
    return result_holder["result"]


if __name__ == "__main__":
    port = int(os.getenv("LM_PORT_NO", "8000"))
    url = os.getenv("LM_SERVER_URL", "0.0.0.0")
    uvicorn.run(app, host=url, port=port)
