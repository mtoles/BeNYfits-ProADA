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
import threading
import queue
import time
from dotenv import load_dotenv


from server_internals import ForwardRequest, forward_hf

"""
run with:
uvicorn server.model_server:app --reload
"""


load_dotenv()
LM_PORT_NO = int(os.getenv("LM_PORT_NO"))

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")


current_name_of_model = None
model = None
tk = None
# Create a queue and shared results dictionary
q = queue.Queue()
results = {}


# Define a worker function
def worker(q, results):
    while True:
        item, event = q.get()  # Get an item and its associated event
        if item is None:  # Exit signal
            break
        print(f"Processing item: {item}")
        # time.sleep(1)  # Simulate work

        # fr = ForwardRequest(item
        # )
        r = forward_hf(item)
        # Store the result (could be more complex in real cases)
        results[item.json()] = r
        event.set()  # Signal that the item has been processed
        q.task_done()  # Mark the task as done


num_threads = 1
threads = []
for i in range(num_threads):
    t = threading.Thread(target=worker, args=(q, results))
    t.start()
    threads.append(t)


# Function to process items and wait for them
def process_item(item):
    event = threading.Event()  # Create an event for this item
    q.put((item, event))  # Add the item and event to the queue
    print(f"Waiting for item {item} to be processed...")
    event.wait()  # Wait until the item is processed
    # Retrieve the result from the shared dictionary
    processed_item = results[item.json()]
    print(f"Item {item} has been processed with result: {processed_item}")
    return processed_item


item_threads = []


@app.post("/forward")
def forward(request: ForwardRequest):
    # try:
    #     print("at /forward")
    #     if request.name_of_model.startswith("gpt"):
    #         raise NotImplementedError  # gpt moved to client side
    #         # output = forward_gpt(request)
    #     else:
    #         output = forward_hf(request)
    #     return output
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=500, detail=f"traceback:\n{traceback.format_exc()}"
    #     )
    t = threading.Thread(target=process_item, args=(request,))
    # print(f"starting thread for {request}")
    t.start()
    item_threads.append(t)
    t.join()
    return results[request.json()]


if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=55244)
    server_thread = threading.Thread(
        target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": LM_PORT_NO}
    )
    server_thread.start()

    # call forward a few times in multiple threads
    def call_forward(i):
        x = forward(i)
        print(f">>>>>>>>> client received: {x}")

    threads = []
    for i in range(10):
        prompt = f"what is 10 + {i}"
        fr = ForwardRequest(
            name_of_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            history=[{"role": "user", "content": prompt}],
            use_cache=False,
            constraints=None,
            constraint_type="none",
            response_format=None,
        )
        t = threading.Thread(target=call_forward, args=(fr,))
        t.start()
        threads.append(t)
    print("all threads started")

    # wait for all threads to finish
    for t in threads:
        t.join()

    for i in range(100, 110):
        prompt = f"what is 10 + {i}"
        fr = ForwardRequest(
            name_of_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            history=[{"role": "user", "content": prompt}],
            use_cache=False,
            constraints=None,
            constraint_type="none",
            response_format=None,
        )
        t = threading.Thread(target=call_forward, args=(fr,))
        t.start()
        threads.append(t)

    print("all threads started")
