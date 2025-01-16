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


from server.server_internals import ForwardRequest, forward_hf

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

BATCH_SIZE = 4

# Define a worker function
# def worker(q, results):
#     while True:
#         items = []
#         try:
#             item, event = q.get()  # Get an item and its associated event
#             if item is None:  # Exit signal
#                 continue
#         except queue.Empty:
#             continue
        
#         print(f"Processing item: {item}")
#         # time.sleep(1)  # Simulate work
#         try:
#             r = forward_hf(items)
#         except Exception as e:
#             print(traceback.format_exc())
#             print("Error while processing items:", e)
#             results[item.json()] = str(e)
#         else:
#             results[item.json()] = r
#         event.set()  # Signal that the item has been processed
#         q.task_done()  # Mark the task as done
def process_batch(items_with_events, results):
    """Helper to call forward_hf on the batch and record results."""
    items = [ie[0] for ie in items_with_events]
    events = [ie[1] for ie in items_with_events]
    try:
        outputs = forward_hf(items)  # Assume this returns a list of results
        for item, event, out in zip(items, events, outputs):
            results[item.json()] = out
            event.set()
    except Exception as e:
        traceback.print_exc()
        for item, event in items_with_events:
            results[item.json()] = str(e)
            event.set()

def worker(q, results):
    batch = []
    while True:
        try:
            item, event = q.get()
            if item is None:  # Signal to flush and exit
                if batch:
                    process_batch(batch, results)
                break
            batch.append((item, event))
            if len(batch) == BATCH_SIZE:
                process_batch(batch, results)
                batch.clear()
        except queue.Empty:
            pass  # Continue waiting

    # In case anything remains
    if batch:
        process_batch(batch, results)

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
