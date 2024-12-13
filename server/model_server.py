from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import openai
import os
from fastapi import FastAPI, HTTPException
from joblib import Memory
from typing import Union, Optional
import outlines
import ast
import traceback

"""
run with:
uvicorn server.model_server:app --reload
"""

memory = Memory(".joblib_cache", verbose=0)

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")


class ForwardRequest(BaseModel):
    name_of_model: str
    history: list[dict]
    use_cache: bool
    constraints: Optional[Union[list[str], str]]
    constraint_type: Optional[str]


current_name_of_model = None
model = None
tk = None


def _str_to_type(s):
    if s == "int":
        return int
    elif s == "float":
        return float
    else:
        raise NotImplementedError


@memory.cache
def forward_hf(request: ForwardRequest):
    global current_name_of_model, model, tk
    name_of_model = request.name_of_model
    history = request.history
    print(f"hf Received: {history}")

    # constraint = None
    # if request.constraints == "int":
    #     constraint = int
    # elif request.constraints == "float":
    #     constraint = float
    # elif type(request.constraints) == list:
    #     constraint = request.constraints
    # elif request.constraints is None:
    #     constraint = None
    # else:
    # #     constraint = ast.literal_eval(request.constraints)  # list of options
    #     raise NotImplementedError
    #     # assert type(constraint) == list
    #     # assert set([type(x) == str for x in constraint]) == set([str])

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
            del model
            del tk
            torch.cuda.empty_cache()
        try:
            tk = AutoTokenizer.from_pretrained(name_of_model)
            model = outlines.models.transformers(name_of_model, device="cuda:0")
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
        if request.constraint_type == "none":
            generator = outlines.generate.text(model)
        elif request.constraint_type == "choice":
            generator = outlines.generate.choice(model, request.constraints)
        elif request.constraint_type == "types":
            assert len(constraints) == 1
            generator = outlines.generate.format(model, constraints[0])
        elif request.constraint_type == "regex":
            generator = outlines.generate.regex(model, request.constraints)
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


@memory.cache
def forward_gpt(request: ForwardRequest):
    try:
        response = openai.ChatCompletion.create(
            model=request.name_of_model,
            messages=request.history,
            max_tokens=50,
            temperature=0.7,
        )
        generated_text = response.choices[0].message["content"].strip()
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forward")
def forward(request: ForwardRequest):
    try:
        print("at /forward")
        if request.name_of_model.startswith("gpt"):
            output = forward_gpt(request)
        else:
            output = forward_hf(request)
        return output
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"traceback:\n{traceback.format_exc()}"
        )


if __name__ == "__main__":
    # if False:
    # name_of_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    name_of_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    history = [
        {
            "role": "system",
            "content": "You are a chatbot. Respond to the question:",
        },
        {
            "role": "user",
            "content": "How many words are in the sentence 'Hello World'?",
        },
    ]

    output = forward(
        ForwardRequest(
            name_of_model=name_of_model,
            history=history,
            use_cache=False,
            constraints=None,
        )
    )

    print(output)

    output = forward(
        ForwardRequest(
            name_of_model=name_of_model,
            history=history,
            use_cache=False,
            constraints=["1", "2", "3"],
        )
    )
    print(output)
    print
