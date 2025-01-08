from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import openai
import os
from fastapi import FastAPI, HTTPException
from joblib import Memory
from typing import Union, Optional, Any
import outlines
import ast
import traceback
from openai import OpenAI
import uvicorn

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
    constraints: Optional[Union[BaseModel, list[str], str]]
    # constraints: Optional[Union[list[str], str]]
    constraint_type: Optional[str]
    response_format: Any


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
            print("CUDA devices available:")
            for i in range(torch.cuda.device_count()):
                print(f"- {torch.cuda.get_device_name(i)}")
            # model = outlines.models.transformers(name_of_model, device="cuda", kwargs={"torch_dtype": torch.bfloat16})
            raw_model = AutoModelForCausalLM.from_pretrained(
                name_of_model, torch_dtype=torch.bfloat16
            ).to("cuda")
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


@app.post("/forward")
def forward(request: ForwardRequest):
    try:
        print("at /forward")
        if request.name_of_model.startswith("gpt"):
            raise NotImplementedError  # gpt moved to client side
            # output = forward_gpt(request)
        else:
            output = forward_hf(request)
        return output
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"traceback:\n{traceback.format_exc()}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=55244)