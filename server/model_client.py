from server.model_server import ForwardRequest
import requests
from enum import Enum
from typing import Union, Optional
from openai import OpenAI, NotGiven
from anthropic import Anthropic
import anthropic
from joblib import Memory
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json


class Options(BaseModel):
    options: list[str]


class ResponseFormat(Enum):
    options = "options"
    none = None


memory = Memory(".joblib_cache", verbose=0)


port = os.getenv("LM_PORT_NO")  # Read 'PORT' environment variable
url = os.getenv("LM_SERVER_URL")


@memory.cache
def gpt_forward_cached(name_of_model, history, response_format):

    client = OpenAI()

    temperature = 0.7
    if name_of_model.startswith("o1") or name_of_model.startswith("o3"):
        response_format = None
        temperature = 1
    # if response_format is None:
    # completion = client.beta.chat.completions.parse(
    completion = client.chat.completions.create(
        model=name_of_model,
        messages=history,
        temperature=temperature,
        response_format=response_format,
    )

    generated_text = completion.choices[0].message.content.strip()
    return generated_text


@memory.cache
def claude_forward_cached(name_of_model, history, response_format, claude_tool_def):
    claude_tool_def = [] if claude_tool_def is None else claude_tool_def
    client = Anthropic()

    temperature = 0.7
    if name_of_model.startswith("claude-3"):
        temperature = 1

    # Convert OpenAI-style messages to Claude format
    messages = []
    for msg in history:
        if msg["role"] == "system":
            messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})

    completion = client.messages.create(
        model=name_of_model,
        messages=messages,
        temperature=temperature,
        tools=claude_tool_def,
        max_tokens=8192,
    )
    if claude_tool_def:
        first_tool_use_block = [
            x
            for x in completion.content
            if isinstance(x, anthropic.types.tool_use_block.ToolUseBlock)
        ][0]
        generated_text = json.dumps(first_tool_use_block.input)
    else:
        generated_text = completion.content[0].text.strip()
    print(f"claude generated_text: {generated_text}")
    return generated_text


class ModelAPIClient:
    def __init__(self, api_url, random_seed, lm_logger=None):
        self.api_url = url
        self.lm_logger = lm_logger
        self.random_seed = random_seed

    def forward(
        self,
        history: str,
        chat_model_id: str,
        use_cache: bool,
        logging_role: str,
        constraint_type: str = "none",
        constraints: Optional[Union[list[str], list[type], BaseModel]] = [],
        openai_response_format=None,
        claude_tool_def=None,
    ):
        assert constraint_type in ["types", "choice", "regex", "none"]
        assert not (constraint_type == "none" and constraints)
        # if constraints:
        #     assert "int" not in constraints  # probably an error
        #     assert "float" not in constraints  # probably an error
        #     assert type(constraints) == list

        if constraint_type == "types":
            constraints = [(x).__name__ for x in constraints]

        fr = ForwardRequest(
            name_of_model=chat_model_id,
            history=history,
            use_cache=use_cache,
            constraints=constraints,
            constraint_type=constraint_type,
            response_format=openai_response_format,
            random_seed=self.random_seed,
            claude_tool_def=claude_tool_def,
        )
        if (
            fr.name_of_model.startswith("gpt")
            or fr.name_of_model.startswith("o1")
            or fr.name_of_model.startswith("o3")
        ):
            response = self.forward_gpt(fr)
            # self.lm_logger.log_io(
            #     lm_input=history, lm_output=response, role=logging_role
            # )
            # return response
        elif fr.name_of_model.startswith("claude"):
            response = self.forward_claude(fr)
        else:

            response_package = requests.post(
                f"{self.api_url}:{port}/forward", json=vars(fr)
            )
            status_code = response_package.status_code
            response = response_package.json()
            if status_code != 200:
                print(f"Prediction error: {response['detail']}")
                raise Exception("Prediction error")

        generated_text = response["generated_text"]
        if self.lm_logger:
            self.lm_logger.log_io(
                lm_input=history, lm_output=generated_text, role=logging_role
            )
        # print(f"prompt: {history[-1]['content']}")
        print(f"response: {generated_text}")
        print("==================================")
        return generated_text

    def forward_gpt(self, request: ForwardRequest):

        completion = gpt_forward_cached(
            request.name_of_model, request.history, request.response_format
        )
        generated_text = completion
        return {"generated_text": generated_text}

    def forward_claude(self, request: ForwardRequest):
        completion = claude_forward_cached(
            request.name_of_model,
            request.history,
            request.response_format,
            request.claude_tool_def,
        )
        generated_text = completion
        return {"generated_text": generated_text}


if __name__ == "__main__":

    ModelAPIClient = ModelAPIClient(f"{url}:{port}", lm_logger=None, random_seed=0)

    history = [
        {
            "role": "user",
            "content": "How many words are in the sentence 'Hello World'?",
        }
    ]
    output = gpt_forward_cached(
        "gpt-4o-mini-2024-07-18",
        history,
        response_format=None,
    )

    print(output)
    print
