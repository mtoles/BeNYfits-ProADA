from server.model_server import ForwardRequest
import requests
from enum import Enum
from typing import Union, Optional
from openai import OpenAI, NotGiven
from joblib import Memory
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

memory = Memory(".joblib_cache", verbose=0)

# load_dotenv(override=False)  # Load environment variables from a .env file

port = os.getenv("LM_PORT_NO")  # Read 'PORT' environment variable
# url = os.getenv("LM_SERVER_URL")
url = "http://localhost"

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
        )
        if fr.name_of_model.startswith("gpt"):
            response = self.forward_gpt(fr)
            # self.lm_logger.log_io(
            #     lm_input=history, lm_output=response, role=logging_role
            # )
            # return response
        else:
            response_package = requests.post(
                f"{self.api_url}:{port}/forward", json=vars(fr)
            )
            status_code = response_package.status_code
            response = response_package.json()
            if status_code != 200:
                raise Exception(f"Prediction error: {response['detail']}")

        generated_text = response["generated_text"]
        if self.lm_logger:
            self.lm_logger.log_io(
                lm_input=history, lm_output=generated_text, role=logging_role
            )
        print(f"prompt: {history[-1]['content']}")
        print(f"response: {generated_text}")
        print("==================================")
        return generated_text

    @memory.cache
    def forward_gpt(request: ForwardRequest):
        # assert request.prefix is None
        client = OpenAI()
        # try:
        # if request.constraints is not None and request.constraint_type=="openai":
        #     # if request.constraint_type == "options":
        #     response_format = request.constraints
        # else:
        #     response_format = None
        if request.response_format is None:
            # completion = client.beta.chat.completions.parse(
            completion = client.chat.completions.create(
                model=request.name_of_model,
                messages=request.history,
                temperature=0.7,
            )
        else:
            completion = client.beta.chat.completions.parse(
                model=request.name_of_model,
                messages=request.history,
                temperature=0.7,
                response_format=request.response_format,
            )
        generated_text = completion.choices[0].message.content.strip()
        return {"generated_text": generated_text}
        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":

    ModelAPIClient = ModelAPIClient(f"{url}:{port}", lm_logger=None, random_seed=0)
    # ModelAPIClient = ModelAPIClient("http://localhost:55244")
    # ModelAPIClient = ModelAPIClient("http://localhost:8000")

    # request = ForwardRequest(
    #     name_of_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    #     history=[
    #         {
    #             "role": "user",
    #             "content": "How many words are in the sentence 'Hello World'?",
    #         },
    #     ],
    #     use_cache=False,
    #     constraints="int",
    # )
    history = [
        {
            "role": "user",
            "content": "How many words are in the sentence 'Hello World'?",
        }
    ]

    output = ModelAPIClient.forward(
        history,
        use_cache=True,
        logging_role="test",
        chat_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        # chat_model_id="unsloth/llama-3-8b-Instruct-bnb-4bit",
        # chat_model_id="codellama/CodeLlama-7b-hf",
    )

    print(output)
    print
