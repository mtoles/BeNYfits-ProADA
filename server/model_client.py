from server.model_server import ForwardRequest
import requests
from typing import Union, Optional
from openai import OpenAI, NotGiven
from joblib import Memory
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import threading
import traceback

load_dotenv()
memory = Memory(".joblib_cache", verbose=0)
LM_PORT_NO = int(os.getenv("LM_PORT_NO"))

results = {}


class ModelAPIClient:
    def __init__(self, api_url, port_no, lm_logger=None):
        self.api_url = api_url
        self.port_no = port_no
        self.lm_logger = lm_logger

    def _forward(self, fr: ForwardRequest):
        url = f"{self.api_url}:{self.port_no}/forward"
        try:
            response = requests.post(url, json=vars(fr))
            response.raise_for_status()
            results[fr.json()] = response
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"traceback:\n{traceback.format_exc()}"
            )

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

        if constraint_type == "types":
            constraints = [(x).__name__ for x in constraints]

        fr = ForwardRequest(
            name_of_model=chat_model_id,
            history=history,
            use_cache=use_cache,
            constraints=constraints,
            constraint_type=constraint_type,
            response_format=openai_response_format,
        )
        if fr.name_of_model.startswith("gpt"):
            response = self.forward_gpt(fr)
            status_code = response["status_code"]
        else:
            t = threading.Thread(target=self._forward, args=(fr,))
            t.start()
            t.join()
            result = results[fr.json()]
            status_code = result.status_code

        if status_code == 200:
            if isinstance(result, str):
                generated_text = result
            else:
                generated_text = result.json()

            # generated_text = response_package["generated_text"]
            if self.lm_logger is not None:
                self.lm_logger.log_io(
                    lm_input=history, lm_output=generated_text, role=logging_role
                )
            return generated_text
        else:
            raise Exception(f"Prediction error: {response.json()['detail']}")

    @memory.cache
    def forward_gpt(request: ForwardRequest):
        client = OpenAI()
        if request.response_format is None:
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
        return {"generated_text": generated_text, "status_code": 200}


if __name__ == "__main__":
    ModelAPIClient = ModelAPIClient(f"http://coffee.cs.columbia.edu", LM_PORT_NO)

    def hist(i):
        return [
            {
                "role": "user",
                "content": f"How many words are in the sentence '{' '.join(['buffalo']*i)}'?",
            }
        ]

    threads = []
    for i in range(2, 4):
        history = hist(i)
        t = threading.Thread(
            target=ModelAPIClient.forward,
            args=(history,),
            kwargs={
                "use_cache": False,
                "logging_role": "test",
                "chat_model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            },
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
        # print all output
        print(results)

    print("done")
    print(list(results.values()))
