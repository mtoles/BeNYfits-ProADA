import requests
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
import json
from lmwrapper.openai_wrapper import OpenAiModelNames

class ModelAPIClient:
    def __init__(self, api_url):
        self.api_url = api_url

    def load_model(self, family, hf_name):
        payload = {"family": family, "hf_name": hf_name}
        print(f"Payload to /load_model: {payload}")  # Debugging step

        response = requests.post(f"{self.api_url}/load_model", json=payload)
        if response.status_code == 200:
            return response.json()["model"]
        else:
            raise Exception(f"Error loading model: {response.json()['detail']}")

    def predict_many(self, id_of_model, prompts: list[LmPrompt]):
        prompts_data = [
            {
                "text": p.text,
                "cache": p.cache,
                "logprobs": p.logprobs,
                "max_tokens": p.max_tokens,
            }
            for p in prompts
        ]

        # print(json.dumps(prompts_data, indent=2))

        response = requests.post(
            f"{self.api_url}/predict_many",
            json={
                "id_of_model": id_of_model,
                "prompts": prompts_data,
            },
        )

        if response.status_code == 200:
            return response.json()["responses"]
        else:
            raise Exception(f"Prediction error: {response.json()['detail']}")

# Usage
client = ModelAPIClient("http://localhost:8000")
# model_info = client.load_model("llama", "meta-llama/Meta-Llama-3-8B-Instruct")

model_info = client.load_model("gpt", OpenAiModelNames.gpt_4o_mini_2024_07_18)

print(f"Model Info: {model_info}")

# Create prompts
prompts = [
    LmPrompt("Hello, how are you?", cache=True, logprobs=0, max_tokens=50),
    LmPrompt("What's the weather today?", cache=False, logprobs=0, max_tokens=100),
]

response = client.predict_many(model_info, prompts)

print(response)