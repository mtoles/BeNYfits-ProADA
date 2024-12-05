from model_server import ForwardRequest
import requests


class ModelAPIClient:
    def __init__(self, api_url):
        self.api_url = api_url

    def forward(self, ForwardRequest):
        response = requests.post(f"{self.api_url}/forward", json=vars(ForwardRequest))

        if response.status_code == 200:
            return response.json()["generated_text"]
        else:
            raise Exception(f"Prediction error: {response.json()['detail']}")


if __name__ == "__main__":
    ModelAPIClient = ModelAPIClient("http://localhost:8000")

    request = ForwardRequest(
        name_of_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        history=[
            {
                "role": "user",
                "content": "How many words are in the sentence 'Hello World'?",
            },
        ],
        use_cache=False,
        constraints="int",
    )

    output = ModelAPIClient.forward(request)

    print(output)
