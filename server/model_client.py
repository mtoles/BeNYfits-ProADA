from server.model_server import ForwardRequest
import requests


class ModelAPIClient:
    def __init__(self, api_url, logger=None):
        self.api_url = api_url
        self.logger = logger

    def forward(
        self,
        history: str,
        chat_model_id: str,
        use_cache: bool,
        logging_role: str,
        constraints: str = None,
    ):
        fr = ForwardRequest(
            name_of_model=chat_model_id,
            history=history,
            use_cache=use_cache,
            constraints=constraints,
        )
        response = requests.post(f"{self.api_url}/forward", json=vars(fr))

        if response.status_code == 200:
            return response.json()["generated_text"]
        else:
            raise Exception(f"Prediction error: {response.json()['detail']}")


if __name__ == "__main__":
    ModelAPIClient = ModelAPIClient("http://localhost:8000")

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
        chat_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    )

    print(output)
    print
