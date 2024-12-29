from typing import Dict
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt
import requests

# https://github.com/DaiseyCode/lmwrapper

class LanguageModelWrapper:
    def __init__(self, display_name: str, family: str, hf_name: str):
        self.display_name = display_name
        self.family = family
        self.hf_name = hf_name
        self._language_model_name = None
        self.api_url = "http://localhost:8000"

    @property
    def language_model_name(self):
        if self._language_model_name is None:
            wrapped = self.hf_name not in ["infly/OpenCoder-8B-Instruct"]
            payload = {"family": self.family, "hf_name": self.hf_name, "wrapped": wrapped}
            print(f"Payload to /load_model: {payload}")  # Debugging step

            response = requests.post(f"{self.api_url}/load_model", json=payload)
            if response.status_code == 200:
                self._language_model_name = response.json()["model"]
            else:
                raise Exception(f"Error loading model: {response.json()['detail']}")
        
            print(f"Model Pipeline Instantiated with API: {self.display_name} {self.family} -- {self._language_model_name}")
        return self._language_model_name

    def __str__(self):
        return f"{self.display_name} ({self.family})"
    
    def apply_chat_template(self, history: list[dict]):
        response = requests.post(
            f"{self.api_url}/apply_chat_template",
            json={
                "id_of_model": self.language_model_name,
                "history": history,
            },
        )

        if response.status_code == 200:
            return response.json()["formatted_chat"]
        else:
            raise Exception(f"Prediction error: {response.json()['detail']}")

    def predict_many_outputs(self, prompts: list[LmPrompt]):
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
                "id_of_model": self.language_model_name,
                "prompts": prompts_data,
            },
        )

        if response.status_code == 200:
            return response.json()["responses"]
        else:
            raise Exception(f"Prediction error: {response.json()['detail']}")


MODEL_MAP: Dict[str, LanguageModelWrapper] = {
    "meta-llama/Meta-Llama-3-8B-Instruct": LanguageModelWrapper(
        "Llama 8B Instruct", "llama", "meta-llama/Meta-Llama-3-8B-Instruct"
    ),
    "meta-llama/Meta-Llama-3-13B-Instruct": LanguageModelWrapper(
        "Llama 13B Instruct", "llama", "meta-llama/Meta-Llama-3-13B-Instruct"
    ),
    "meta-llama/Meta-Llama-3-70B-Instruct": LanguageModelWrapper(
        "Llama 70B Instruct", "llama", "meta-llama/Meta-Llama-3-70B-Instruct"
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": LanguageModelWrapper(
        "Llama 8B Instruct", "llama", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": LanguageModelWrapper(
        "Llama 70B Instruct", "llama", "meta-llama/Meta-Llama-3.1-70B-Instruct"
    ),
    "meta-llama/CodeLlama-7b-Instruct-hf": LanguageModelWrapper(
        "meta-llama/CodeLlama-7b-Instruct-hf",
        "llama",
        "meta-llama/CodeLlama-Instruct-7b-hf",
    ),
    "meta-llama/CodeLlama-13b-Instruct-hf": LanguageModelWrapper(
        "meta-llama/CodeLlama-13b-Instruct-hf",
        "llama",
        "meta-llama/CodeLlama-13b-Instruct-hf",
    ),
    "gpt2": LanguageModelWrapper("GPT-2", "llama", "gpt2"),
    "gpt-3-5-turbo-instruct": LanguageModelWrapper(
        "GPT-3.5-Turbo-Instruct",
        "gpt",
        OpenAiModelNames.gpt_3_5_turbo_instruct,
    ),
    "gpt-3-5-turbo": LanguageModelWrapper(
        "GPT-3.5-Turbo", "gpt", OpenAiModelNames.gpt_3_5_turbo
    ),
    "gpt-4o-2024-05-13": LanguageModelWrapper(
        "gpt-4o-2024-05-13", "gpt", OpenAiModelNames.gpt_4o_2024_05_13
    ),
    "gpt-4o-mini-2024-07-18": LanguageModelWrapper(
        "gpt-4o-mini-2024-07-18", "gpt", OpenAiModelNames.gpt_4o_mini_2024_07_18
    ),
    # "o1-preview-2024-09-12": LanguageModelWrapper(
    #     "o1-preview-2024-09-12", "o1", OpenAiModelNames.o1_preview_2024_09_12
    # ),
    "google/gemma-2b-it": LanguageModelWrapper(
        "Gemma 2B Instruction Tuned", "gemma", "google/gemma-2b-it"
    ),
    "google/gemma-7b-it": LanguageModelWrapper(
        "Gemma 7B Instruction Tuned", "gemma", "google/gemma-7b-it"
    ),
    "gpt-4o-mini": LanguageModelWrapper(
        "GPT 4o Mini", "gpt", "gpt-4o-mini"
    ),
    # Add more models here as needed
}


def load_lm(model_name: str) -> LanguageModelWrapper:
    print(f"Loading LM: {model_name}")

    if model_name not in MODEL_MAP:
        available_models = ", ".join(MODEL_MAP.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available models are: {available_models}"
        )

    return MODEL_MAP[model_name]


def main():
    # Test different variants
    models_to_test = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama-70b-instruct",
        "gpt2",
        "LLAMA-8B-INSTRUCT",  # Test case insensitivity
        "unknown-model",  # Test error case
    ]

    for model_name in models_to_test:
        try:
            print(f"Loading model: {model_name}")
            lm_wrapper = load_lm(model_name)
            print(f"Successfully loaded {lm_wrapper}")
            print(f"Hugging Face model name: {lm_wrapper.hf_name}")
            # Access the language model (this will trigger loading if not already loaded)
            # TODO _ RATTAN _ FIX THIS MAIN FUNCTION
            lm = lm_wrapper.language_model
            print(f"Language model loaded: {lm is not None}")
            # You can add more tests here, e.g., generating text
            # print(lm.generate("Hello, world!", max_length=50))
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
        print("---")


if __name__ == "__main__":
    main()
