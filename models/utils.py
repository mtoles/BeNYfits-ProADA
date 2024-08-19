from enum import Enum
from typing import Dict, Optional
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt
# https://github.com/DaiseyCode/lmwrapper

class ModelFamily(Enum):
    LLAMA = "llama"
    GPT = "gpt"
    GEMMA = "gemma"
    MISTRAL = "mistral"

class LanguageModelWrapper:
    def __init__(self, display_name: str, family: ModelFamily, hf_name: str):
        self.display_name = display_name
        self.family = family
        self.hf_name = hf_name
        self._language_model = None

    @property
    def language_model(self):
        if self._language_model is None:
            if self.family in [ModelFamily.LLAMA, ModelFamily.GEMMA]:
                self._language_model = get_huggingface_lm(self.hf_name)
            else:
                self._language_model = get_open_ai_lm(self.hf_name)
            self._language_model._tokenizer.pad_token_id = self._language_model._tokenizer.eos_token_id
            self._language_model._tokenizer.padding_side = "left"
            print("Hello")
        return self._language_model

    def __str__(self):
        return f"{self.display_name} ({self.family.value})"

MODEL_MAP: Dict[str, LanguageModelWrapper] = {
    "meta-llama/Meta-Llama-3-8B-Instruct": LanguageModelWrapper("Llama 8B Instruct", ModelFamily.LLAMA, "meta-llama/Meta-Llama-3-8B-Instruct"),
    "meta-llama/Meta-Llama-3-70B-Instruct": LanguageModelWrapper("Llama 70B Instruct", ModelFamily.LLAMA, "meta-llama/Meta-Llama-3-70B-Instruct"),
    "gpt2": LanguageModelWrapper("GPT-2", ModelFamily.GPT, "gpt2"),
    "gpt-3-5-turbo-instruct": LanguageModelWrapper("GPT-3.5-Turbo-Instruct", ModelFamily.GPT, OpenAiModelNames.gpt_3_5_turbo_instruct),
    "google/gemma-2b-it": LanguageModelWrapper("Gemma 2B Instruction Tuned", ModelFamily.GEMMA, "google/gemma-2b-it"),
    "google/gemma-7b-it": LanguageModelWrapper("Gemma 7B Instruction Tuned", ModelFamily.GEMMA, "google/gemma-7b-it"),
    # Add more models here as needed
}

def load_lm(model_name: str) -> LanguageModelWrapper:
    if model_name not in MODEL_MAP:
        available_models = ", ".join(MODEL_MAP.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models are: {available_models}")

    return MODEL_MAP[model_name]

def main():
    # Test different variants
    models_to_test = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama-70b-instruct",
        "gpt2",
        "LLAMA-8B-INSTRUCT",  # Test case insensitivity
        "unknown-model"  # Test error case
    ]

    for model_name in models_to_test:
        try:
            print(f"Loading model: {model_name}")
            lm_wrapper = load_lm(model_name)
            print(f"Successfully loaded {lm_wrapper}")
            print(f"Hugging Face model name: {lm_wrapper.hf_name}")
            # Access the language model (this will trigger loading if not already loaded)
            lm = lm_wrapper.language_model
            print(f"Language model loaded: {lm is not None}")
            # You can add more tests here, e.g., generating text
            # print(lm.generate("Hello, world!", max_length=50))
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
        print("---")

if __name__ == "__main__":
    main()