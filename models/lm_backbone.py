from tqdm import tqdm
from typing import List, Optional
from tqdm import tqdm
import os
from huggingface_hub import login
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
from typing import List, Callable
from enum import Enum
from models.lm_logging import LmLogger
from models.model_utils import load_lm
from inspect import currentframe

tqdm.pandas()

### TEMPLATES ###


class PromptMode(Enum):
    DEFAULT = "default"


class LmBackboneModel:
    def __init__(
        self,
        lm_wrapper,
        mode: PromptMode = PromptMode.DEFAULT,
        lm_logger: Optional[LmLogger] = None,
    ):
        self.lm_wrapper = lm_wrapper
        self.mode = mode
        self.lm_logger = lm_logger
        # self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if lm_wrapper.family in ["llama"]:
            self.hf_api_key = os.getenv("HF_TOKEN")
            login(token=self.hf_api_key)

    def _get_format_func(self) -> Callable:
        format_funcs = {
            ("llama", PromptMode.DEFAULT): self._format_llama_prompt_default,
            ("gpt", PromptMode.DEFAULT): self._format_gpt_prompt_default,
            ("o1", PromptMode.DEFAULT): self._format_o1_prompt_default,
            ("gemma", PromptMode.DEFAULT): self._format_gemma_prompt_default,
            ("mistral", PromptMode.DEFAULT): self._format_mistral_prompt_default,
        }
        return format_funcs.get(
            (self.lm_wrapper.family, self.mode), self._format_default_prompt
        )

    def _format_llama_prompt_default(self, history: list[dict]) -> str:
        return self.lm_wrapper.language_model._tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )

    def _format_gpt_prompt_default(self, history: list[dict]) -> str:
        # bechmark_template_json = 'Return the questions as a list in JSON format, as in {{"questions": ["The first question?", "The second question?"]}}'
        # return benchmark_template.format(document=document, task=task, n_clarifying_questions=1, plural="", json="")
        return history

    def _format_o1_prompt_default(self, history: list[dict]) -> str:
        # replace 'system' calls with 'user' calls
        for turn in history:
            if turn["role"] == "system":
                turn["role"] = "user"
        return history

    def _format_gemma_prompt_default(self, history: list[dict]) -> str:
        raise NotImplementedError

    def _format_mistral_prompt_default(self, history: list[dict]) -> str:
        raise NotImplementedError

    def _format_default_prompt(self, history: list[dict]) -> str:
        raise NotImplementedError

    def forward(
        self,
        history: List[dict],
        num_completions: Optional[int] = None,
        logging_role: str = "No_Role",
    ) -> str | List[str]:
        format_func = self._get_format_func()
        formatted_prompt = format_func(history)
        sequences = list(
            self.lm_wrapper.language_model.predict_many(
                [
                    LmPrompt(
                        formatted_prompt,
                        cache=True,
                        logprobs=0,
                        num_completions=num_completions,
                    )
                ],
                completion_window=CompletionWindow.ASAP,
            )
        )
        sequence = sequences[0]

        # Sequence is a OpenAiLmPrediction object
        if num_completions is None:
            output = sequence.completion_text
        # Sequence is a list of OpenAiLmPrediction objects
        else:
            output = [x.completion_text for x in sequence]
        if self.lm_logger:
            first_output = output[-1] if isinstance(output, list) else output
            self.lm_logger.log_io(
                lm_input=history, lm_output=first_output, role=logging_role
            )
        return output


if __name__ == "__main__":
    ### testing

    chat_history = [
        {
            "role": "system",
            "content": "respond to the user's next question",
        },
        {"role": "user", "content": "what is your name?"},
    ]
    gpt_wrapper = load_lm("gpt-3-5-turbo")
    model = LmBackboneModel(gpt_wrapper)
    output = model.forward(chat_history)
    print(output)

    llama3_wrapper = load_lm("meta-llama/Meta-Llama-3-8B-Instruct")
    model = LmBackboneModel(llama3_wrapper)
    output = model.forward(chat_history)
    print(output)

    # apply model
