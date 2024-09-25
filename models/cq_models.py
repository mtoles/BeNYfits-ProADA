from utils import *
from json import loads
from tqdm import tqdm
from typing import List
from tqdm import tqdm
import os
from huggingface_hub import login
import numpy as np
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
from typing import List, Callable
from enum import Enum

from models.model_utils import load_lm

tqdm.pandas()

### TEMPLATES ###

# benchmark_template = "Context: {document}\n\nTask:{task}\n\nYou are trying to complete the task but do not have enough information from the document. Ask {n_clarifying_questions} question{plural} about the situation that can help you complete the task. In each question, only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Do not say anything other than the question. {json}"
# bechmark_template_json = 'Return the questions as a list in JSON format, as in {{"questions": ["The first question?", "The second question?"]}}'


class PromptMode(Enum):
    DEFAULT = "default"


class BaseClarifyingQuestionModel:
    def __init__(self, lm_wrapper, mode: PromptMode = PromptMode.DEFAULT):
        self.lm_wrapper = lm_wrapper
        self.mode = mode
        # self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_api_key = os.getenv("HF_TOKEN")
        login(token=self.hf_api_key)

    def _get_format_func(self) -> Callable:
        format_funcs = {
            ("llama", PromptMode.DEFAULT): self._format_llama_prompt_default,
            ("gpt", PromptMode.DEFAULT): self._format_gpt_prompt_default,
            ("gemma", PromptMode.DEFAULT): self._format_gemma_prompt_default,
            (
                "mistral",
                PromptMode.DEFAULT,
            ): self._format_mistral_prompt_default,
        }
        return format_funcs.get(
            (self.lm_wrapper.family, self.mode), self._format_default_prompt
        )

    def _format_llama_prompt_default(self, history: list[dict]) -> str:
        # llama_user_message = benchmark_template.format(document=document, task=task, n_clarifying_questions=1, plural="s", json="")

        # formatted_user_messages = [
        #     {
        #         "role": "system",
        #         "content": f"You are trying to help the user with the task below.",
        #     },
        #     {
        #         "role": "user",
        #         "content": llama_user_message,
        #     },
        # ]

        return self.lm_wrapper.language_model._tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )

    def _format_gpt_prompt_default(self, history: list[dict]) -> str:
        # bechmark_template_json = 'Return the questions as a list in JSON format, as in {{"questions": ["The first question?", "The second question?"]}}'
        # return benchmark_template.format(document=document, task=task, n_clarifying_questions=1, plural="", json="")
        return history

    def _format_gemma_prompt_default(self, history: list[dict]) -> str:
        raise NotImplementedError

    def _format_mistral_prompt_default(self, history: list[dict]) -> str:
        raise NotImplementedError

    def _format_default_prompt(self, history: list[dict]) -> str:
        raise NotImplementedError

    # def forward_batch_generate_single_question(
    #     self, documents: List[str], tasks: List[str]
    # ) -> List[str]:
    #     format_func = self._get_format_func()
    #     formatted_instructions = [
    #         format_func(document, task) for document, task in zip(documents, tasks)
    #     ]

    #     sequences = self.lm_wrapper.language_model.predict_many(
    #         [LmPrompt(p, cache=False, max_tokens=512) for p in formatted_instructions],
    #         completion_window=CompletionWindow.ASAP,
    #     )

    #     outputs = [x.completion_text for x in sequences]

    #     return outputs

    def forward(self, history: List[dict]) -> str:
        format_func = self._get_format_func()
        formatted_prompt = format_func(history)

        sequences = self.lm_wrapper.language_model.predict_many(
            [LmPrompt(formatted_prompt, cache=False, max_tokens=512)],
            completion_window=CompletionWindow.ASAP,
        )
        output = [x.completion_text for x in sequences]
        return output



if __name__ == "__main__":
    ### testing

    chat_history = [
        {
            "role": "system",
            "content": "respond to the user's next question",
        },
        {
            "role": "user",
            "content": "what is your name?"
        }
    ]
    gpt_wrapper = load_lm("gpt-3-5-turbo-0125")
    model = BaseClarifyingQuestionModel(gpt_wrapper)
    output = model.forward(chat_history)
    print(output)

    llama3_wrapper = load_lm("meta-llama/Meta-Llama-3-8B-Instruct")
    model = BaseClarifyingQuestionModel(llama3_wrapper)
    output = model.forward(chat_history)
    print(output)

    # apply model
