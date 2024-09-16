from utils import *
from transformers import AutoTokenizer
import transformers
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd
from tqdm import tqdm
from typing import List
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
from models.utils import ModelFamily


load_dotenv()


class PrimaryModel:
    """
    Base class for primary models. Subclass this class to implement a new primary model.
    """

    def __init__(self):
        self.prompt_template = "Memorize the following document and then follow the instructions below:\n\nDocument:\n\n%s\n\nInstructions: %s\n\n"
        pass

    def forward(x: str) -> str:
        """
        Subclass this method to implement a new primary model.

        Parameters:
            x (str): the input document

        Returns:
            str: the output of the primary model
        """
        raise NotImplementedError
        return x

    def prepare_instruction(self, doc: str, prompt: str) -> str:
        """
        Format the document and prompt into a single string that can be used as input for the primary model.

        Parameters:
            doc (str): the input document
            prompt (str): the input prompt

        Returns:
            str: the formatted instruction
        """
        return self.prompt_template % (doc, prompt)

    def prepare_ca_instruction(self, doc_summ: str, ca: str, prompt: str):
        return self.prepare_instruction("\n\n".join([doc_summ, ca]), prompt)

    def process(self, instructions: pd.Series) -> pd.Series:
        """
        Helper method for running forward on an entire series of inputs. Subclass this method.

        Parameters:
            instructions (pd.Series): the input instructions

        Returns:
            pd.Series: the output of the primary model
        """
        raise NotImplementedError
        pass


class GPTPrimaryModel(PrimaryModel):
    def __init__(self, model_name, use_cache):
        super().__init__()
        self.model_name = model_name
        self.use_cache = use_cache
        pass

    def forward(self, instruction: str, temperature=0.0) -> str:
        """
        Parameters:
            instruction (str): the input instruction
            temperature (float): the temperature to use for the GPT model
            model (str): the name of the model to use

        Returns:
            str: the output of the GPT model
        """
        completion = conditional_openai_call(
            instruction,
            model=self.model_name,
            temperature=temperature,
            use_cache=self.use_cache,
        )
        openai_output = completion.choices[0].message.content

        return openai_output

    def forward_list(self, instructions: List[str], temperature=0.0) -> List[str]:
        """helper method to call forward when the input is a list of instructions"""
        outputs = []
        for instruction in instructions:
            outputs.append(self.forward(instruction, temperature))
        return outputs

    def process_list(self, instructions: pd.Series) -> pd.Series:
        "call forward on each item of a series where the item is a ***list***"
        pm_output = instructions.progress_apply(lambda x: self.forward_list(x))
        return pm_output

    def process_single(self, instructions: pd.Series) -> pd.Series:
        "call forward on each item of a series where the item is a ***string***"
        pm_output = instructions.progress_apply(lambda x: self.forward(x))
        return pm_output

# Base class for primary model
class BasePrimaryModel:
    def __init__(self, lm_wrapper):
        self.lm_wrapper = lm_wrapper
        # self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_api_key = os.getenv("HF_TOKEN")
        login(token=self.hf_api_key)

    def prepare_instruction(self, doc: str, prompt: str) -> str:
        instruction = "Memorize the following document and then follow the instructions below:\n\nDocument:\n\n%s\n\nInstructions: %s\n\n"
        return instruction % (doc, prompt)

    def prepare_ca_instruction(self, doc_summ: str, ca: str, prompt: str):
        return self.prepare_instruction("\n\n".join([doc_summ, ca]), prompt)

        formatted_user_messages = [
            [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": instruction,
                },
            ]
            for instruction in instructions
        ]
        llama_formatted_prompts = [
            self.pipeline._tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            for prompt in formatted_user_messages
        ]
        # sequences = self.pipeline(
        sequences = self.pipeline.predict_many(
            ([LmPrompt(p, cache=False) for p in llama_formatted_prompts]),
            completion_window=CompletionWindow.ASAP,
        )

        outputs = [x.completion_text for x in sequences]
        # for seq, llama_formatted_prompt in zip(sequences, llama_formatted_prompts):
        #     llama_parsed_output = seq[0]["generated_text"]
        #     llama_parsed_output = llama_parsed_output[len(llama_formatted_prompt) :]
        #     llama_parsed_output = llama_parsed_output.strip()

        #     outputs.append(llama_parsed_output)

        return pd.Series(outputs)

    def forward(self, instruction: str) -> str:
        format_func = {
            ModelFamily.LLAMA: self._format_llama_prompt,
            ModelFamily.GPT: self._format_gpt_prompt,
            ModelFamily.GEMMA: self._format_gemma_prompt,
            ModelFamily.MISTRAL: self._format_mistral_prompt,
        }.get(self.lm_wrapper.family, self._format_default_prompt)

        formatted_instruction = format_func(instruction)

        sequences = self.lm_wrapper.language_model.predict_many(
            ([LmPrompt(formatted_instruction, cache=False)]),
            completion_window=CompletionWindow.ASAP,
        )
        return sequences[0].completion_text

    def process_single(self, instructions: pd.Series) -> pd.Series:
        pm_output = instructions.progress_apply(lambda x: self.forward(x))
        return pm_output

    def forward_list(self, instructions: pd.Series) -> pd.Series:
        format_func = {
            ModelFamily.LLAMA: self._format_llama_prompt,
            ModelFamily.GPT: self._format_gpt_prompt,
            ModelFamily.GEMMA: self._format_gemma_prompt,
            ModelFamily.MISTRAL: self._format_mistral_prompt,
        }.get(self.lm_wrapper.family, self._format_default_prompt)

        formatted_prompts = [format_func(instruction) for instruction in instructions]

        sequences = self.lm_wrapper.language_model.predict_many(
            [LmPrompt(p, cache=False, max_tokens=512) for p in formatted_prompts],
            completion_window=CompletionWindow.ASAP,
        )

        outputs = [x.completion_text for x in sequences]
        return pd.Series(outputs)

    def process_list(self, instructions: pd.Series) -> pd.Series:
        return self.forward_list(instructions)

    def _format_llama_prompt(self, instruction: str) -> str:
        llama_system_prompt = "You are a helpful assistant. Always answer the question and be faithful to the provided document."
        formatted_user_message = [
            {
                "role": "system",
                "content": llama_system_prompt,
            },
            {
                "role": "user",
                "content": instruction,
            },
        ]

        return self.lm_wrapper.language_model._tokenizer.apply_chat_template(
            formatted_user_message, tokenize=False, add_generation_prompt=True
        )

    def _format_gpt_prompt(self, instruction: str) -> str:
        return instruction

    def _format_gemma_prompt(self, instruction: str) -> str:
        return instruction

    def _format_mistral_prompt(self, instruction: str) -> str:
        return instruction

    def _format_default_prompt(self, instruction: str) -> str:
        return instruction


if __name__ == "__main__":
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")