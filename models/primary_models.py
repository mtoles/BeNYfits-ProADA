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


class Llama3PrimaryModel(PrimaryModel):
    """
    Llama3 chat primary model.
    """

    def __init__(self, model_name, batch_size, pipeline):
        super().__init__()
        self.model_name = model_name

        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        login(token=self.hf_api_key)
        self.pipeline = pipeline
        # if pipeline:
        #     self.pipeline = pipeline
        # else:
        #     self.pipeline = transformers.pipeline(
        #         "text-generation",
        #         model=self.model_name,
        #         torch_dtype=torch.bfloat16,
        #         device_map="auto",
        #     )

        # self.system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
        self.system_prompt = "You are a helpful assistant. Always answer the question and be faithful to the provided document."

        self.batch_size = batch_size

    def process_single(self, instructions: pd.Series) -> pd.Series:
        # llama_formatted_input = [
        #     f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"
        #     for instruction in instructions
        # ]
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
            [LmPrompt(p, cache=False) for p in llama_formatted_prompts],
            completion_window=CompletionWindow.ASAP,
        )

        outputs = [x.completion_text for x in sequences]
        # for seq, llama_formatted_prompt in zip(sequences, llama_formatted_prompts):
        #     llama_parsed_output = seq[0]["generated_text"]
        #     llama_parsed_output = llama_parsed_output[len(llama_formatted_prompt) :]
        #     llama_parsed_output = llama_parsed_output.strip()

        #     outputs.append(llama_parsed_output)

        return pd.Series(outputs)

    def process_list(self, instructions: pd.Series) -> pd.Series:
        list_len = len(instructions[0])
        flat_instructions = instructions.explode()
        model_outputs = self.process_single(flat_instructions)
        # unexplode
        model_outputs = model_outputs.groupby(level=0).apply(list)
        return model_outputs

# Base class for primary model
class BasePrimaryModel:
    def __init__(self, lm_wrapper):
        self.lm_wrapper = lm_wrapper
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
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
            ModelFamily.MISTRAL: self._format_mistral_prompt
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
            ModelFamily.MISTRAL: self._format_mistral_prompt
        }.get(self.lm_wrapper.family, self._format_default_prompt)

        formatted_prompts = [ format_func(instruction) for instruction in instructions]

        sequences = self.lm_wrapper.language_model.predict_many(
            [LmPrompt(p, cache=False) for p in formatted_prompts],
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

    # l3_pipeline = transformers.pipeline(
    #     "text-generation",
    #     model="meta-llama/Meta-Llama-3-8B-Instruct",
    #     # model=lm,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     token=hf_api_key,
    # )
    lm = get_huggingface_lm("meta-llama/Meta-Llama-3-8B-Instruct")
    # l3_pipeline.model = lm

    l3_model = Llama3PrimaryModel("meta-llama/Meta-Llama-3-8B-Instruct", 1, lm)
    doc = "I don't know how to cook spaghetti but my girlfriend is coming over and wants to eat it"
    prompt = "What advice would you give to this person?"
    instruction = l3_model.prepare_instruction(doc, prompt)
    output = l3_model.process_single(pd.Series([instruction] * 2))
    print(output)
    print
