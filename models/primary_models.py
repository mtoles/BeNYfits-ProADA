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
        return self.prepare_instruction(
                "\n\n".join([doc_summ, ca]), prompt
            )

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


class Llama2PrimaryModel(PrimaryModel):
    """
    Llama2 chat primary model.
    """

    def __init__(self, model_size, batch_size):
        super().__init__()
        if model_size == "7b":
            self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        elif model_size == "13b":
            self.model_name = "meta-llama/Llama-2-13b-chat-hf"
        elif model_size == "70b":
            self.model_name = "meta-llama/Llama-2-70b-chat-hf"
        else:
            raise ValueError(f"Unknown llama2 model size {model_size}")
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        login(token=self.hf_api_key)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.pipeline.tokenizer.pad_token_id = 0
        self.pipeline.tokenizer.padding_side = "left"
        # self.system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
        self.system_prompt = "You are a helpful assistant. Always answer the question and be faithful to the provided document."

        self.batch_size = batch_size

    def process_single(self, instructions: pd.Series) -> pd.Series:
        llama_formatted_input = [
            f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"
            for instruction in instructions
        ]
        # wrap the pipeline so we can have a progress bar
        sequences = []
        for i in tqdm(range(0, len(llama_formatted_input), self.batch_size)):
            batch = llama_formatted_input[i : i + self.batch_size]
            sequences.extend(
                self.pipeline(
                    batch,
                    # do_sample=True,
                    # top_k=10,
                    # num_return_sequences=1,
                    # eos_token_id=self.tokenizer.eos_token_id,
                    # max_length=300,
                )
            )

        outputs = [sequence[0]["generated_text"] for sequence in sequences]
        # delete the prompt
        # outputs = [output[len(llama_formatted_input) :] for output in outputs]
        outputs = pd.Series(
            [x[len(y) :] for x, y in zip(outputs, llama_formatted_input)]
        )
        return outputs

    def process_list(self, instructions: pd.Series) -> pd.Series:
        list_len = len(instructions[0])
        flat_instructions = instructions.explode()
        model_outputs = self.process(flat_instructions)
        # unexplode
        model_outputs = model_outputs.groupby(level=0).apply(list)
        return model_outputs


class Llama3PrimaryModel(PrimaryModel):
    """
    Llama3 chat primary model.
    """

    def __init__(self, model_name, batch_size, pipeline=None):
        super().__init__()
        self.model_name = model_name

        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        login(token=self.hf_api_key)
        if pipeline:
            self.pipeline = pipeline
        else:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.pipeline.tokenizer.pad_token_id = 0
        self.pipeline.tokenizer.padding_side = "left"
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
            self.pipeline.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            for prompt in formatted_user_messages
        ]
        sequences = self.pipeline(llama_formatted_prompts)

        outputs = []
        for seq, llama_formatted_prompt in zip(sequences, llama_formatted_prompts):
            llama_parsed_output = seq[0]["generated_text"]
            llama_parsed_output = llama_parsed_output[len(llama_formatted_prompt) :]
            llama_parsed_output = llama_parsed_output.strip()

            outputs.append(llama_parsed_output)

        return pd.Series(outputs)

    def process_list(self, instructions: pd.Series) -> pd.Series:
        list_len = len(instructions[0])
        flat_instructions = instructions.explode()
        model_outputs = self.process_single(flat_instructions)
        # unexplode
        model_outputs = model_outputs.groupby(level=0).apply(list)
        return model_outputs
