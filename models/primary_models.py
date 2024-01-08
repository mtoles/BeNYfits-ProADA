from utils import *
from transformers import AutoTokenizer
import transformers
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd

load_dotenv()


class PrimaryModel:
    def __init__(self):
        self.prompt_template = "Memorize the following document and then follow the instructions below:\n\nDocument:\n\n%s\n\nInstructions: %s\n\n"
        pass

    def forward(x):
        # subclass this method
        return x

    def prepare_instruction(self, doc: str, prompt: str):
        return self.prompt_template % (doc, prompt)


class GPTPrimaryModel(PrimaryModel):
    def __init__(self, use_cache):
        super().__init__()
        self.use_cache = use_cache
        pass

    def forward(self, instruction: str, temperature=0.7, model="gpt-4"):
        completion = conditional_openai_call(
            instruction,
            model=model,
            n=1,
            temperature=temperature,
            use_cache=self.use_cache,
        )
        openai_output = completion.choices[0].message.content

        return openai_output

    def process(self, instructions: pd.Series):
        pm_output = instructions.progress_apply(lambda x: self.forward(x))
        return pm_output


class Llama2PrimaryModel(PrimaryModel):
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
        self.system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
        self.batch_size = batch_size

    def process(self, instructions: pd.Series):
        llama_formatted_input = [
            f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"
            for instruction in instructions
        ]
        sequences = self.pipeline(
            llama_formatted_input,
            batch_size=self.batch_size,
            # do_sample=True,
            # top_k=10,
            # num_return_sequences=1,
            # eos_token_id=self.tokenizer.eos_token_id,
            # max_length=300,
        )
        outputs = [sequence[0]["generated_text"] for sequence in sequences]
        # delete the prompt
        outputs = [output[len(llama_formatted_input) :] for output in outputs]
        return outputs
