from utils import *
from transformers import AutoTokenizer
import transformers
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd
from tqdm import tqdm

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
    
    def process(self, instructions: pd.Series) -> pd.Series:
        """
        Helper method for running forward on an entire series of inputs. Subclass this method.

        Parameters:
            instructions (pd.Series): the input instructions

        Returns:
            pd.Series: the output of the primary model
        """
        pass


class GPTPrimaryModel(PrimaryModel):
    def __init__(self, use_cache):
        super().__init__()
        self.use_cache = use_cache
        pass

    def forward(self, instruction: str, temperature=0.7, model="gpt-4") -> str:
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
            model=model,
            temperature=temperature,
            use_cache=self.use_cache,
        )
        openai_output = completion.choices[0].message.content

        return openai_output

    def process(self, instructions: pd.Series) -> pd.Series:
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

    def process(self, instructions: pd.Series) -> pd.Series:
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
        outputs = [x[len(y) :] for x, y in zip(outputs, llama_formatted_input)]
        return outputs
