from utils import *
import pandas as pd
from json import loads
from tqdm import tqdm
from typing import List

from transformers import AutoTokenizer
import transformers
import os
from huggingface_hub import login
import torch

import click
import numpy as np
import os


class Llama2PrimaryModel:
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
        self.system_prompt = "You are a helpful assistant. Always answer the question and be faithful to the provided document. If you do not have enough information to answer the question, ask a clarifying question instead."

        self.batch_size = batch_size

    def process(
        self,
        documents: pd.Series,
        tasks: pd.Series,
        answers: pd.Series,
    ) -> pd.Series:
        # ans = "\n".join(answers.to_list())
        # llama_formatted_input = [
        #     f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{d}\n\n{t}\n\n{a} [/INST]"
        #     for d, t, a in zip(documents, tasks, ans)
        # ]
        llama_formatted_input = []
        for doc, task, anss in zip(documents, tasks, answers):
            ans = "\n".join(anss)
            llama_formatted_input.append(
                f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{doc}\n{ans}\n\n{task} [/INST]"
            )

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


@click.command()
@click.option("--pm_name", default="llama2", help="Name of the primary model to use")
@click.option(
    "--pm_size", help="Size of the primary model to use, one of {7b, 13b, 70b}"
)
@click.option("--pm_batch_size", default=4, help="Batch size for the primary model")
@click.option(
    "--prompt_gen_temperature", default=0.7, help="Temperature for prompt generation"
)
@click.option("--ds_path", help="Path to the dataset")
@click.option(
    "--ds_downsample",
    default=None,
    type=int,
    help="Use at most this many rows of the dataset",
)
@click.option(
    "--n_clarifying_questions",
    default=1,
    help="Number of clarifying questions to generate",
)
@click.option(
    "--ds_shift", default=0, type=int, help="Shift the dataset by this many rows"
)
@click.option("--use_cache", default=False, help="Use the GPT-4 shelved cache")
@click.option(
    "--intermediate_results_path",
    default=None,
    help="Path to load results containing summaries and pm output. Skips directly to alpaca eval",
)
def main(
    pm_name,
    pm_size,
    pm_batch_size,
    prompt_gen_temperature,
    ds_path,
    ds_downsample,
    n_clarifying_questions,
    ds_shift,
    use_cache,
    intermediate_results_path,
):
    assert pm_name in ["gpt4", "llama2"]
    tqdm.pandas()
    np.random.seed(42)
    if intermediate_results_path is not None:
        # load intermediate results instead and skip directly to alpaca eval
        df = pd.read_json(intermediate_results_path)
    else:
        raise NotImplementedError("dont do this")

    pm = Llama2PrimaryModel(pm_size, pm_batch_size)
    df["vibes"] = pm.process(
        documents=df["doc_summ"], tasks=df["prompt"], answers=pd.Series([""] * len(df))
    )
    df_to_md(df, "vibes.md")
    # df.to_csv("vibes.tsv", sep="\t")
    print


if __name__ == "__main__":
    main()
