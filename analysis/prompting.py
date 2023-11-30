# Setup
from openai import OpenAI
import pandas as pd
import json
from models.summarization_models import GPTSummarizer
from prompt_generator_models import GPTPromptGenerator
from primary_models import GPTPrimaryModel, Llama2PrimaryModel
from reward_models import GPTRewardModel
from tqdm import tqdm
import click
import numpy as np


@click.command()
@click.option("--pm_name", default="llama2", help="Name of the primary model to use")
@click.option("--n_prompts", default=1, help="Number of prompts to generate")
@click.option(
    "--prompt_gen_temperature", default=0.7, help="Temperature for prompt generation"
)
@click.option("--use_cache", default=False, help="Use the GPT-4 shelved cache")
@click.option(
    "--downsample_size",
    default=None,
    type=int,
    help="Use at most this many rows of the dataset",
)
def main(pm_name, n_prompts, prompt_gen_temperature, use_cache, downsample_size):
    assert pm_name in ["gpt4", "llama2"]
    tqdm.pandas()
    np.random.seed(42)

    # Read docs from the dataset
    # ds_path = "data/prompting_mini_dataset.json"
    ds_path = "data/prompting_mini_dataset_2.csv"
    if ds_path.lower().endswith(".json"):
        with open(ds_path, "r") as f:
            ds_json = json.load(f)
            df = pd.DataFrame(ds_json)
    elif ds_path.lower().endswith(".csv"):
        df = pd.read_csv(ds_path)
    # Apply downsampling
    if downsample_size is not None:
        df = df.head(downsample_size)
    # summarize each item of the dataset to 50% of its original length
    summarizer = GPTSummarizer()

    df["doc_summ"] = df["doc_orig"].progress_apply(lambda x: summarizer.forward(x))

    # Generate primary tasks
    prompt_generator = GPTPromptGenerator()

    df["prompts"] = df["doc_orig"].progress_apply(
        lambda x: prompt_generator.forward(x, n_prompts, prompt_gen_temperature)
    )

    # Run the primary task
    if pm_name == "gpt4":
        primary_model = GPTPrimaryModel()
    elif pm_name == "llama2":
        primary_model = Llama2PrimaryModel()
    else:
        raise ValueError(f"Unknown primary model name {pm_name}")
    df["pm_answer_full"] = df.progress_apply(
        lambda x: primary_model.forward(x["doc_orig"], x["prompts"][0]),
        axis=1,
    )

    df["pm_answer_summ"] = df.progress_apply(
        lambda x: primary_model.forward(x["doc_summ"], x["prompts"][0]),
        axis=1,
    )

    # Compare the summary answers and full answers using the reward model
    df.to_csv("llama2.csv")
    reward_model = GPTRewardModel()
    df["selection"] = df.progress_apply(
        lambda x: reward_model.forward(
            x["doc_orig"],
            x["pm_answer_full"],
            x["pm_answer_summ"],
            x["prompts"][0],
            temperature=0.7,
        ),
        axis=1,
    )
    num_full_selected = len(df[df["selection"] == "full"])
    percent_full_selected = num_full_selected / len(df)
    print(
        f"Percent of full docs selected: {percent_full_selected} | {num_full_selected} / {len(df)}"
    )
    print()


if __name__ == "__main__":
    main()
