# Setup
from openai import OpenAI
import pandas as pd
import json
from models.summarization_models import GPTSummarizer
from prompt_generator_models import GPTPromptGenerator
from primary_models import GPTPrimaryModel
from tqdm import tqdm
from shelved_cache import PersistentCache
from cachetools import LRUCache
import cachetools
import click


@click.command()
@click.option("--n_prompts", default=3, help="Number of prompts to generate")
@click.option(
    "--prompt_gen_temperature", default=0.7, help="Temperature for prompt generation"
)
def main(n_prompts, prompt_gen_temperature):
    # for persistent caching
    cache_filename = "shelved_cache/shelved_cache"
    pc = PersistentCache(LRUCache, cache_filename, maxsize=10000)

    tqdm.pandas()
    client = OpenAI()

    # Read docs from the dataset
    ds_path = "data/prompting_mini_dataset.json"
    with open(ds_path, "r") as f:
        ds_json = json.load(f)
        df = pd.DataFrame(ds_json)
    # df = df.tail(1)
    # summarize each item of the dataset to 50% of its original length
    summarizer = GPTSummarizer()

    # make the summarizer cacheable
    # @cachetools.cached(pc)
    def generate_summary(x):
        return summarizer.forward(x)

    df["summary"] = df["original_document"].progress_apply(
        lambda x: generate_summary(x)
    )

    # Generate primary tasks
    prompt_generator = GPTPromptGenerator()

    # make the summarizer cacheable
    # @cachetools.cached(pc)
    def generate_prompts(x):
        return prompt_generator.forward(x, n_prompts, prompt_gen_temperature)

    df["prompts"] = df["original_document"].progress_apply(
        lambda x: generate_prompts(x)
    )

    # Run the primary task
    def generate_pm_answer_full(x):
        primary_model = GPTPrimaryModel()
        return primary_model.forward(
            x["original_document"], x["prompts"][0], temperature=0.7
        )

    df["pm_answer_full"] = df.progress_apply(
        lambda x: generate_pm_answer_full(x), axis=1
    )

    def generate_pm_answer_summ(x):
        primary_model = GPTPrimaryModel()
        return primary_model.forward(x["summary"], x["prompts"][0], temperature=0.7)

    df["pm_answer_summ"] = df.progress_apply(
        lambda x: generate_pm_answer_summ(x), axis=1
    )

    print()


if __name__ == "__main__":
    main()
