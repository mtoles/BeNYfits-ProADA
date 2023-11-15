"""
Profile the number of tokens in the dataset 
"""


from transformers import LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import click
import json


@click.command()
@click.option("--dataset_name", type=str)
@click.option("--downsample_size", type=int, default=None)
def main(dataset_name, downsample_size):
    tokenizer = LlamaTokenizer.from_pretrained(
        "/local-scratch1/data/shared/llama/model_weights/tokenizer.model"
    )
    dataset = load_dataset(dataset_name)

    if dataset_name == "Yukang/LongAlpaca-12k":
        dataset = dataset["train"]
        if downsample_size is not None:
            dataset = dataset.select(range(downsample_size))
        instruction_lens = dataset.map(
            lambda x: {"len": len(tokenizer(x["instruction"])["input_ids"])}
        )
    elif dataset_name == "yahma/alpaca-cleaned":
        dataset = dataset["train"]
        if downsample_size is not None:
            dataset = dataset.select(range(downsample_size))
        instruction_lens = dataset.map(
            lambda x: {
                "len": len(tokenizer(x["instruction"])["input_ids"])
                + len(tokenizer(x["input"])["input_ids"])
            }
        )
    elif dataset_name == "mrm8488/unnatural-instructions-full":

        def get_len(x):
            length = len(
                tokenizer(x["instances"][0]["instruction_with_input"])["input_ids"]
            )
            return length

        dataset = dataset["train"]
        if downsample_size is not None:
            dataset = dataset.select(range(downsample_size))
        instruction_lens = dataset.map(lambda x: {"len": get_len(x)})
    elif dataset_name == "Muennighoff/natural-instructions":
        dataset = dataset["train"]
        if downsample_size is not None:
            dataset = dataset.select(range(downsample_size))
        instruction_lens = dataset.map(
            lambda x: {
                "len": len(tokenizer(x["definition"])["input_ids"])
                + len(tokenizer(x["inputs"])["input_ids"])
            }
        ) 
    else:
        raise NotImplementedError

    df = instruction_lens.flatten().to_pandas()
    bins = list(range(0, 16384, 512))
    g = sns.histplot(data=df, x="len", bins=bins + [float("inf")])
    g.set_xticks(bins[::2])
    g.set_xticklabels(bins[::2], rotation=45)
    num_in_ideal_range = df[(df["len"] >= 512) & (df["len"] < 4096)]
    percent_in_ideal_range = len(num_in_ideal_range) / len(df)
    print(f"Number of instructions in ideal range: {len(num_in_ideal_range)}")
    print(f"Percent of instructions in ideal range: {percent_in_ideal_range * 100}")

    # write the plot to a file
    plt.savefig(f"outputs/instruction_lens_{dataset_name.split('/')[-1]}.png")
    print


if __name__ == "__main__":
    main()
