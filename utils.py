from openai import OpenAI
from shelved_cache import PersistentCache
from cachetools import LRUCache
import cachetools
from dotenv import load_dotenv
import pandas as pd
from openai._types import NotGiven
from typing import Optional


load_dotenv()

cache_filename = "shelved_cache/shelved_cache"
pc = PersistentCache(LRUCache, cache_filename, maxsize=10000)
client = OpenAI()


# @cachetools.cached(pc)
# def cached_openai_call(
def openai_call(
    x: str,
    model: str,
    response_format: Optional[str] = None,
    temperature: float = 0.7,
    use_cache: bool = False,  # unused, here for convenience
):
    """
    Call OpenAI's API, but cache the results in a shelved cache

    Parameters:
        x (str): the input to the model
        model (str): the name of the OpenAI model to use
        response_format (str): the response format to use {"json", None}
        temperature (float): the temperature to use for the GPT model

    Returns:
        str: the output of the GPT model
    """
    assert response_format in [None, "json"]
    response_format = (
        {"type": "json_object"} if response_format == "json" else NotGiven()
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": x},
        ],
        temperature=temperature,
        response_format=response_format,
    )
    return completion


@cachetools.cached(pc)
def cached_openai_call(*args, **kwargs):
    """
    Call OpenAI API USING the shelved cache
    """
    return openai_call(*args, **kwargs)


def uncached_openai_call(*args, **kwargs):
    """
    Call OpenAI API WITHOUT using the shelved cache
    """
    return openai_call(*args, **kwargs)


def conditional_openai_call(*args, **kwargs):
    """
    Call OpenAI API conditional on the values of `kwargs["use_cache"]`
    """
    if kwargs["use_cache"]:
        return cached_openai_call(*args, **kwargs)
    else:
        return uncached_openai_call(*args, **kwargs)


def df_to_md(df: pd.DataFrame, output_path: str):
    """
    Convert a dataframe to a markdown table and save to disk.

    Parameters:
        df (pd.DataFrame): the dataframe to convert. Must have columns "subreddit", "doc_orig", "doc_summ", "prompt", "pm_answer_full", "pm_answer_summ", "selection"
        output_path (str): the path to save the markdown file
    """
    # delete the existing file and create a new one
    with open(output_path, "w") as f:
        col_to_header = {
            "subreddit": "subreddit",
            "doc_orig": "original document",
            "doc_summ": "summary document",
            "prompt": "prompt",
            "pm_answer_full": "full answer",
            "pm_answer_summ": "summary answer",
            "cq": "clarifying question",
            "selection": "selection",
        }
        # substrs = [
        #     f"## subreddit",
        #     row["subreddit"],
        #     f"## doc_orig",
        #     row["doc_orig"],
        #     f"## doc_summ",
        #     row["doc_summ"],
        #     f"## prompt",
        #     row["prompt"],
        #     f"## pm_answer_full",
        #     row["pm_answer_full"],
        #     f"## pm_answer_summ",
        #     row["pm_answer_summ"],
        #     f"## clarifying question",
        #     row["cq"],
        #     f"selection: {row['selection']}",
        #     f"\n\n{'='*50}\n\n",
        # ]
        substrs = []
        for col, header in col_to_header.items():
            if col in df.columns:
                substrs.append(f"## {header}")
                substrs.append(df[col])
        substrs.append(f"\n\n{'='*50}\n\n")
        # only include each substrs pair if the column exists in the dataframe
        for row in df.iloc:
            md_row = "\n\n".join(substrs)
            f.write(md_row)
