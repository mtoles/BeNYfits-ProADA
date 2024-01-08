from openai import OpenAI
from shelved_cache import PersistentCache
from cachetools import LRUCache
import cachetools
from dotenv import load_dotenv
import pandas as pd
from openai._types import NotGiven

load_dotenv()

cache_filename = "shelved_cache/shelved_cache"
pc = PersistentCache(LRUCache, cache_filename, maxsize=10000)
client = OpenAI()


@cachetools.cached(pc)
def cached_openai_call(
    x: str,  model, response_format=None, temperature=0.7,
):
    """Call OpenAI's API, but cache the results in a shelved cache"""
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


def df_to_md(df: pd.DataFrame, output_path: str):
    """Convert a dataframe to a markdown table"""
    # delete the existing file and create a new one
    with open(output_path, "w") as f:
        for row in df.iloc:
            md_row = "\n\n".join(
                [
                    f"## subreddit",
                    row["subreddit"],
                    f"## doc_orig",
                    row["doc_orig"],
                    f"## doc_summ",
                    row["doc_summ"],
                    f"## prompt",
                    row["prompt"],
                    f"## pm_answer_full",
                    row["pm_answer_full"],
                    f"## pm_answer_summ",
                    row["pm_answer_summ"],
                    f"preference: {row['preference']}",
                    f"\n\n{'='*50}\n\n",
                ]
            )
            f.write(md_row)
