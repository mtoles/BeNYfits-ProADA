from openai import OpenAI
from shelved_cache import PersistentCache
from cachetools import LRUCache
import cachetools
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

cache_filename = "shelved_cache/shelved_cache"
pc = PersistentCache(LRUCache, cache_filename, maxsize=10000)
client = OpenAI()


@cachetools.cached(pc)
def cached_openai_call(x: str, temperature=0.7, model="gpt-4"):
    """Call OpenAI's API, but cache the results in a shelved cache"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": x},
        ],
        temperature=temperature,
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
