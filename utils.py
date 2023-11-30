from openai import OpenAI
from shelved_cache import PersistentCache
from cachetools import LRUCache
import cachetools
from dotenv import load_dotenv

load_dotenv()

cache_filename = "shelved_cache/shelved_cache"
pc = PersistentCache(LRUCache, cache_filename, maxsize=10000)
client = OpenAI()


@cachetools.cached(pc)
def cached_openai_call(x: str, temperature=0.7, n=1, model="gpt-4"):
    """Call OpenAI's API, but cache the results in a shelved cache"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": x},
        ],
        n=n,
        temperature=temperature,
    )
    return completion
