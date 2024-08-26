"""
Prep:
    Run `python3 preprocessing/get_data.py to get the dataset`

Make two train-test splits:
    1. Post sampling: Assign 20% of all examples to the test set
    2. Subreddit sampling: Assign 20% of each subreddit to the test set
"""

import pandas as pd
import numpy as np

full_data_path = "full_data/reddit_tldr_dataset.jsonl"

# Load the dataset
df = pd.read_json(full_data_path, lines=True)

# Post sampling
df_post = df.sample(frac=0.8)
df_test_post = df.drop(df_post.index)
df_train_post = df[df.index.isin(df_post.index)]

# Subreddit sampling
subreddits = np.array(df["subreddit"].unique())
np.random.shuffle(subreddits)
test_subreddits = subreddits[: len(subreddits) // 5]
df_test_subreddit = df[df["subreddit"].isin(test_subreddits)]
df_train_subreddit = df[~df["subreddit"].isin(test_subreddits)]

# Save the splits
df_train_post.to_json("full_data/train_post.jsonl", orient="records", lines=True)
df_test_post.to_json("full_data/test_post.jsonl", orient="records", lines=True)
df_train_subreddit.to_json(
    "full_data/train_subreddit.jsonl", orient="records", lines=True
)
df_test_subreddit.to_json(
    "full_data/test_subreddit.jsonl", orient="records", lines=True
)
