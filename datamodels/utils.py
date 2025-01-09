import pandas as pd

df = pd.read_csv("dataset/benefits_clean.csv")
# save to jsonl
df.to_json("dataset/benefits_clean.jsonl", orient="records", lines=True)
