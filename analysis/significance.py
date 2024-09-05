import pandas as pd
import numpy as np
from scipy import stats

df1 = pd.read_json(
    "results/2024-07-01-00:05:35_cq-meta-llama-Meta-Llama-3-8B-Instruct_or-meta-llama-Meta-Llama-3-8B-Instruct_pm-meta-llama-Meta-Llama-3-8B-Instruct_n=200.json"
)
df2 = pd.read_json(
    "results/2024-07-01-13:33:24_cq-imaginellama:meta-llama-Meta-Llama-3-8B-Instruct_or-meta-llama-Meta-Llama-3-8B-Instruct_pm-meta-llama-Meta-Llama-3-8B-Instruct_n=200.json"
)

wins1 = df1["pref_bm"] == "ex"
wins2 = df2["pref_bm"] == "ex"

# calculate statistical significance with a 1 tailed t-test
t_stat, p_val = stats.ttest_ind(wins2, wins1, alternative="greater")
print(f"t-statistic: {t_stat}, p-value: {p_val}")
