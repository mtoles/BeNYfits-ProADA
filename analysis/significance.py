import pandas as pd
import numpy as np
from scipy import stats
import os
import seaborn as sns

# df1 = pd.read_json(
#     "results/2024-07-01-00:05:35_cq-meta-llama-Meta-Llama-3-8B-Instruct_or-meta-llama-Meta-Llama-3-8B-Instruct_pm-meta-llama-Meta-Llama-3-8B-Instruct_n=200.json"
# )
# df2 = pd.read_json(
#     "results/2024-07-01-13:33:24_cq-imaginellama:meta-llama-Meta-Llama-3-8B-Instruct_or-meta-llama-Meta-Llama-3-8B-Instruct_pm-meta-llama-Meta-Llama-3-8B-Instruct_n=200.json"
# )

# wins1 = df1["pref_bm"] == "ex"
# wins2 = df2["pref_bm"] == "ex"

# # calculate statistical significance with a 1 tailed t-test
# t_stat, p_val = stats.ttest_ind(wins2, wins1, alternative="greater")
# print(f"t-statistic: {t_stat}, p-value: {p_val}")

### load data ##
e_dir0 = "results/e1"
e_dir1 = "results/e4"
# labels_df = pd.read_json("dataset/procedural_hh_dataset_1.0.1_annotated_50.jsonl", lines=True)[top_8_programs]

pred_dfs = [{}, {}]

for i, dir in enumerate([e_dir0, e_dir1]):
    for subdir in os.listdir(dir):
        program_abbrev = subdir.split("_")[-1]
        if os.path.isdir(os.path.join(dir, subdir)):
            for filepath in os.listdir(os.path.join(dir, subdir)):
                if filepath.endswith(".jsonl"):
                    df1 = pd.read_json(os.path.join(dir, subdir, filepath), lines=True).mean()
                    pred_dfs[i][program_abbrev] = df1
pass

for key in pred_dfs[0].keys():
    print(key)
    print(f"Backbone:  {pred_dfs[0][key]['f1'].mean()}")
    print(f"Notebook:  {pred_dfs[1][key]['f1'].mean()}")
    print(f"winner:    " + ("Backbone" if pred_dfs[0][key]['f1'].mean() > pred_dfs[1][key]['f1'].mean() else "Notebook"))
