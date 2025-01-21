import pandas as pd
import numpy as np
from scipy import stats
import os
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

exp_dir = "results/paper/2025-01-19_20:55:54_82_4o"
ctl_dir = "results/paper/2025-01-17_13:02:35_82_backbone"

exp_df = pd.read_json(os.path.join(exp_dir, "predictions.jsonl"), lines=True)
ctrl_df = pd.read_json(os.path.join(ctl_dir, "predictions.jsonl"), lines=True)
exp_df = exp_df.astype(int)
ctrl_df = ctrl_df.astype(int)
ctrl_labels = pd.read_json(os.path.join(ctl_dir, "labels.jsonl"), lines=True)
exp_labels = pd.read_json(os.path.join(exp_dir, "labels.jsonl"), lines=True)

ctrl_correct = (exp_df.to_numpy().flatten() == ctrl_df.to_numpy().flatten()).astype(int)
exp_correct = (exp_df.to_numpy().flatten() == exp_labels.to_numpy().flatten()).astype(
    int
)

n = len(exp_correct)
# chi squared test

# Example binary data
# Group 1: 20 successes out of 50 trials
success1, n1 = ctrl_correct.sum(), n

# Group 2: 15 successes out of 40 trials
success2, n2 = exp_correct.sum(), n

# Perform z-test for proportions
count = [success1, success2]
nobs = [n1, n2]
z_stat, p_value = proportions_ztest(count, nobs)

print(f"control accuracy: {ctrl_correct.mean()}")
print(f"experiment accuracy: {exp_correct.mean()}")
print("Z-statistic:", z_stat)
print("p-value:", p_value)
print
