"""
Determine if zero-shot models agree with human judgements on AITA posts
"""

import pandas as pd
import argparse
import lmwrapper
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, f1_score
from lmwrapper.structs import LmPrompt
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.huggingface_wrapper import get_huggingface_lm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scruples_json_path",
    help="path to the Scruples dataset",
)
parser.add_argument(
    "--downsample_size",
    help="number of samples to downsample to",
    type=int,
)

np.random.seed(42)

args = parser.parse_args()

df = pd.read_json(args.scruples_json_path, lines=True)
df = df.sample(frac=1)

# historical only
df = df[df["post_type"] == "HISTORICAL"]

# drop rows with less than 10 labels
min_labels = 10
min_majority = 0.0


def is_valid_example(x):
    n_right = x["binarized_label_scores"]["RIGHT"]
    n_wrong = x["binarized_label_scores"]["WRONG"]
    sufficient_count = n_right + n_wrong >= min_labels
    sufficient_majority = n_right / (n_right + n_wrong) >= min_majority
    return sufficient_count and sufficient_majority


len_before = len(df)
df = df[df.apply(is_valid_example, axis=1)]
print(
    f"dropped {len_before - len(df)}/{len_before} rows with less than {min_labels} labels or less than {100*min_majority}% majority"
)

if args.downsample_size:
    df = df.iloc[: args.downsample_size]

# fmt: off
# ### full doc
# determine_ah_prompt = (
#     "Read the following Reddit post and determine who is primarily at fault:\n\n"
#     "Title: {title}\n\n"
#     "Post: {doc}\n\n"
#     # "Who do you think is the asshole? If the speaker is the asshole, answer AUTHOR. If another person is the asshole, answer OTHER. If all parties are assholes, answer EVERYBODY. If nobody is an asshole, answer NOBODY. Do not say anything besides AUTHOR, OTHER, EVERYBODY, or NOBODY.\n\n"
#     "Who do you think is the most at fault? If the speaker is at primarily at fault, answer AUTHOR. If any other person is primarily at fault, answer OTHER. If nobody has done anything wrong whatsoever, answer NOBODY. If both the author and another party are equally at fault, answer EVERYBODY. Do not say anything besides AUTHOR, OTHER, NOBODY, or EVERYBODY.\n\n"
# )
# ### title only
determine_ah_prompt = (
    "Read the following summary and determine who is primarily at fault:\n\n"
    "Summary: {title}\n\n"
    # "Post: {doc}\n\n"
    # "Who do you think is the asshole? If the speaker is the asshole, answer AUTHOR. If another person is the asshole, answer OTHER. If all parties are assholes, answer EVERYBODY. If nobody is an asshole, answer NOBODY. Do not say anything besides AUTHOR, OTHER, EVERYBODY, or NOBODY.\n\n"
    "Who do you think is the most at fault? If the speaker is at primarily at fault, answer AUTHOR. If any other person is primarily at fault, answer OTHER. If nobody has done anything wrong whatsoever, answer NOBODY. If both the author and another party are equally at fault, answer EVERYBODY. Do not say anything besides AUTHOR, OTHER, NOBODY, or EVERYBODY.\n\n"
)
# fmt: on

# set up the OpenAI model with lmwrapper
lm = get_open_ai_lm(
    model_name=OpenAiModelNames.gpt_4o_2024_05_13,
)
# set up llama 3 model with lmwrapper
# lm = get_huggingface_lm("meta-llama/Meta-Llama-3-8B-Instruct")

generations = []
valid_generations = ["AUTHOR", "OTHER", "EVERYBODY", "NOBODY"]
for i, row in tqdm(df.iterrows()):
    prompt = determine_ah_prompt.format(
        title=row["action"],
        doc=row["text"],
    )
    generation = None
    i = 0
    max_tries = 3
    for i in range(max_tries):
        lm_prompt = LmPrompt(prompt, cache=True)
        generation = lm.predict(lm_prompt).completion_text.strip()
        if generation in valid_generations:
            break
        if i == max_tries - 1:
            generation = "ERROR"
    generations.append(generation)
df["model_label"] = generations

percent_agreement = (df["model_label"] == df["label"]).mean()
f1 = f1_score(df["model_label"], df["label"], average="macro")
kappa = cohen_kappa_score(df["model_label"], df["label"])
num_errors = (~df.model_label.isin(valid_generations)).sum()

print("percent agreement:", percent_agreement)
print("f1:", f1)
print("kappa:", kappa)
print("num errors:", num_errors)

# plot confusion matrix
true_labels = df["label"]
predicted_labels = df["model_label"]
cm = confusion_matrix(true_labels, predicted_labels, labels=valid_generations)

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=valid_generations,
    yticklabels=valid_generations,
)
plt.xlabel("Predicted")
plt.ylabel("True")
# save to png
plt.savefig("confusion_matrix.png")
print
