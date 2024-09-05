# %%

import argparse
import pandas as pd
from models.summarization_models import GPTSummarizer
from models.prompt_generator_models import *
from models.primary_models import (
    BasePrimaryModel,
)
from models.cq_models import *
from models.oracle_models import (
    BaseOracleModel,
)

from models.ranking_models import (
    GPTPMPairwiseRankingModel,
)
from tqdm import tqdm
import numpy as np
from utils import df_to_md
import torch
import json
from models.utils import load_lm
from sklearn.metrics import cohen_kappa_score, f1_score

# hugging face log in
import os
from dotenv import load_dotenv

# from huggingface_hub import HfApi
import dotenv
from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
# hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
torch.manual_seed(0)


def print_current_device():
    if torch.cuda.is_available():
        print("Current Device: GPU")
    else:
        print("Current Device: CPU")


parser = argparse.ArgumentParser(description="Run model evaluations.")
parser.add_argument(
    "--bm_name",
    default="gpt-4-0125-preview",
    help="Name of the benchmark model to use.",
)
parser.add_argument(
    "--cq_name",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Name of the experimental clarifying question model to use.",
)
parser.add_argument(
    "--cq_ex_mode",
    default="default",
    help="Mode of prompting CQ Experimental Model - Default / COT",
)
parser.add_argument(
    "--pm_name",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Name of the primary model to use.",
)
parser.add_argument(
    "--oracle_name",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Name of the oracle model to use.",
)
parser.add_argument(
    "--oracle_batch_size", default=1, help="Batch size for the oracle model.", type=int
)
parser.add_argument(
    "--pm_batch_size", default=1, help="Batch size for the primary model.", type=int
)
parser.add_argument(
    "--cq_batch_size",
    default=1,
    help="Batch size for the clarifying question model.",
    type=int,
)
parser.add_argument(
    "--bm_batch_size", default=1, help="Batch size for the benchmark model.", type=int
)
parser.add_argument(
    "--prompt_gen_temperature", default=0.7, help="Temperature for prompt generation."
)
parser.add_argument(
    "--ds_path",
    help="Path to the dataset.",
    # default="dataset/reddit_tldr_dataset.jsonl",
)
parser.add_argument(
    "--ds_downsample",
    default=None,
    type=int,
    help="Use at most this many rows of the dataset.",
)
parser.add_argument(
    "--ds_shift", default=0, type=int, help="Shift the dataset by this many rows."
)
parser.add_argument("--use_cache", default=True, help="Use the GPT-4 shelved cache.")
parser.add_argument(
    "--intermediate_results_path",
    default=None,
    help="Path to load intermediate results.",
)
parser.add_argument(
    "--manual_note",
    default=None,
    help="Manual note to add to the results file for experiment tracking",
)
args = parser.parse_args()

print_current_device()

tqdm.pandas()
np.random.seed(42)

# Read docs from the dataset
if args.ds_path.lower().endswith(".jsonl"):
    df = pd.read_json(args.ds_path, lines=True)
elif args.ds_path.lower().endswith(".json"):
    df = pd.read_json(args.ds_path, lines=False)
elif args.ds_path.lower().endswith(".csv"):
    df = pd.read_csv(args.ds_path)
if "selftext" in df.columns:
    df["doc_full"] = df["selftext"].astype(str)
    df = df.drop(columns=["selftext"])

# Shuffle the datast
df = df.sample(frac=1).reset_index(drop=True)

# Shift the dataset
df = pd.concat([df.iloc[args.ds_shift :], df.iloc[: args.ds_shift]]).reset_index(
    drop=True
)

# Apply downsampling
if args.ds_downsample is not None:
    df = df.head(args.ds_downsample)

# Summarize each item of the dataset to 50% of its original length
# summarizer = GPTSummarizer(args.use_cache)
# print("summarizing...")
# if "doc_summ" not in df.columns:
#     df["doc_summ"] = df["doc_full"].progress_apply(lambda x: summarizer.forward(x))
# else:
#     print("Skipping summarization because doc_summ is already present")
df = df.rename(columns={"title": "doc_summ", "text": "doc_full"})


# Generate primary tasks
# prompt_generator = GPTPromptGenerator(args.use_cache)
# prompt_generator = StaticAdviceGenerator()
# print("generating prompts...")
# if "prompt" not in df.columns:
#     df["prompt"] = df["doc_full"].progress_apply(
#         # lambda x: prompt_generator.forward(x, args.prompt_gen_temperature)
#         lambda x: prompt_generator.forward(x)
#     )
# else:
#     print("Skipping prompt generation because it is already present")
# df["prompt"] = (
#     "Read the following summary and determine who is primarily at fault:\n\n"
#     "Summary: {title}\n\n"
#     # "Post: {doc}\n\n"
#     # "Who do you think is the asshole? If the speaker is the asshole, answer AUTHOR. If another person is the asshole, answer OTHER. If all parties are assholes, answer EVERYBODY. If nobody is an asshole, answer NOBODY. Do not say anything besides AUTHOR, OTHER, EVERYBODY, or NOBODY.\n\n"
#     "Who do you think is the most at fault? If the speaker is at primarily at fault, answer AUTHOR. If any other person is primarily at fault, answer OTHER. If nobody has done anything wrong whatsoever, answer NOBODY. If both the author and another party are equally at fault, answer EVERYBODY. Do not say anything besides AUTHOR, OTHER, NOBODY, or EVERYBODY.\n\n"
# )

###### CQ STEP ######

print("running cq model...")

cq_bm_lm_wrapper = load_lm(args.bm_name)
cq_bm_model = BaseClarifyingQuestionModel(cq_bm_lm_wrapper)

df["bm_cq"] = cq_bm_model.forward_batch_generate_single_question(df["doc_summ"])

# generate cq, ca, output for experimental model
print("running experimental cq model...")
if args.cq_ex_mode == "cot":
    # TODO - Write generic class for COT
    ex_cq_model = GPTCOTClarifyingQuestionModel(args.use_cache)
else:
    cq_ex_model_wrapper = load_lm(args.cq_name)
    ex_cq_model = BaseClarifyingQuestionModel(cq_ex_model_wrapper)

df[f"ex_cq"] = ex_cq_model.forward_batch_generate_single_question(df["doc_summ"])

###### ORACLE STEP ######
oracle_lm_wrapper = load_lm(args.oracle_name)
oracle_model = BaseOracleModel(oracle_lm_wrapper, args.oracle_batch_size)

print("running abstractive oracle model to answer clarifying questions...")

# Ask the clarifying question to the oracle
df["bm_ca"] = oracle_model.forward_batch(df["doc_full"], df["bm_cq"])
df[f"ex_ca"] = oracle_model.forward_batch(df["doc_full"], df["ex_cq"])

###### PRIMARY MODEL STEP ######
# Load the primary model
pm_lm_wrapper = load_lm(args.pm_name)
primary_model = BasePrimaryModel(pm_lm_wrapper)

print("preparing instructions")
df["instructions_bm_ca"] = df.apply(
    lambda x: primary_model.prepare_ca_instruction(x["doc_summ"], x["bm_ca"]),
    axis=1,
)

print(
    "preparing instructions for joint summ + ca contexts to be fed to the primary model"
)
df["instructions_ex_ca"] = df.progress_apply(
    lambda x: primary_model.prepare_ca_instruction(x["doc_summ"], x["ex_ca"]),
    axis=1,
)

print("running primary models for for joint summ + ca contexts")

df["ca_bm_pm_output"] = primary_model.process_list(df["instructions_bm_ca"])
df["ca_ex_pm_output"] = primary_model.process_list(df["instructions_ex_ca"])

### EVALUATION ###

valid_generations = ["AUTHOR", "OTHER", "EVERYBODY", "NOBODY"]
percent_agreement_bm = (df["ca_bm_pm_output"] == df["label"]).mean()
f1_bm = f1_score(df["ca_bm_pm_output"], df["label"], average="macro")
kappa_bm = cohen_kappa_score(df["ca_bm_pm_output"], df["label"])
num_errors_bm = (~df["ca_bm_pm_output"].isin(valid_generations)).sum()

percent_agreement_ex = (df["ca_ex_pm_output"] == df["label"]).mean()
f1_ex = f1_score(df["ca_ex_pm_output"], df["label"], average="macro")
kappa_ex = cohen_kappa_score(df["ca_ex_pm_output"], df["label"])
num_errors_ex = (~df["ca_bm_pm_output"].isin(valid_generations)).sum()

### RECORD RESULTS ###

save_path = os.path.join(
    "results",
    f"{now}_cq-{args.cq_name}_or-{args.oracle_name}_pm-{args.pm_name}_n={args.ds_downsample}.json".replace(
        "/", "-"
    ),
)
df.to_json(save_path)


def save_inputs_and_outputs():
    results = {
        "f1 bm": f1_bm,
        "kappa bm": kappa_bm,
        "percent agreement bm": percent_agreement_bm,
        "num errors bm": num_errors_bm,
        "f1 ex": f1_ex,
        "kappa ex": kappa_ex,
        "percent agreement ex": percent_agreement_ex,
        "num errors ex": num_errors_ex,
        "bm_name": args.bm_name,
        "cq_name": args.cq_name,
        "pm_name": args.pm_name,
        "oracle_name": args.oracle_name,
        "ds_path": args.ds_path,
        "ds_downsample": args.ds_downsample,
        "prompt_gen_temperature": args.prompt_gen_temperature,
        "use_cache": args.use_cache,
        "manual_note": args.manual_note,
        "start_datetime": now,
    }
    with open(f"results/{now}_results.json", "w") as f:
        # make everything a string
        results = {k: str(v) for k, v in results.items()}
        json.dump(results, f, indent=4)


save_inputs_and_outputs()
