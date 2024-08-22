# %%

import argparse
import pandas as pd
from models.summarization_models import GPTSummarizer
from models.prompt_generator_models import GPTPromptGenerator
from models.primary_models import (
    GPTPrimaryModel,
    Llama3PrimaryModel,
    PrimaryModel,
    BasePrimaryModel
)
from models.cq_models import *
from models.oracle_models import GPTOracleAbstractiveModel, Llama3OracleModel, BaseOracleModel

from models.ranking_models import (
    GPTClarifyingAnswersRankingModel,
    GPTPMOutputRankingModel,
    GPTPMPairwiseRankingModel,
)
from tqdm import tqdm
import numpy as np
from utils import df_to_md
import torch
import json
from models.utils import load_lm

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
    default="full_data/reddit_tldr_dataset.jsonl",
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
# Testing: Set testing args
# args = parser.parse_args(args=[])
# args.ds_path = "/local/data/mt/toa-modeling/full_data/reddit_tldr_dataset.jsonl"
# args.ds_downsample = 3


# %% Loading Models

# llamas = [
#     x for x in [args.bm_name, args.pm_name, args.oracle_name] if "llama-3" in x.lower()
# ]
# llama_pipelines = {
#     "meta-llama/Meta-Llama-3-8B-Instruct": None,
#     "meta-llama/Meta-Llama-3-70B-Instruct": None,
# }
# if "meta-llama/Meta-Llama-3-8B-Instruct" in llamas:
#     llama_pipelines["meta-llama/Meta-Llama-3-8B-Instruct"] = get_huggingface_lm(
#         "meta-llama/Meta-Llama-3-8B-Instruct"
#     )
# if "meta-llama/Meta-Llama-3-70B-Instruct" in llamas:
#     llama_pipelines["meta-llama/Meta-Llama-3-70B-Instruct"] = get_huggingface_lm(
#         "meta-llama/Meta-Llama-3-70B-Instruct"
#     )
# for pipeline in llama_pipelines.values():
#     if pipeline is not None:
#         pipeline._tokenizer.pad_token_id = pipeline._tokenizer.eos_token_id
#         pipeline._tokenizer.padding_side = "left"


#

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
summarizer = GPTSummarizer(args.use_cache)
print("summarizing...")
if "doc_summ" not in df.columns:
    df["doc_summ"] = df["doc_full"].progress_apply(lambda x: summarizer.forward(x))
else:
    print("Skipping summarization because doc_summ is already present")

# Generate primary tasks
prompt_generator = GPTPromptGenerator(args.use_cache)
print("generating prompts...")
if "prompt" not in df.columns:
    df["prompt"] = df["doc_full"].progress_apply(
        lambda x: prompt_generator.forward(x, args.prompt_gen_temperature)
    )
else:
    print("Skipping prompt generation because it is already present")

# Load the primary model
pm_lm_wrapper = load_lm(args.pm_name)
primary_model = BasePrimaryModel(pm_lm_wrapper)

###### CQ STEP ######

# Run the cq model
if "gpt" in args.bm_name:
    bm_cq_model = GPTClarifyingQuestionModel(args.bm_name, args.use_cache)
elif "Llama-3" in args.bm_name:
    bm_lm_wrapper = load_lm(args.bm_name)
    bm_cq_model = Llama3ClarifyingQuestionModel(
        model_name=args.bm_name,
        batch_size=args.bm_batch_size,
        pipeline=bm_lm_wrapper.language_model,
    )
else:
    raise ValueError(f"Unknown benchmark model name {args.bm_name}")
print("running cq model...")

df["bm_cq"] = bm_cq_model.forward(df["doc_summ"], df["prompt"])

# generate cq, ca, output for experimental model
if args.cq_name == "gpt-cot":
    ex_cq_model = GPTCOTClarifyingQuestionModel(args.use_cache)
elif "gpt-4" in args.cq_name.lower():
    ex_cq_model = GPTClarifyingQuestionModel(args.use_cache)
elif "imaginellama" in args.cq_name:
    image_lm_wrapper = load_lm(args.cq_name.split(":")[-1])
    ex_cq_model = Llama3ImagineClarifyingQuestionModel(
        model_name=args.cq_name.split(":")[-1],
        batch_size=args.pm_batch_size,
        pipeline=image_lm_wrapper.language_model,
    )
elif "llama-3" in args.cq_name.lower():
    cq_lm_wrapper = load_lm(args.cq_name)
    ex_cq_model = Llama3ClarifyingQuestionModel(
        model_name=args.cq_name,
        batch_size=args.pm_batch_size,
        pipeline=cq_lm_wrapper.language_model,
    )
else:
    raise ValueError(f"Unknown experimental clarifying question model name {args.cq_name}")

print("running experimental cq model...")
df[f"ex_cq"] = ex_cq_model.forward(df["doc_summ"], df["prompt"])

###### ORACLE STEP ######
# Oracle Model should be declared independent of LM - GPT / Oracle
oracle_lm_wrapper = load_lm(args.oracle_name)
oracle_model = BaseOracleModel(oracle_lm_wrapper, args.oracle_batch_size)

print("running abstractive oracle model to answer clarifying questions...")

# Ask the clarifying question to the oracle
df["bm_ca"] = oracle_model.forward_batch(df["doc_full"], df["bm_cq"])
df[f"ex_ca"] = oracle_model.forward_batch(df["doc_full"], df["ex_cq"])

###### PRIMARY MODEL STEP ######
print("preparing instructions")
df["instructions_bm_ca"] = df.apply(
    lambda x: primary_model.prepare_ca_instruction(
        x["doc_summ"], x["bm_ca"], x["prompt"]
    ),
    axis=1,
)

print(
    "preparing instructions for joint summ + ca contexts to be fed to the primary model"
)
df["instructions_ex_ca"] = df.progress_apply(
    lambda x: primary_model.prepare_ca_instruction(
        x["doc_summ"], x["ex_ca"], x["prompt"]
    ),
    axis=1,
)

print("running primary models for for joint summ + ca contexts")

df["ca_bm_pm_output"] = primary_model.process_list(df["instructions_bm_ca"])
df["ca_ex_pm_output"] = primary_model.process_list(df["instructions_ex_ca"])

### Ranking ###
ranking_model = GPTPMPairwiseRankingModel(use_cache=args.use_cache)

df["pref_bm"] = df.progress_apply(
    lambda x: ranking_model.forward(
        x["prompt"], x["doc_full"], x["ca_ex_pm_output"], x["ca_bm_pm_output"]
    ),
    axis=1,
).apply(lambda x: "ex" if x == 0 else "bm")

# calculate total wins
wins = 0
win_rate_bm = (df["pref_bm"] == "ex").sum() / len(df)

### Record Results ###
print(win_rate_bm)
# dump preferences to a json
df_to_md(df.iloc[:10], "tmp.md")
save_path = os.path.join(
    "results",
    f"{now}_cq-{args.cq_name}_or-{args.oracle_name}_pm-{args.pm_name}_n={args.ds_downsample}.json".replace(
        "/", "-"
    ),
)
df.to_json(save_path)


def save_inputs_and_outputs():
    results = {
        "win_rate_bm": win_rate_bm,
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
    with open(f"../results/{now}_results.json", "w") as f:
        json.dump(results, f, indent=4)


save_inputs_and_outputs()
