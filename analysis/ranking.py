# %%

import argparse
import pandas as pd
from models.summarization_models import GPTSummarizer
from models.prompt_generator_models import GPTPromptGenerator
from models.primary_models import (
    GPTPrimaryModel,
    Llama2PrimaryModel,
    Llama3PrimaryModel,
    PrimaryModel,
)
from models.cq_models import *
from models.oracle_models import GPTOracleAbstractiveModel, Llama3OracleModel
from models.ranking_models import (
    GPTClarifyingAnswersRankingModel,
    GPTPMOutputRankingModel,
    GPTPMPairwiseRankingModel,
)
from tqdm import tqdm
import numpy as np
from utils import df_to_md
import torch

# hugging face log in
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
import dotenv

hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
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
    "--oracle_batch_size", default=8, help="Batch size for the oracle model.", type=int
)
parser.add_argument(
    "--pm_batch_size", default=8, help="Batch size for the primary model.", type=int
)
parser.add_argument(
    "--cq_batch_size", default=8, help="Batch size for the clarifying question model.", type=int
)
parser.add_argument(
    "--bm_batch_size", default=8, help="Batch size for the benchmark model.", type=int
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
    "--n_clarifying_questions",
    default=1,
    help="Number of clarifying questions to generate.",
    type=int,
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
args = parser.parse_args()
# Testing: Set testing args
# args = parser.parse_args(args=[])
# args.ds_path = "/local/data/mt/toa-modeling/full_data/reddit_tldr_dataset.jsonl"
# args.ds_downsample = 3


# %% Loading Models

llamas = [
    x for x in [args.bm_name, args.pm_name, args.oracle_name] if "llama-3" in x.lower()
]
llama_pipelines = {
    "meta-llama/Meta-Llama-3-8B-Instruct": None,
    "meta-llama/Meta-Llama-3-70B-Instruct": None,
}
if "meta-llama/Meta-Llama-3-8B-Instruct" in llamas:
    llama_pipelines["meta-llama/Meta-Llama-3-8B-Instruct"] = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_api_key,
    )
if "meta-llama/Meta-Llama-3-70B-Instruct" in llamas:
    llama_pipelines["meta-llama/Meta-Llama-3-70B-Instruct"] = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_api_key,
    )

# %%

tqdm.pandas()
np.random.seed(42)
# if args.intermediate_results_path is not None:
#     # load intermediate results instead and skip directly to alpaca eval
#     df = pd.read_json(args.intermediate_results_path)
# else:
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
# summarize each item of the dataset to 50% of its original length
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
if "gpt" in args.pm_name.lower():
    primary_model = GPTPrimaryModel(args.pm_name, args.use_cache)
elif "llama-3" in args.pm_name.lower():
    primary_model = Llama3PrimaryModel(
        model_name=args.pm_name,
        batch_size=args.cq_batch_size,
        pipeline=llama_pipelines[args.pm_name],
    )
else:
    raise ValueError(f"Unknown primary model name {args.pm_name}")
# %%

###### CQ STEP ######

# Run the cq model
if "gpt" in args.bm_name:
    bm_cq_model = GPTClarifyingQuestionModel(args.bm_name, args.use_cache)
elif "Llama-3" in args.bm_name:
    bm_cq_model = Llama3ClarifyingQuestionModel(
        model_name=args.bm_name,
        batch_size=args.bm_batch_size,
        pipeline=llama_pipelines[args.bm_name],
    )
else:
    raise ValueError(f"Unknown benchmark model name {args.bm_name}")
print("running cq model...")

benchmark_cqs = bm_cq_model.forward_batch(
    df["doc_summ"], df["prompt"], args.n_clarifying_questions
)
for i in range(args.n_clarifying_questions):
    df[f"bm_cq_{i}"] = [cqs[i] for cqs in benchmark_cqs]

###### ORACLE STEP ######

if "gpt" in args.oracle_name.lower():
    oracle_model = GPTOracleAbstractiveModel(
        model_name=args.oracle_name, use_cache=args.use_cache
    )
elif "llama-3" in args.oracle_name.lower():
    oracle_model = Llama3OracleModel(
        model_name=args.oracle_name,
        batch_size=args.oracle_batch_size,
        pipeline=llama_pipelines[args.oracle_name],
    )
print("running abstractive oracle model to answer clarifying questions...")
# Ask the clarifying question to the oracle
for i in range(args.n_clarifying_questions):
    df[f"bm_ca_{i}"] = oracle_model.forward_batch(df["doc_full"], df[f"bm_cq_{i}"])

###### PRIMARY MODEL STEP ######

print("preparing instructions")
# Prepare instructions for the full example
df["pm_instruction_full"] = df.apply(
    lambda x: primary_model.prepare_instruction(x["doc_full"], x["prompt"]),
    axis=1,
)
# Prepare instructions for the summary example
df["pm_instruction_summ"] = df.apply(
    lambda x: primary_model.prepare_instruction(x["doc_summ"], x["prompt"]),
    axis=1,
)


# Prepare instructions for the answer inputs
for i in range(args.n_clarifying_questions):
    df[f"instructions_bm_ca_{i}"] = df.apply(
        lambda x: primary_model.prepare_ca_instruction(
            x["doc_summ"], x[f"bm_ca_{i}"], x["prompt"]
        ),
        axis=1,
    )

print(
    "preparing instructions for joint summ + ca contexts to be fed to the primary model"
)

print("running primary models for for joint summ + ca contexts")

# get answered, summary, and original outputs
# df["full_pm_output"] = primary_model.process_single(df["pm_instruction_full"])
# df["summ_pm_output"] = primary_model.process_single(df["pm_instruction_summ"])
for i in range(args.n_clarifying_questions):
    df[f"ca_{i}_pm_output"] = primary_model.process_single(
        df[f"instructions_bm_ca_{i}"]
    )

ranking_model = GPTPMPairwiseRankingModel(use_cache=args.use_cache)
opponents = [f"ca_{i}_pm_output" for i in range(args.n_clarifying_questions)] + [
    # "full_pm_output",
    # "summ_pm_output",
]

df["pref"] = df.progress_apply(
    lambda x: ranking_model.forward(
        x["prompt"], x["doc_full"], x["bm_ca_0"], x["bm_ca_1"]
    ),
    axis=1,
)

df["winner_cq"] = df.apply(lambda x: x["bm_cq_0"] if x["pref"] == 0 else x["bm_cq_1"], axis=1)

df["loser_cq"] = df.apply(lambda x: x["bm_cq_1"] if x["pref"] == 1 else x["bm_cq_0"], axis=1)

df["winner_ca"] = df.apply(lambda x: x["bm_ca_0"] if x["pref"] == 0 else x["bm_ca_1"], axis=1)

df["loser_ca"] = df.apply(lambda x: x["bm_ca_1"] if x["pref"] == 1 else x["bm_ca_0"], axis=1)

df["winner_pm_output"] = df.apply(lambda x: x["ca_0_pm_output"] if x["pref"] == 0 else x["ca_1_pm_output"], axis=1)

df["loser_pm_output"] = df.apply(lambda x: x["ca_1_pm_output"] if x["pref"] == 1 else x["ca_0_pm_output"], axis=1)

save_path = f"results/intermediate/pm-{args.pm_name.split('/')[-1]}_or-{args.oracle_name.split('/')[-1]}_{str(args.ds_downsample)}-pref.json"
df.to_json(save_path)
