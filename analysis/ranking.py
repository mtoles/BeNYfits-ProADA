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
<<<<<<< HEAD
from models.cq_models import GPTClarifyingQuestionModel, GPTExperimentalClarifyingQuestionModel
=======
from models.cq_models import GPTClarifyingQuestionModel
>>>>>>> @{-1}
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

def print_current_device():
    if torch.cuda.is_available():
        print("Current Device: GPU")
    else:
        print("Current Device: CPU")

### FUNCTIONS ###
def prepare_ca_instructions(
    doc_summ: str, ca: list[str], prompt: str, primary_model: PrimaryModel
):
    instructions = []
    for clarifying_answer in ca:
        instructions.append(
            primary_model.prepare_instruction(
                "\n\n".join([doc_summ, clarifying_answer]), prompt
            )
        )
    return instructions


parser = argparse.ArgumentParser(description="Run model evaluations.")
parser.add_argument(
    "--pm_name", default="gpt-3.5-turbo", help="Name of the primary model to use."
)
parser.add_argument(
    "--oracle_name", default="gpt-3.5-turbo", help="Name of the oracle model to use."
)
parser.add_argument(
    "--oracle_size", default="8b", help="Size of the oracle model to use."
)
parser.add_argument(
    "--oracle_batch_size", default=4, help="Batch size for the oracle model.", type=int
)
parser.add_argument("--pm_size", default="7b", help="Size of the primary model to use.")
parser.add_argument(
    "--pm_batch_size", default=4, help="Batch size for the primary model.", type=int
)
parser.add_argument(
    "--prompt_gen_temperature", default=0.7, help="Temperature for prompt generation."
)
parser.add_argument("--ds_path", help="Path to the dataset.")
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

assert args.pm_name in ["gpt4", "llama2", "llama3", "gpt-3.5-turbo", "gpt-4-turbo"]
tqdm.pandas()
np.random.seed(42)
if args.intermediate_results_path is not None:
    # load intermediate results instead and skip directly to alpaca eval
    df = pd.read_json(args.intermediate_results_path)
else:
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
    df["prompt"] = df["doc_full"].progress_apply(
        lambda x: prompt_generator.forward(x, args.prompt_gen_temperature)
    )

    # Load the primary model
    if "gpt" in args.pm_name.lower():
        primary_model = GPTPrimaryModel(args.pm_name, args.use_cache)
    elif args.pm_name == "llama2":
        primary_model = Llama2PrimaryModel(args.pm_size, args.pm_batch_size)
    elif args.pm_name == "llama3":
        primary_model = Llama3PrimaryModel(args.pm_size, args.pm_batch_size)
    else:
        raise ValueError(f"Unknown primary model name {args.pm_name}")
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
    # Run the cq model
    cq_model = GPTClarifyingQuestionModel(args.use_cache)
    print("running cq model...")
    df["cq"] = df.progress_apply(
        lambda x: cq_model.forward(
            x["doc_summ"], x["prompt"], args.n_clarifying_questions
        ),
        axis=1,
    )
    if "gpt" in args.oracle_name.lower():
        oracle_model = GPTOracleAbstractiveModel(
            model_name=args.oracle_name, use_cache=args.use_cache
        )
    elif "llama3" in args.oracle_name.lower():
        oracle_model = Llama3OracleModel(
            model_size=args.oracle_size, batch_size=args.oracle_batch_size
        )
    print("running abstractive oracle model to answer clarifying questions...")
    # Ask the clarifying question to the oracle
    df["ca"] = df.progress_apply(
        lambda x: oracle_model.forward_list(x["doc_full"], x["cq"]), axis=1
    )

    print(
        "preparing instructions for joint summ + ca contexts to be fed to the primary model"
    )

    df["instructions_summ_ca"] = df.progress_apply(
        lambda x: prepare_ca_instructions(
            x["doc_summ"], x["ca"], x["prompt"], primary_model
        ),
        axis=1,
    )

    print("running primary models for for joint summ + ca contexts")

    # get answered, summary, and original outputs
    df["full_pm_output"] = primary_model.process_single(df["pm_instruction_full"])
    df["summ_pm_output"] = primary_model.process_single(df["pm_instruction_summ"])
    df["summ_ca_pm_outputs"] = primary_model.process_list(df["instructions_summ_ca"])

    # create a shuffled order of the outputs
    def shuffle_outputs(row) -> str:
        # worst to best
        pm_outputs_list = (
            [row["summ_pm_output"]]
            + row["summ_ca_pm_outputs"]
            + [row["full_pm_output"]]
        )
        ordering = np.random.permutation(len(pm_outputs_list))
        shuffled_outputs = [pm_outputs_list[i] for i in ordering]

        # reduce to two random indices
        ordering = np.random.choice(len(pm_outputs_list), 2, replace=False)
        return ordering, shuffled_outputs

    df["order"], df["pm_output_candidates"] = zip(
        *df.progress_apply(shuffle_outputs, axis=1)
    )

    pm_output_ranking_model = GPTPMOutputRankingModel(use_cache=args.use_cache)

    # get the preferences of the SHUFFLED candidates
    df["ranking"] = df.progress_apply(
        lambda x: pm_output_ranking_model.forward(
            x["doc_full"], x["prompt"], x["pm_output_candidates"], x["order"]
        ),
        axis=1,
    )

    # df["preference_ordering"] = df.progress_apply(
    #     lambda x: reconstruct(x["preference_ordering_shuffled"], x["order"]),
    # )

    # dump preferences to a json
    df.to_json(
        f"results/intermediate/pm-{args.pm_name}_or-{args.oracle_name}_{str(args.ds_downsample)}.json"
    )
