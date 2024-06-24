# Setup
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
import click
import numpy as np
from utils import df_to_md


@click.command()
@click.option(
    "--bm_name",
    help="Name of the benchmark model to use. If using a gpt model, use the exact api call name, e.g., 'gpt-3.5-turbo'",
)
@click.option(
    "--pm_name",
    default="gpt-3.5-turbo",
    help="Name of the primary model to use. If using a gpt model, use the exact api call name, e.g., 'gpt-3.5-turbo', 'llama2'",
)
@click.option(
    "--oracle_name",
    default="gpt-3.5-turbo",
    help="Name of the primary model to use. If using a gpt model, use the exact api call name, e.g., 'gpt-3.5-turbo'",
)
@click.option("--oracle_size", default="8b", help="Size of the oracle model to use")
@click.option("--oracle_batch_size", default=4, help="Batch size for the oracle model")
@click.option("--pm_batch_size", default=4, help="Batch size for the primary model")
@click.option(
    "--prompt_gen_temperature", default=0.7, help="Temperature for prompt generation"
)
@click.option("--ds_path", help="Path to the dataset")
@click.option(
    "--ds_downsample",
    default=None,
    type=int,
    help="Use at most this many rows of the dataset",
)
@click.option(
    "--n_clarifying_questions",
    default=1,
    help="Number of clarifying questions to generate",
)
@click.option(
    "--ds_shift", default=0, type=int, help="Shift the dataset by this many rows"
)
@click.option("--use_cache", default=True, help="Use the GPT-4 shelved cache")
@click.option(
    "--intermediate_results_path",
    default=None,
    help="Path to load results containing summaries and pm output. Skips directly to alpaca eval",
)
def main(
    bm_name,
    pm_name,
    oracle_name,
    oracle_size,
    oracle_batch_size,
    pm_batch_size,
    prompt_gen_temperature,
    ds_path,
    ds_downsample,
    n_clarifying_questions,
    ds_shift,
    use_cache,
    intermediate_results_path,
):
    # assert pm_name in ["gpt4", "llama2", "llama3", "gpt-3.5-turbo", "gpt-4-turbo"]
    # check if we can share weights between llama3 models
    llamas = [x for x in [bm_name, pm_name, oracle_name] if "llama-3" in x.lower()]
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
        )
    if "meta-llama/Meta-Llama-3-70B-Instruct" in llamas:
        llama_pipelines["meta-llama/Meta-Llama-3-70B-Instruct"] = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    tqdm.pandas()
    np.random.seed(42)
    if intermediate_results_path is not None:
        # load intermediate results instead and skip directly to alpaca eval
        df = pd.read_json(intermediate_results_path)
    else:
        # Read docs from the dataset
        if ds_path.lower().endswith(".jsonl"):
            df = pd.read_json(ds_path, lines=True)
        elif ds_path.lower().endswith(".json"):
            df = pd.read_json(ds_path, lines=False)
        elif ds_path.lower().endswith(".csv"):
            df = pd.read_csv(ds_path)
        if "selftext" in df.columns:
            df["doc_full"] = df["selftext"].astype(str)
            df = df.drop(columns=["selftext"])
        # Shuffle the datast
        df = df.sample(frac=1).reset_index(drop=True)
        # Shift the dataset
        df = pd.concat([df.iloc[ds_shift:], df.iloc[:ds_shift]]).reset_index(drop=True)
        # Apply downsampling
        if ds_downsample is not None:
            df = df.head(ds_downsample)
        # summarize each item of the dataset to 50% of its original length
        summarizer = GPTSummarizer(use_cache)
        print("summarizing...")
        if "doc_summ" not in df.columns:
            df["doc_summ"] = df["doc_full"].progress_apply(
                lambda x: summarizer.forward(x)
            )
        else:
            print("Skipping summarization because doc_summ is already present")

        # Generate primary tasks
        prompt_generator = GPTPromptGenerator(use_cache)
        print("generating prompts...")
        df["prompt"] = df["doc_full"].progress_apply(
            lambda x: prompt_generator.forward(x, prompt_gen_temperature)
        )

        # Load the primary model
        if "gpt" in pm_name.lower():
            primary_model = GPTPrimaryModel(pm_name, use_cache)
        elif "llama-3" in pm_name.lower():
            primary_model = Llama3PrimaryModel(
                model_name=pm_name,
                batch_size=pm_batch_size,
                pipeline=llama_pipelines[pm_name],
            )
        else:
            raise ValueError(f"Unknown primary model name {pm_name}")

        ###### CQ STEP ######

        # Run the cq model
        if "gpt" in bm_name:
            bm_cq_model = GPTClarifyingQuestionModel(bm_name, use_cache)
        elif "Llama-3" in bm_name:
            bm_cq_model = Llama3ClarifyingQuestionModel(
                model_name=bm_name,
                batch_size=pm_batch_size,
                pipeline=llama_pipelines[bm_name],
            )
        else:
            raise ValueError(f"Unknown benchmark model name {bm_name}")
        print("running cq model...")
        # benchmark_cqs = df.progress_apply(
        #     lambda x: bm_cq_model.forward(
        #         x["doc_summ"], x["prompt"], n_clarifying_questions
        #     ),
        #     axis=1,
        # )
        benchmark_cqs = bm_cq_model.forward_batch(
            df["doc_summ"], df["prompt"], n_clarifying_questions
        )
        for i in range(n_clarifying_questions):
            df[f"bm_cq_{i}"] = [cqs[i] for cqs in benchmark_cqs]

        # generate cq, ca, output for experimental model
        # ex_cq_model = GPTClarifyingQuestionModel(use_cache)
        ex_cq_model = GPTCOTClarifyingQuestionModel(use_cache)
        # df[f"ex_cq"] = df.progress_apply(
        #     lambda x: ex_cq_model.forward(x["doc_summ"], x["prompt"]),
        #     axis=1,
        # )
        df[f"ex_cq"] = ex_cq_model.forward_batch(df["doc_summ"], df["prompt"])
        print("running experimental cq model...")

        ###### ORACLE STEP ######

        if "gpt" in oracle_name.lower():
            oracle_model = GPTOracleAbstractiveModel(
                model_name=oracle_name, use_cache=use_cache
            )
        elif "llama-3" in oracle_name.lower():
            oracle_model = Llama3OracleModel(
                model_name=oracle_name,
                batch_size=oracle_batch_size,
                pipeline=llama_pipelines[oracle_name],
            )
        print("running abstractive oracle model to answer clarifying questions...")
        # Ask the clarifying question to the oracle
        for i in range(n_clarifying_questions):
            # df[f"bm_ca_{i}"] = df.progress_apply(
            #     lambda x: oracle_model.forward_single(x["doc_full"], x[f"bm_cq_{i}"]),
            #     axis=1,
            # )
            df[f"bm_ca_{i}"] = oracle_model.forward_batch(
                df["doc_full"], df[f"bm_cq_{i}"]
            )
        # df[f"ex_ca"] = df.progress_apply(
        #     lambda x: oracle_model.forward_single(x["doc_full"], x["ex_cq"]), axis=1
        # )
        df[f"ex_ca"] = oracle_model.forward_batch(df["doc_full"], df["ex_cq"])

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
        for i in range(n_clarifying_questions):
            df[f"instructions_bm_ca_{i}"] = df.apply(
                lambda x: primary_model.prepare_ca_instruction(
                    x["doc_summ"], x[f"bm_ca_{i}"], x["prompt"]
                ),
                axis=1,
            )
        df["instructions_ex_ca"] = df.apply(
            lambda x: primary_model.prepare_ca_instruction(
                x["doc_summ"], x["ex_ca"], x["prompt"]
            ),
            axis=1,
        )

        print("running primary models for for joint summ + ca contexts")

        # get answered, summary, and original outputs
        df["full_pm_output"] = primary_model.process_single(df["pm_instruction_full"])
        df["summ_pm_output"] = primary_model.process_single(df["pm_instruction_summ"])
        for i in range(n_clarifying_questions):
            df[f"ca_{i}_pm_output"] = primary_model.process_single(
                df[f"instructions_bm_ca_{i}"]
            )
        df["ca_ex_pm_output"] = primary_model.process_single(df["instructions_ex_ca"])

        ranking_model = GPTPMPairwiseRankingModel(use_cache=use_cache)
        opponents = [f"ca_{i}_pm_output" for i in range(n_clarifying_questions)] + [
            # "full_pm_output",
            # "summ_pm_output",
        ]
        for opponent in opponents:
            assert opponent[-10:] == "_pm_output"
            opp = opponent[:-10]

            df[f"pref_{opp}"] = df.progress_apply(
                lambda x: ranking_model.forward(
                    x["prompt"], x["doc_full"], x["ca_ex_pm_output"], x[opponent]
                ),
                axis=1,
            ).apply(lambda x: "ex" if x == 0 else "bm")

        # calculate total wins
        wins = 0
        for opponent in [f"ca_{i}_pm_output" for i in range(n_clarifying_questions)]:
            opp = opponent[:-10]
            wins += (df[f"pref_{opp}"] == "ex").sum()
        win_rate_bm = wins / (len(df) * n_clarifying_questions)
        # win_rate_bm = (df["pref_01"] == "ex").sum() / len(df)
        print(win_rate_bm)
        # dump preferences to a json
        df_to_md(df.iloc[:1], "tmp.md")
        df.to_json(
            f"results/intermediate/pm-{pm_name}_or-{oracle_name}_{str(ds_downsample)}.json"
        )


if __name__ == "__main__":
    main()
