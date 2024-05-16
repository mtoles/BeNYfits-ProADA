# Setup
import pandas as pd
from models.summarization_models import GPTSummarizer
from models.prompt_generator_models import GPTPromptGenerator
from models.primary_models import GPTPrimaryModel, Llama2PrimaryModel, PrimaryModel
from models.cq_models import GPTClarifyingQuestionModel
from models.oracle_models import GPTOracleAbstractiveModel
from models.ranking_models import (
    GPTClarifyingAnswersRankingModel,
    GPTPrimaryModelOutputRankingModel,
)
from tqdm import tqdm
import click
import numpy as np
import random


@click.command()
@click.option("--pm_name", default="llama2", help="Name of the primary model to use. If using a gpt model, use the exact api call name, e.g., 'gpt-3.5-turbo'")
@click.option(
    "--pm_size",
    default="7b",
    help="Size of the primary model to use, one of {7b, 13b, 70b}",
)
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
    pm_name,
    pm_size,
    pm_batch_size,
    prompt_gen_temperature,
    ds_path,
    ds_downsample,
    n_clarifying_questions,
    ds_shift,
    use_cache,
    intermediate_results_path,
):
    assert pm_name in ["gpt4", "llama2", "gpt-3.5-turbo"]
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
            df["doc_orig"] = df["selftext"].astype(str)
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
            df["doc_summ"] = df["doc_orig"].progress_apply(
                lambda x: summarizer.forward(x)
            )
        else:
            print("Skipping summarization because doc_summ is already present")

        # Generate primary tasks
        prompt_generator = GPTPromptGenerator(use_cache)
        print("generating prompts...")
        df["prompt"] = df["doc_orig"].progress_apply(
            lambda x: prompt_generator.forward(x, prompt_gen_temperature)
        )

        # Load the primary model
        if "gpt" in pm_name.lower():
            primary_model = GPTPrimaryModel(pm_name, use_cache)
        elif pm_name == "llama2":
            primary_model = Llama2PrimaryModel(pm_size, pm_batch_size)
        else:
            raise ValueError(f"Unknown primary model name {pm_name}")
        # Prepare instructions for the full example
        df["pm_instruction_full"] = df.apply(
            lambda x: primary_model.prepare_instruction(x["doc_orig"], x["prompt"]),
            axis=1,
        )
        # Prepare instructions for the summary example
        df["pm_instruction_summ"] = df.apply(
            lambda x: primary_model.prepare_instruction(x["doc_summ"], x["prompt"]),
            axis=1,
        )
        # Run the cq model
        cq_model = GPTClarifyingQuestionModel(use_cache)
        print("running cq model...")
        df["cq"] = df.progress_apply(
            lambda x: cq_model.forward(
                x["doc_summ"], x["prompt"], n_clarifying_questions
            ),
            axis=1,
        )

        oracle_model = GPTOracleAbstractiveModel(use_cache=use_cache)
        print("running abstractive oracle model to answer clarifying questions...")
        # Ask the clarifying question to the oracle
        df["ca"] = df.progress_apply(
            lambda x: oracle_model.forward(x["doc_orig"], x["cq"]), axis=1
        )

        clarifying_answers_ranking_model = GPTClarifyingAnswersRankingModel(
            use_cache=use_cache
        )
        primary_model_output_ranking_model = GPTPrimaryModelOutputRankingModel(
            use_cache=use_cache
        )

        print("running ranking model to order clarifying questions...")
        # Ask the clarifying question to the oracle
        df["ordered_cq_on_ca"] = df.progress_apply(
            lambda x: clarifying_answers_ranking_model.forward(
                x["doc_summ"], x["prompt"], x["cq"], x["ca"]
            ),
            axis=1,
        )

        print(
            "preparing instructions for joint summ + ca contexts to be fed to the primary model"
        )

        df["joint_summ_ca_instructions"] = df.progress_apply(
            lambda x: prepare_ca_instructions(
                x["doc_summ"], x["ca"], x["prompt"], primary_model
            ),
            axis=1,
        )

        # print("preparing instructions for joint summ + ca contexts")

        # df["joint_summ_ca_instructions"] = df.progress_apply(
        #     lambda x: prepare_ca_instructions(
        #         x["doc_summ"], x["ca"], x["prompt"], primary_model
        #     ),
        #     axis=1,
        # )

        print("running primary models for for joint summ + ca contexts")

        # TODO: Modify this to make parallel calls for Llama
        # joint_summ_ca_pm_outputs = df.progress_apply(
        #     lambda x: primary_model.process(pd.Series(x["joint_summ_ca_instructions"])),
        #     axis=1,
        # )
        df["joint_summ_ca_pm_outputs"] = primary_model.process_list(
            df["joint_summ_ca_instructions"]
        )

        # Save the primary model's output for summary
        df["summ_pm_output"] = primary_model.process(df["pm_instruction_summ"])

        # Save the primary model's output for original context
        df["full_pm_output"] = primary_model.process(df["pm_instruction_full"])

        df["ordered_cq_on_pm_outputs"] = df.progress_apply(
            lambda x: primary_model_output_ranking_model.forward(
                x["doc_summ"],
                x["prompt"],
                x["cq"],
                x["joint_summ_ca_pm_outputs"],
                x["full_pm_output"],
            ),
            axis=1,
        )

        def get_preference(row):
            if random.uniform(a=0, b=1) < 0.5:
                model_preference = "First"
            else:
                model_preference = "Second"

            return model_preference

        df["model_preference"] = df.progress_apply(get_preference, axis=1)

        def adjust_according_to_preference(row):
            if row["model_preference"] == "First":
                return (
                    "1. "
                    + row["ordered_cq_on_pm_outputs"][0]
                    + "\n\n2. "
                    + row["ordered_cq_on_pm_outputs"][-1]
                )
            else:
                return (
                    "1. "
                    + row["ordered_cq_on_pm_outputs"][-1]
                    + "\n\n2. "
                    + row["ordered_cq_on_pm_outputs"][0]
                )

        df["preference_eval_cq"] = df.progress_apply(
            adjust_according_to_preference, axis=1
        )

        # Save your results
        df.to_csv(f"results/ranked_dataset.csv", index=False)

        df.to_json(f"results/intermediate/{pm_name}-{pm_size}_{ds_downsample}.json")


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


if __name__ == "__main__":
    main()
