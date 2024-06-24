# Setup
import pandas as pd
from models.summarization_models import GPTSummarizer
from models.prompt_generator_models import GPTPromptGenerator
from models.primary_models import GPTPrimaryModel, Llama2PrimaryModel, Llama3PrimaryModel, PrimaryModel
from models.cq_models import GPTClarifyingQuestionModel
from models.oracle_models import GPTOracleAbstractiveModel, Llama3OracleModel
from models.ranking_models import (
    GPTClarifyingAnswersRankingModel,
    GPTPMOutputRankingModel,
)
from tqdm import tqdm
import click
import numpy as np
import random


@click.command()
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
@click.option(
    "--oracle_size",
    default="8b",
    help="Size of the oracle model to use"
)
@click.option("--oracle_batch_size", default=4, help="Batch size for the oracle model")
@click.option(
    "--pm_size",
    default="7b",
    help="Size of the primary model to use, one of {7b, 13b, 70b}",  # todo: update for llama3
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
    oracle_name,
    oracle_size,
    oracle_batch_size,
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
    assert pm_name in ["gpt4", "llama2", "llama3", "gpt-3.5-turbo", "gpt-4-turbo"]
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
        elif pm_name == "llama2":
            primary_model = Llama2PrimaryModel(pm_size, pm_batch_size)
        elif pm_name == "llama3":
            primary_model = Llama3PrimaryModel(pm_size, pm_batch_size)
        else:
            raise ValueError(f"Unknown primary model name {pm_name}")
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
        cq_model = GPTClarifyingQuestionModel(use_cache)
        print("running cq model...")
        df["cq"] = df.progress_apply(
            lambda x: cq_model.forward(
                x["doc_summ"], x["prompt"], n_clarifying_questions
            ),
            axis=1,
        )
        if "gpt" in oracle_name.lower():
            oracle_model = GPTOracleAbstractiveModel(
                model_name=oracle_name, use_cache=use_cache
            )
        elif "llama3" in oracle_name.lower():
            oracle_model = Llama3OracleModel(
                model_size=oracle_size, batch_size=oracle_batch_size
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
        df["summ_ca_pm_outputs"] = primary_model.process_list(
            df["instructions_summ_ca"]
        )

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

        pm_output_ranking_model = GPTPMOutputRankingModel(use_cache=use_cache)

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
            f"results/intermediate/pm-{pm_name}_or-{oracle_name}_{str(ds_downsample)}.json"
        )

        # calculate percent time option 1 is best
        # option_0_mean_position = df["ranking"].apply(lambda x: x.index(0)).mean()
        # option_4_mean_position = df["ranking"].apply(lambda x: x.index(4)).mean()
        # print(f"Option 0 (summ) mean position: {option_0_mean_position}")
        # print(f"Option 4 (full) mean position: {option_4_mean_position}")
        # print

        # def get_preference(row):
        #     if random.uniform(a=0, b=1) < 0.5:
        #         model_preference = "First"
        #     else:
        #         model_preference = "Second"

        #     return model_preference

        # df["model_preference"] = df.progress_apply(get_preference, axis=1)

        # def adjust_according_to_preference(row):
        #     if row["model_preference"] == "First":
        #         return (
        #             "1. "
        #             + row["ordered_cq_on_pm_outputs"][0]
        #             + "\n\n2. "
        #             + row["ordered_cq_on_pm_outputs"][-1]
        #         )
        #     else:
        #         return (
        #             "1. "
        #             + row["ordered_cq_on_pm_outputs"][-1]
        #             + "\n\n2. "
        #             + row["ordered_cq_on_pm_outputs"][0]
        #         )

        # df["preference_eval_cq"] = df.progress_apply(
        #     adjust_according_to_preference, axis=1
        # )

        # # Save your results
        # df.to_csv(f"results/ranked_dataset.csv", index=False)

        # df.to_json(f"results/intermediate/{pm_name}-{pm_size}_{ds_downsample}.json")


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
