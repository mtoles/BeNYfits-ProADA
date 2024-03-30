# Setup
from openai import OpenAI
import pandas as pd
import json
from models.summarization_models import GPTSummarizer
from models.prompt_generator_models import GPTPromptGenerator
from models.primary_models import GPTPrimaryModel, Llama2PrimaryModel
from models.reward_models import GPTRewardModel, run_alpaca_eval
from models.cq_models import GPTClarifyingQuestionModel
from models.oracle_models import GPTOracleModel, GPTOracleAbstractiveModel, Llama2OracleModel
from tqdm import tqdm
import click
import numpy as np
import os


@click.command()
@click.option("--pm_name", default="llama2", help="Name of the primary model to use")
@click.option(
    "--pm_size", help="Size of the primary model to use, one of {7b, 13b, 70b}"
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
@click.option("--use_cache", default=False, help="Use the GPT-4 shelved cache")
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
    assert pm_name in ["gpt4", "llama2"]
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

        print(len(df))
        # # Shift the dataset
        # df = pd.concat([df.iloc[ds_shift:], df.iloc[:ds_shift]]).reset_index(drop=True)
        # # Apply downsampling
        # if ds_downsample is not None:
        #     df = df.head(ds_downsample)
        # # summarize each item of the dataset to 50% of its original length
        # summarizer = GPTSummarizer(use_cache)
        # print("summarizing...")
        # if "doc_summ" not in df.columns:
        #     df["doc_summ"] = df["doc_orig"].progress_apply(
        #         lambda x: summarizer.forward(x)
        #     )
        # else:
        #     print("Skipping summarization because doc_summ is already present")

        # # Generate primary tasks
        # prompt_generator = GPTPromptGenerator(use_cache)
        # print("generating prompts...")
        # df["prompt"] = df["doc_orig"].progress_apply(
        #     lambda x: prompt_generator.forward(x, prompt_gen_temperature)
        # )

        # # Load the primary model
        # if pm_name == "gpt4":
        #     primary_model = GPTPrimaryModel(use_cache)
        # elif pm_name == "llama2":
        #     primary_model = Llama2PrimaryModel(pm_size, pm_batch_size)
        # else:
        #     raise ValueError(f"Unknown primary model name {pm_name}")
        # # Prepare instructions for the full example
        # df["pm_instruction_full"] = df.apply(
        #     lambda x: primary_model.prepare_instruction(x["doc_orig"], x["prompt"]),
        #     axis=1,
        # )
        # # Prepare instructions for the summary example
        # df["pm_instruction_summ"] = df.apply(
        #     lambda x: primary_model.prepare_instruction(x["doc_summ"], x["prompt"]),
        #     axis=1,
        # )
        # # Run the cq model
        # cq_model = GPTClarifyingQuestionModel(use_cache)
        # print("running cq model...")
        # df["cq"] = df.progress_apply(
        #     lambda x: cq_model.forward(
        #         x["doc_summ"], x["prompt"], n_clarifying_questions
        #     ),
        #     axis=1,
        # )

        df['cq'] = df['Clarifying Question']

        df.to_csv(f"results/clarifying_qa_middle.csv")

        print("Running Oracle Now")

        llama_oracle_model = Llama2OracleModel("7b")

        df["llama_answers"] = llama_oracle_model.forward(
            documents=df["doc_summ"].tolist(),
            questions=df['cq'].apply(lambda x: [x]).tolist()
        )

        oracle_model = GPTOracleAbstractiveModel(use_cache=use_cache)
        # # Ask the clarifying question to the oracle
        df["answers"] = df.progress_apply(
            lambda x: oracle_model.forward(x["doc_orig"], x["cq"]), axis=1
        )

        # Write the results to a csv
        df.to_csv(f"results/clarifying_qa.csv")


        # run primary models
        # print("running primary model (full)...")
        # df["pm_answer_full"] = primary_model.process(df["pm_instruction_full"])
        # print("running primary model (summ)...")
        # df["pm_answer_summ"] = primary_model.process(df["pm_instruction_summ"])
        # df.to_json(f"results/intermediate/{pm_name}-{pm_size}_{ds_downsample}.json")

    # Evaluation
    # alpaca_preferences = run_alpaca_eval(
    #     df["pm_answer_summ"],
    #     df["pm_answer_full"],
    #     instruction=df["pm_instruction_full"],
    # )
    # df["preference"] = alpaca_preferences.apply(
    #     lambda x: {0: "summ", 1: "full", "None": "None"}[x]
    # )

    # Compare the summary answers and full answers using the reward model
    # reward_model = GPTRewardModel(use_cache)
    # print("running reward model...")
    # df["selection"] = df.progress_apply(
    #     lambda x: reward_model.forward(
    #         x["doc_orig"],
    #         x["pm_answer_full"],
    #         x["pm_answer_summ"],
    #         x["prompt"][0],
    #         temperature=0.7,
    #     ),
    #     axis=1,
    # )
    # num_full_selected = len(df[df["selection"] == "full"])
    # percent_full_selected = num_full_selected / len(df)
    # print(
    #     f"Percent of full docs selected: {percent_full_selected} | {num_full_selected} / {len(df)}"
    # )
    # print()
    # results_path = f"results/{pm_name}-{pm_size}_{ds_downsample}.json"
    # if not os.path.exists("results"):
    #     os.makedirs("results")
    # df.to_json(results_path)
    # print


if __name__ == "__main__":
    main()