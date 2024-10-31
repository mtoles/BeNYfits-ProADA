import os
import argparse

import pandas as pd
from models.model_utils import load_lm
from datamodels.userprofile import UserProfile
from datamodels.chatbot import *
from datamodels.syntheticuser import SyntheticUser
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime
from tqdm import tqdm
from acc_over_time_experiment import plot_metrics_per_turn
from models.lm_logging import LmLogger
from dataset_procedural import show_household
from users import Household

parser = argparse.ArgumentParser(description="Build benefits bot")
parser.add_argument(
    "--chatbot_model_name",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Name of the benefits bot model to use.",
)
parser.add_argument(
    "--chatbot_strategy",
    default="backbone",
    help="Strategy to use for the benefits bot.",
)
parser.add_argument(
    "--synthetic_user_model_name",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Name of the synthetic user model to use.",
)
parser.add_argument(
    "--max_dialog_turns",
    default=10,
    help="Maximum number of iterations between benefits bot and synthetic user",
    type=int,
)
parser.add_argument(
    "--eligibility_requirements",
    default="./dataset/benefits_short.jsonl",
    help="Path to the chat history or benefits description",
)
parser.add_argument(
    "--dataset_path",
    default="dataset/procedural_hh_dataset_1.0.1_annotated_50.jsonl",
    help="Path to the chat history or benefits description",
)
parser.add_argument(
    "--downsample_size",
    default=None,
    type=int,
    help="Downsample the dataset to this size",
)
parser.add_argument(
    "--top_k",
    default=None,
    type=int,
    help="Number of similar sentences to pick from natural language profile to match with question in synthetic user",
)
parser.add_argument(
    "--predict_every_turn",
    default=False,
    type=bool,
    help="Predict eligibility after every dialog turn",
)
parser.add_argument(
    "--programs",
    default=None,
    # type=str,
    help="Number of programs in the dataset",
    nargs="+",
)
parser.add_argument(
    "--num_programs",
    default=None,
    type=int,
    help="Downsample to the first n programs",
)
parser.add_argument(
    "--estring",
    default="tmp",
    type=str,
    help="Experiment tracking string",
)
args = parser.parse_args()

now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
programs_abbreviation = "_".join(
    ["".join([char for char in program if char.isupper()]) for program in args.programs]
)

output_dir = f"./results/{args.estring}/{now}_{programs_abbreviation}"


# Read the chat history from the file
def read_eligibility_requirements(file_path, num_programs):
    with open(file_path, "r") as file:
        file_content = file.read()
        # strip lines at the top starting with #
        file_content = "\n".join(
            [line for line in file_content.split("\n") if not line.startswith("#")]
        )
        assert len(file_content) > 0, "File is empty"
        eligibility_df = pd.read_json(file_content, lines=True)
        if num_programs is not None:
            eligibility_df = eligibility_df.iloc[:num_programs]
        return eligibility_df


def eligibility_to_string(eligibility_df):
    eligibility_def = ""
    for index, row in eligibility_df.iterrows():
        eligibility_def += f"**{index}. {row['program']}**\n"
        eligibility_def += f"{row['description']}\n\n"
    return eligibility_def


# Description about all the benefits eligbility in natural language
eligibility_df = read_eligibility_requirements(
    args.eligibility_requirements, args.num_programs
)
if args.programs is not None:
    eligibility_df = eligibility_df[
        eligibility_df["program"].apply(lambda x: x in args.programs)
    ].reset_index(drop=True)
eligibility_requirements = eligibility_to_string(eligibility_df)


# Load the dataset
df = pd.read_json(args.dataset_path, lines=True)
df["hh"] = df["hh"].apply(lambda hh: Household.from_dict(hh))
if args.downsample_size:
    df = df[: args.downsample_size]

# if args.num_programs is not None:
# df["programs"] = df["programs"].apply(lambda x: x[: args.num_programs])
# df["labels"] = df["labels"].apply(lambda x: x[: args.num_programs])

predictions = []
histories = []
per_turn_all_predictions = []
last_turn_iteration = []

user = UserProfile()
num_benefits = len(args.programs)
# print(f"Total number of programs: {no_of_benefits}")
# print(f"Household Description: {hh_nl_desc}")

# Load language models and pipeline setup
chatbot_model_wrapper = load_lm(args.chatbot_model_name)


lm_logger = LmLogger(log_dir=output_dir)


def get_model(model_name: str, chatbot_model_wrapper=chatbot_model_wrapper) -> ChatBot:
    if model_name == "backbone":
        chatbot = ChatBot(
            chatbot_model_wrapper, num_benefits, eligibility_requirements, lm_logger
        )
    elif model_name == "backbone_fixed":
        chatbot = ChatBotBackboneFixed(
            chatbot_model_wrapper, num_benefits, eligibility_requirements, lm_logger
        )
    elif model_name == "prompt_engineering_loose":
        chatbot = ChatBotPredictCQPromptLoose(
            chatbot_model_wrapper, num_benefits, eligibility_requirements, lm_logger
        )
    elif model_name == "notetaker":
        chatbot = NotetakerChatBot(
            chatbot_model_wrapper,
            num_benefits,
            eligibility_requirements,
            notebook_only=False,
            lm_logger=lm_logger,
        )
    elif model_name == "notetaker-2":
        chatbot = NotetakerChatBot(
            chatbot_model_wrapper,
            num_benefits,
            eligibility_requirements,
            notebook_only=True,
            lm_logger=lm_logger,
        )
    elif model_name == "cot":
        chatbot = CotChatBot(
            chatbot_model_wrapper,
            num_benefits,
            eligibility_requirements,
            lm_logger=lm_logger,
        )
    elif model_name == "coderef":
        chatbot = CodeRefChatBot(
            chatbot_model_wrapper,
            num_benefits,
            eligibility_requirements,
            lm_logger=lm_logger,
        )
    else:
        raise ValueError(f"Invalid chatbot strategy: {args.chatbot_strategy}")
    return chatbot


synthetic_user_model_wrapper = load_lm(args.synthetic_user_model_name)

for index, row in tqdm(df.iterrows()):
    # reinstantiate the model every time to dump the chat history
    labels = row[args.programs]
    lm_logger.add_empty_convo(labels.to_dict())
    chatbot = get_model(args.chatbot_strategy)
    # Temporarily load codellama if we are using llama
    codellama_mode = ("meta-llama/Meta-Llama-3-" in chatbot.lm_backbone.lm_wrapper.hf_name) and "code" in args.chatbot_strategy
    if codellama_mode:
        print("temporarily entering codellama mode")
        chatbot.lm_wrapper = LanguageModelWrapper("Codellama 7B Instruct", "llama", "codellama/CodeLlama-7b-Instruct-hf")
    chatbot.pre_conversation(eligibility_requirements=eligibility_requirements)
    if codellama_mode:
        # unload codellama
        print("exiting codellama mode")
        chatbot.lm_wrapper = LanguageModelWrapper("Llama 8B Instruct", "llama", "meta-llama/Meta-Llama-3-8B-Instruct")


    hh_nl_desc = row["hh_nl_desc"]
    synthetic_user = SyntheticUser(
        user,
        hh_nl_desc,
        synthetic_user_model_wrapper,
        lm_logger=lm_logger,
        top_k=args.top_k,
    )
    history = [
        {
            "role": "system",
            "content": f"You are a language model trying to help user to determine eligbility of user for benefits. Currently, you do not know anything about the user. Ask questions that will help you determine the eligibility of user for benefits as quickly as possible. Ask only one question at a time. The eligibility requirements are as follows:\n\n{eligibility_requirements}",
        },
        # {
        #     "role": "assistant",
        #     "content": f"Hello, I am BenefitsBot. I will be helping you determine your eligibility for benefits. Please answer the following questions to the best of your knowledge.",
        # },
    ]
    print(f"Index: {index}")

    cur_iter_count = 0
    per_turn_predictions = []
    decision = None
    while True:
        if args.predict_every_turn:
            if cur_iter_count != 0:
                per_turn_predictions.append(
                    chatbot.predict_benefits_eligibility(history, args.programs)
                )
            else:
                # default to all zero prediction on 0th round
                default_predictions = dict([(x, 0) for x in args.programs])
                per_turn_predictions.append(default_predictions)
        else:
            per_turn_predictions.append(None)
        ### break if out of dialog turns ###
        if cur_iter_count >= args.max_dialog_turns:
            print(f"Max dialog turns ({args.max_dialog_turns}) reached")
            print("==" * 20)
            last_turn_iteration.append(cur_iter_count)
            decision = per_turn_predictions[-1]
            print(f"Decision:  {decision}")
            print(f"label:     {labels.to_dict()}")
            print("==" * 20)
            break
        ### break if benefits eligibility is ready ###
        if (
            cur_iter_count > 0
            and str(chatbot.predict_benefits_ready(history)) == "True"
        ):
            print(
                f"Benefits eligibility decided on turn {cur_iter_count}/{args.max_dialog_turns}"
            )
            decision = per_turn_predictions[-1]
            print(f"Decision:  {decision}")
            print(f"label:     {labels.to_dict()}")
            print("==" * 20)
            # fill the remaining turns with None
            per_turn_predictions.extend(
                [decision] * (args.max_dialog_turns - cur_iter_count)
            )
            last_turn_iteration.append(cur_iter_count)
            break
        ### otherwise, ask a question ###
        cq = chatbot.predict_cq(history, cur_iter_count)
        history.append({"role": "assistant", "content": cq})
        cq_answer = synthetic_user.answer_cq(cq)
        history.append({"role": "user", "content": cq_answer})

        print(f"Turn Number:         {cur_iter_count}")
        print(f"Clarifying Question: {cq}")
        print(f"Answer:              {cq_answer}")
        chatbot.post_answer(history)  # optional
        print("==" * 20)
        cur_iter_count += 1
        # chatbot.append_chat_question_and_answer(cq, cq_answer)
    per_turn_all_predictions.append(per_turn_predictions)
    lm_logger.log_predictions(per_turn_predictions)
    lm_logger.log_hh_diff(row["hh"])
    history.append({"role": "assistant", "content": per_turn_all_predictions[-1]})
    histories.append(history)


lm_logger.save()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if args.predict_every_turn:
    # call one final time
    plot_metrics_per_turn(
        per_turn_all_predictions,
        df[args.programs],
        last_turn_iteration,
        output_dir=output_dir,
        experiment_params={
            "Backbone Model": args.chatbot_model_name,
            "Strategy": f"{args.estring} {args.chatbot_strategy}",
            "Programs": ", ".join(args.programs),
            "Max Dialog Turns": args.max_dialog_turns,
            "Downsample Size": args.downsample_size,
            "Top K Sentences": args.top_k,
        },
    )

with open(f"{output_dir}/transcript.md", "w") as f:
    for i, transcript in enumerate(histories):
        f.write(f"Transcript {i}\n")
        f.write(f"{transcript}\n")
        f.write("\n\n==========\n\n")

pass
