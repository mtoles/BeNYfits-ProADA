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
    default="dataset/procedural_hh_dataset_1.0.0_annotated_50.jsonl",
    help="Path to the chat history or benefits description",
)
parser.add_argument(
    "--downsample_size",
    default=None,
    type=int,
    help="Downsample the dataset to this size",
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
if args.chatbot_strategy == "backbone":
    chatbot = ChatBot(chatbot_model_wrapper, num_benefits, eligibility_requirements)
elif args.chatbot_strategy == "notetaker":
    chatbot = NotetakerChatBot(chatbot_model_wrapper, num_benefits, eligibility_requirements)
else:
    raise ValueError(f"Invalid chatbot strategy: {args.chatbot_strategy}")

synthetic_user_model_wrapper = load_lm(args.synthetic_user_model_name)

for index, row in tqdm(df.iterrows()):
    # reinstantiate the model every time
    chatbot = chatbot.__class__(chatbot_model_wrapper, num_benefits, eligibility_requirements)
    labels = row[args.programs]
    hh_nl_desc = row["hh_nl_desc"]
    synthetic_user = SyntheticUser(user, hh_nl_desc, synthetic_user_model_wrapper)
    history = [
        {
            "role": "system",
            "content": f"You are a language model trying to help user to determine eligbility of user for benefits. Currently, you do not know anything about the user. Ask questions that will help you determine the eligibility of user for benefits as quickly as possible. Ask only one question at a time. The eligibility requirements are as follows:\n\n{eligibility_requirements}",
        },
        {
            "role": "assistant",
            "content": f"Hello, I am BenefitsBot. I will be helping you determine your eligibility for benefits. Please answer the following questions to the best of your knowledge.",
        },
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
        if chatbot.predict_benefits_ready(history) == "True":
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
        cq = chatbot.predict_cq(history)
        history.append({"role": "assistant", "content": cq})
        cq_answer = synthetic_user.answer_cq(cq)
        history.append({"role": "user", "content": cq_answer})

        print(f"Turn Number:         {cur_iter_count}")
        print(f"Clarifying Question: {cq}")
        print(f"Answer:              {cq_answer}")
        chatbot.post_answer(history) # optional
        print("==" * 20)
        cur_iter_count += 1
        # chatbot.append_chat_question_and_answer(cq, cq_answer)
    per_turn_all_predictions.append(per_turn_predictions)
    # benefits_prediction = chatbot.predict_benefits_eligibility(history)

    # predictions.append(benefits_prediction)
    # per_turn_all_predictions[index].append(benefits_prediction)
    # print(f"Benefits Prediction: {benefits_prediction}")
    # print("==" * 20)
    history.append({"role": "assistant", "content": per_turn_all_predictions[-1]})
    histories.append(history)

# df["predictions"] = predictions
# df["correct"] = df.apply(lambda x: x["labels"] == x["predictions"], axis=1)
# df["f1"] = None
# non_null_predictions = ~df["predictions"].isnull()
# if (
#     non_null_predictions.sum() > 0
# ):  # pandas gets confused if there are no actual indices to set
#     df.loc[non_null_predictions, "f1"] = df[non_null_predictions].apply(
#         lambda x: f1_score(x["labels"], x["predictions"], average="weighted"), axis=1
#     )
# df.loc[df["predictions"].isnull(), "f1"] = (
#     0  # Set F1 score to 0 if no prediction was made
# )
# print(f"Total F1 Score: {df['f1'].mean()}")

# df_labels = pd.DataFrame(
#     columns=df.programs[0], index=df.index, data=df.labels.tolist()
# )
# df_preds = pd.DataFrame(
#     columns=df.programs[0], index=df.index, data=df.predictions.tolist()
# )
# df_acc = df_labels == df_preds

### Save results file and chat history ###
output_dir = f"./results/{args.estring}/{now}"
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
            "Programs": ", ".join(args.programs),
            "Max Dialog Turns": args.max_dialog_turns,
            "Downsample Size": args.downsample_size,
        },
    )
# with open(f"{output_dir}/results_summary.md", "w") as f:
#     f.write("### Args ###\n")
#     for key, value in args.__dict__.items():
#         f.write(f"{key}: {value}\n")
#     f.write("### Results ###\n")
#     f.write(f"Total F1 Score: {df['f1'].mean()}\n")
#     for program in df.programs[0]:
#         f.write(f"{program}\n")
#         f.write(
#             f"F1: {f1_score(df_labels[program], df_preds[program], average='weighted')}\n"
#         )
#         f.write(f"Accuracy: {df_acc[program].mean()}\n")
#         f.write(
#             f"Precision: {precision_score(df_labels[program], df_preds[program], average='weighted')}\n"
#         )
#         f.write(
#             f"Recall: {recall_score(df_labels[program], df_preds[program], average='weighted')}\n"
#         )

with open(f"{output_dir}/transcript.md", "w") as f:
    for i, transcript in enumerate(histories):
        f.write(f"Transcript {i}\n")
        f.write(f"{transcript}\n")
        f.write("\n\n==========\n\n")
df.to_json(f"{output_dir}/results.jsonl", lines=True, orient="records")
# df_preds.to_json(f"{output_dir}/predictions.jsonl", lines=True, orient="records")
# df_acc.to_json(f"{output_dir}/accuracy.jsonl", lines=True, orient="records")
pass
