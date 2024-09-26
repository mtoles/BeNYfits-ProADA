import os
import argparse
import pandas as pd
from models.model_utils import load_lm
from datamodels.userprofile import UserProfile
from datamodels.chatbot import ChatBot
from datamodels.syntheticuser import SyntheticUser
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Build benefits bot")
parser.add_argument(
    "--chatbot_model_name",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Name of the benefits bot model to use.",
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
    default="./dataset/benefits_short.txt",
    help="Path to the chat history or benefits description",
)
parser.add_argument(
    "--dataset_path",
    default="dataset/procedural_hh_dataset_0.1.8_annotated_50.jsonl",
    help="Path to the chat history or benefits description",
)
parser.add_argument(
    "--downsample_size",
    default=None,
    type=int,
    help="Downsample the dataset to this size",
)
args = parser.parse_args()

now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


# Read the chat history from the file
def read_eligibility_requirements(file_path):
    try:
        with open(file_path, "r") as file:
            file_content = file.read()
            # strip lines at the top starting with #
            file_content = "\n".join(
                [line for line in file_content.split("\n") if not line.startswith("#")]
            )
            assert len(file_content) > 0, "File is empty"
            return file_content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}")
        return None


# Description about all the benefits eligbility in natural language
eligibility_requirements = read_eligibility_requirements(args.eligibility_requirements)

history = [
    {
        "role": "system",
        "content": f"You are a language model trying to help user to determine eligbility of user for benefits. Ask clarifying questions that will help you determine the eligibility of user for benefits as quickly as possible. The eligibility requirements are as follows:\n\n{eligibility_requirements}",
    }
]

# Load the dataset
df = pd.read_json(args.dataset_path, lines=True)
if args.downsample_size:
    df = df[: args.downsample_size]

predictions = []
histories = []
for index, row in tqdm(df.iterrows()):
    print(f"Index: {index}")
    programs = row["programs"]
    labels = row["labels"]
    hh_nl_desc = row["hh_nl_desc"]

    user = UserProfile()
    num_benefits = len(programs)
    # print(f"Total number of programs: {no_of_benefits}")
    # print(f"Household Description: {hh_nl_desc}")

    # Load language models and pipeline setup
    chatbot_model_wrapper = load_lm(args.chatbot_model_name)
    chatbot = ChatBot(chatbot_model_wrapper, num_benefits, eligibility_requirements)

    synthetic_user_model_wrapper = load_lm(args.synthetic_user_model_name)
    synthetic_user = SyntheticUser(user, hh_nl_desc, synthetic_user_model_wrapper)

    cur_iter_count = 0

    while (
        cur_iter_count < args.max_dialog_turns
        and chatbot.predict_benefits_ready(history) != True
    ):
        cur_iter_count += 1
        print(f"Iteration Count: {cur_iter_count}")
        cq = chatbot.predict_cq(history)
        history.append({"role": "assistant", "content": cq})
        print(f"Clarifying Question: {cq}")
        cq_answer = synthetic_user.answer_cq(cq)
        history.append({"role": "user", "content": cq_answer})
        print(f"Answer: {cq_answer}")
        print("==" * 20)
        # chatbot.append_chat_question_and_answer(cq, cq_answer)

    benefits_prediction_str = chatbot.predict_benefits_eligibility(history)
    benefits_prediction = chatbot.extract_prediction(
        benefits_prediction_str, num_benefits
    )
    predictions.append(benefits_prediction)
    print(f"Benefits Prediction: {benefits_prediction}")
    print("==" * 30)
    # transcript.append(benefits_prediction_str)
    # transcript.append(f"Predicted Benefits: {benefits_prediction}")
    # transcripts.append("\n\n".join(transcript))
    history.append({"role": "assistant", "content": benefits_prediction_str})
    histories.append(history)

df["predictions"] = predictions
df["correct"] = df.apply(lambda x: x["labels"] == x["predictions"], axis=1)
df["f1"] = None
non_null_predictions = ~df["predictions"].isnull()
if (
    non_null_predictions.sum() > 0
):  # pandas gets confused if there are no actual indices to set
    df.loc[non_null_predictions, "f1"] = df[non_null_predictions].apply(
        lambda x: f1_score(x["labels"], x["predictions"], average="weighted"), axis=1
    )
df.loc[df["predictions"].isnull(), "f1"] = (
    0  # Set F1 score to 0 if no prediction was made
)
print(f"Total F1 Score: {df['f1'].mean()}")

df_labels = pd.DataFrame(
    columns=df.programs[0], index=df.index, data=df.labels.tolist()
)
df_preds = pd.DataFrame(
    columns=df.programs[0], index=df.index, data=df.predictions.tolist()
)
df_acc = df_labels == df_preds

### Save results file and chat history ###
output_dir = f"./results/{now}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(f"{output_dir}/results_summary.md", "w") as f:
    f.write("### Args ###\n")
    for key, value in args.__dict__.items():
        f.write(f"{key}: {value}\n")
    f.write("### Results ###\n")
    f.write(f"Total F1 Score: {df['f1'].mean()}\n")
    for program in df.programs[0]:
        f.write(f"{program}\n")
        f.write(
            f"F1: {f1_score(df_labels[program], df_preds[program], average='weighted')}\n"
        )
        f.write(f"Accuracy: {df_acc[program].mean()}\n")
        f.write(
            f"Precision: {precision_score(df_labels[program], df_preds[program], average='weighted')}\n"
        )
        f.write(
            f"Recall: {recall_score(df_labels[program], df_preds[program], average='weighted')}\n"
        )

with open(f"{output_dir}/transcript.md", "w") as f:
    for i, transcript in enumerate(histories):
        f.write(f"Transcript {i}\n")
        f.write(f"{transcript}\n")
        f.write("\n\n==========\n\n")
df.to_json(f"{output_dir}/results.jsonl", lines=True, orient="records")
df_preds.to_json(f"{output_dir}/predictions.jsonl", lines=True, orient="records")
df_acc.to_json(f"{output_dir}/accuracy.jsonl", lines=True, orient="records")
pass
