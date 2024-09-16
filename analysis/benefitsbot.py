
import json
import argparse
import pandas as pd
from models.utils import load_lm, LanguageModelWrapper
from models.cq_models import BaseClarifyingQuestionModel
from models.oracle_models import BaseOracleModel
from typing import List
from models.utils import ModelFamily
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
from datamodels.userprofile import UserProfile
from datamodels.chatbot import ChatBot
from datamodels.syntheticuser import SyntheticUser

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
    "--max_chat_iterations",
    default=10,
    help="Maximum number of iterations between benefits bot and synthetic user",
    type=int,
)
parser.add_argument(
    "--chat_history",
    default="./dataset/benefits_short_v0.0.1_manual.txt",
    help="Path to the chat history or benefits description",
)
parser.add_argument(
    "--dataset_path",
    default="./dataset/benefits_dataset_v0.1.0.jsonl",
    help="Path to the chat history or benefits description",
)
args = parser.parse_args()

# Read the chat history from the file
def read_chat_history(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}")
        return None

# Description about all the benefits eligbility in natural language
chat_history = read_chat_history(args.chat_history)

# Load the dataset
df = pd.read_json(args.dataset_path, lines=True)

for index, row in df.iterrows():
    print(f"Index: {index}")
    programs = row['programs']
    labels = row['labels']
    hh_nl_desc = row['hh_nl_desc']

    user = UserProfile()
    no_of_benefits = len(programs)
    # print(f"Total number of programs: {no_of_benefits}")
    # print(f"Household Description: {hh_nl_desc}")

    # Load language models and pipeline setup
    chatbot_model_wrapper = load_lm(args.chatbot_model_name)
    chatbot = ChatBot(chatbot_model_wrapper, no_of_benefits, chat_history)

    synthetic_user_model_wrapper = load_lm(args.synthetic_user_model_name)
    synthetic_user = SyntheticUser(user, hh_nl_desc, synthetic_user_model_wrapper)

    cur_iter_count = 0
    max_chat_iterations = args.max_chat_iterations

    while cur_iter_count<max_chat_iterations and chatbot.benefits_ready() != True:
        cur_iter_count += 1
        print(f"Iteration Count: {cur_iter_count}")
        cq = chatbot.ask_cq()
        print(f"Clarifying Question: {cq}")
        cq_answer = synthetic_user.answer_cq(cq)
        print(f"Answer: {cq_answer}")
        print("=="*20)
        chatbot.append_chat_history_with_cq_answer(cq_answer)

    benefits_prediction = chatbot.predict_benefits_eligibility()

    print(f"Benefits Prediction: {benefits_prediction}")

    print("=="*30)