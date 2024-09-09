
import argparse
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
    default=2,
    help="Maximum number of iterations between benefits bot and synthetic user",
    type=int,
)
args = parser.parse_args()

# TODO - Plug history and user profile in from arguments
user = UserProfile()

# Description about all the benefits eligbility in natural language
chat_history = "Blah blah blah blah"
no_of_benefits = 5

chatbot_model_wrapper = load_lm(args.chatbot_model_name)
chatbot = ChatBot(chatbot_model_wrapper, no_of_benefits, chat_history)

synthetic_user_model_wrapper = load_lm(args.synthetic_user_model_name)
synthetic_user = SyntheticUser(user, synthetic_user_model_wrapper)

cur_iter_count = 0
max_chat_iterations = args.max_chat_iterations

while cur_iter_count<max_chat_iterations and chatbot.benefits_ready() != True:
    cur_iter_count += 1
    cq = chatbot.ask_cq()
    cq_answer = synthetic_user.answer_cq(cq)
    chatbot.append_chat_history_with_cq_answer(cq_answer)

benefits_prediction = chatbot.predict_benefits_eligibility()

print(f"Benefits Prediction: {benefits_prediction}")