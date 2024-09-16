
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
    default=10,
    help="Maximum number of iterations between benefits bot and synthetic user",
    type=int,
)
args = parser.parse_args()

# TODO - Plug history and user profile in from arguments
user = UserProfile()

# Description about all the benefits eligbility in natural language
chat_history = """
    To be eligible for the Child and Dependent Care Tax Credit, you should be able to answer yes to these questions:

    1. Did you pay someone to care for your dependent so that you (and your spouse, if filing a joint return) could work or look for work? Qualifying dependents are:
        - a child under age 13 at the time of care;
        - a spouse or adult dependent who cannot physically or mentally care for themselves.
    2. Did the dependent live with you for more than half of 2023?
    3. Did you (and your spouse if you file taxes jointly) earn income? These can be from wages, salaries, tips, other taxable employee money, or earnings from self-employment.
    4. If you are married, do both you and your spouse work outside of the home?
        - Or, do one of you work outside of the home while the other is a full-time student, has a disability, or is looking for work?
    """
no_of_benefits = 1

chatbot_model_wrapper = load_lm(args.chatbot_model_name)
chatbot = ChatBot(chatbot_model_wrapper, no_of_benefits, chat_history)

synthetic_user_model_wrapper = load_lm(args.synthetic_user_model_name)
synthetic_user = SyntheticUser(user, synthetic_user_model_wrapper)

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