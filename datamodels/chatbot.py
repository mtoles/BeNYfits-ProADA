from models.lm_backbone import LmBackboneModel
from typing import List, Optional
from copy import deepcopy
import re
import ast
import numpy as np
from models.lm_logging import LmLogger
from inspect import currentframe
from server.model_client import ModelAPIClient
from dotenv import load_dotenv
import os
from utils import rename_roles, RoleEnum

load_dotenv(override=False)
np.random.seed(42)

benefits_ready_prompt = {
    "role": "system",
    "content": "Is the information sufficient to determine eligibility of all programs? Answer only in one word True or False.",
}


def example_array(n):
    return str([bool(x % 2) for x in range(n)])


### Backbone Prompts ###
predict_cq_prompt_loose = "Ask a clarifying question that will help you determine the eligibility of user for benefits for benefits asking about one requirement at a time. Start from first program and move to last program and ask about all requirement of one benefit before moving to the next. Only ask about one fact at a time."


def get_last_bool_in_str(s: str) -> str:
    """Get the last True or False mentioned in a string. Additionally, return True if yes or Yes are in the input, and false if no or No are in the input"""
    pattern = r"true|false|yes|no"
    match = re.findall(pattern, s.lower())
    if not match:
        return "False"
    found = match[-1]
    if match:
        if found in ["yes", "true"]:
            return "True"
        elif found in ["no", "false"]:
            return "False"
    else:
        return "False"


class ChatBot:
    """ "Base class for chatbots. Serves as the simple backbone model."""

    def __init__(
        self,
        chat_model_id: str,
        no_of_programs: str,
        eligibility_requirements: str,
        use_cache: bool,
        random_seed: int,
        lm_logger: Optional[LmLogger] = None,
        code_model_id: Optional[str] = None,
    ):
        """
        ChatBot class for keeping the history of user chat and other functions to determine eligbility for benefits
        """
        self.chat_model_id = chat_model_id
        # self.benefits_ready_prompt = {
        #     "role": "user",
        #     "content": "Is the information sufficient to determine eligibility of all programs? Answer only in one word True or False.",
        # }
        self.benefits_ready_prompt = "Eligibility requirements: {eligibility_requirements}. \n\nIs the information sufficient to determine whether any member of the user's household is eligible for all programs? Answer only in one word True or False."
        self.benefits_prediction_prompt = "Eligibility: {eligibility_requirements}. \n\nPredict the programs for which any member of the user's household is eligible. Return only a boolean array of length {num_programs}, e.g. {example_array}, where the value at index `i` is true iff the user is eligible for program `i`. Only return the array. Do not return anything else in the response. If a user's eligibility is unclear, make your best guess."
        self.predict_cq_prompt = "Eligibility: {eligibility_requirements}. \n\nAsk a clarifying question that will help you determine if any member of the user's household is eligible for benefits as efficiently as possible. Only ask about one fact at a time."
        self.use_cache = use_cache
        self.random_seed = random_seed
        self.lm_api = ModelAPIClient(
            f"http://localhost:{os.getenv('LM_PORT_NO')}",
            random_seed=self.random_seed,
            lm_logger=lm_logger,
        )
        self.num_programs = no_of_programs
        self.eligibility_requirements = eligibility_requirements

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        history_ = rename_roles(history)
        prompt = {
            "role": "user",
            "content": self.benefits_ready_prompt.format(
                eligibility_requirements=self.eligibility_requirements
            ),
        }
        raw_lm_output = self.lm_api.forward(
            history_ + [prompt],
            chat_model_id=self.chat_model_id,
            use_cache=self.use_cache,
            logging_role="predict_benefits_ready",
        )
        lm_output = get_last_bool_in_str(str(raw_lm_output))
        return lm_output

    def predict_benefits_eligibility(self, history, programs) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """
        prompt = {
            "role": "user",
            "content": self.benefits_prediction_prompt.format(
                eligibility_requirements=self.eligibility_requirements,
                num_programs=self.num_programs,
                example_array=example_array(self.num_programs),
            ),
        }
        history_ = rename_roles(history)
        lm_output = self.lm_api.forward(
            history_ + [prompt],
            logging_role="predict_benefits_eligibility",
            chat_model_id=self.chat_model_id,
            use_cache=self.use_cache,
            constraint_type="regex",
            constraints=rf"(True|False)(,(True|False)){{{len(programs)-1}}}",
        )
        # TODO - Ensure output is a list of boolean
        # lm_output = self.extract_prediction(lm_output, programs)
        processed_output = ast.literal_eval(f"[{lm_output}]")
        assert len(processed_output) == len(programs)
        output_dict = dict(zip(programs, processed_output))
        return output_dict

    def predict_cq(self, history, chat_model_id) -> str:
        """
        Function to generate clarifying question.
        """
        # _prompt = history + [
        #     {
        #         "role": RoleEnum.user.value,
        #         "content": self.predict_cq_prompt,
        #     }
        # ]

        prompt = history + [
            {
                "role": RoleEnum.CQ_MODEL.value,
                "content": self.predict_cq_prompt.format(
                    eligibility_requirements=self.eligibility_requirements
                ),
            }
        ]
        prompt = rename_roles(prompt)
        cq = self.lm_api.forward(
            prompt,
            chat_model_id=chat_model_id,
            use_cache=self.use_cache,
            logging_role="predict_cq",
        )
        return cq

    def post_answer(self, history):
        """
        Function called after an answer is provided to the chatbot.
        """
        pass

    def extract_prediction(self, prediction: str, programs: list[str]) -> dict:
        """
        Extract the prediction from the model output
        """
        # Regex to match a list-like structure in the string
        pattern = r"\[.*?\]"
        # Find the last list-like match in the string
        matches = re.findall(pattern, prediction)
        match = matches[-1] if matches else None

        # TODO: constraint decoding
        if match:
            # Extract the matched portion and safely evaluate it
            extracted_list_str = match
            # Try getting a "pass"/"fail" list
            try:
                # Safely evaluate the string into a Python list
                bool_output = ast.literal_eval(extracted_list_str)
                assert isinstance(bool_output, list)
                assert len(bool_output) == len(programs)
                str_output = ["pass" if x else "fail" for x in bool_output]
                output = str_output
            except (SyntaxError, NameError, ValueError, AssertionError):
                # If the string can't be evaluated as a list, try parsing it as a list of bools
                try:
                    bool_output = [bool(int(x)) for x in extracted_list_str.split(",")]
                    assert len(bool_output) == len(programs)
                    str_output = ["pass" if x else "fail" for x in bool_output]
                    output = str_output
                except (SyntaxError, NameError, ValueError, AssertionError):
                    output = None
        else:
            output = None
        if output is None:
            # default to all fails
            print(
                "*** WARNING: Could not extract prediction from model output. Defaulting to all Nones. ***"
            )
            output = [None] * len(programs)
        output = [1 if x == "pass" else 0 for x in output]
        # convert to dict
        output = {list(programs)[i]: output[i] for i in range(len(programs))}
        return output

    def pre_conversation(self, eligibility_requirements: str = None):
        """Function called before a new conversation is started"""
        pass


class CotChatBot(ChatBot):
    """Class for chatbot that uses chain-of-thought"""

    def __init__(
        self,
        chat_model_id: str,
        no_of_programs: str,
        eligibility_requirements,
        use_cache: bool,
        random_seed: int,
        lm_logger: Optional[LmLogger] = None,
        code_model_id: Optional[str] = None,
    ):
        super().__init__(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_requirements,
            use_cache=use_cache,
            lm_logger=lm_logger,
            random_seed=random_seed,
        )
        # self.benefits_ready_prompt = {
        #     "role": "user",
        #     "content": "Is the information sufficient to determine eligibility of all programs? Think through your reasoning out loud. Then answer with True or False.",
        # }
        self.benefits_ready_prompt = "Eligibility requirements: {eligibility_requirements}. \n\nIs the information sufficient to determine whether any member of the user's household is eligible for all programs? Think through your reasoning out loud. Then answer with True or False."
        self.predict_benefits_reasoning_prompt = "Eligibility: {eligibility_requirements}. \n\nPredict the programs for which any member of the user's household is eligible. Return only a boolean array of length {num_programs}, e.g. {example_array}, where the value at index `i` is true iff the user is eligible for program `i`. Only return the array. Do not return anything else in the response. If a user's eligibility is unclear, make your best guess.Think through your reasoning out loud."
        self.predict_benefits_constrained_prompt = "Reasoning: {reasoning}. \n\nUsing the reasoning above, predict the programs for which any member of the user's household is eligible. Output a boolean array of length {num_programs}, e.g. {example_array}, where the value at index `i` is true iff the user is eligible for program `i`. If a user's eligibility is unclear, make your best guess."
        self.predict_cq_prompt = "Eligibility: {eligibility_requirements}. \n\nAsk a clarifying question that will help you determine if any member of the user's household is eligible for benefits as efficiently as possible. Only ask about one fact at a time. Think through your reasoning out loud, then state your question after a colon, e.g. Question: What is the user's age?"

    def predict_cq(self, history, chat_model_id) -> str:
        cq = super().predict_cq(history, chat_model_id)
        parts = cq.split(":")
        return parts[-1].strip()

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        prompt = {
            "role": "user",
            "content": self.benefits_ready_prompt.format(
                eligibility_requirements=self.eligibility_requirements
            ),
        }
        raw_lm_output = self.lm_api.forward(
            history + [prompt],
            chat_model_id=self.chat_model_id,
            use_cache=self.use_cache,
            logging_role="predict_benefits_ready",
        )
        lm_output = get_last_bool_in_str(str(raw_lm_output))
        return lm_output

    def predict_benefits_eligibility(self, history, programs) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """
        reasoning_prompt = {
            "role": "user",
            "content": self.predict_benefits_reasoning_prompt.format(
                eligibility_requirements=self.eligibility_requirements,
                num_programs=self.num_programs,
                example_array=example_array(self.num_programs),
            ),
        }
        reasoning = self.lm_api.forward(
            history + [reasoning_prompt],
            logging_role="predict_benefits_eligibility",
            chat_model_id=self.chat_model_id,
            use_cache=self.use_cache,
            # constraint_type="regex",
            # constraints=rf"(True|False)(,(True|False)){{{len(programs)-1}}}",
        )
        # reasoning_turn = {
        #     "role": "user",
        #     "content": reasoning,
        # }
        decision_turn = {
            "role": "user",
            "content": self.predict_benefits_constrained_prompt.format(
                reasoning=reasoning,
                num_programs=self.num_programs,
                example_array=example_array(self.num_programs),
            ),
        }
        decision = self.lm_api.forward(
            history + [decision_turn],
            logging_role="predict_benefits_eligibility",
            chat_model_id=self.chat_model_id,
            use_cache=self.use_cache,
            constraint_type="regex",
            constraints=rf"(True|False)(,(True|False)){{{len(programs)-1}}}",
        )
        # TODO - Ensure output is a list of boolean
        # lm_output = self.extract_prediction(lm_output, programs)
        processed_output = ast.literal_eval(f"[{decision}]")
        assert len(processed_output) == len(programs)
        output_dict = dict(zip(programs, processed_output))
        return output_dict


# class ChatBotBackboneFixed(ChatBot):
#     def get_next_question(self, cur_iter_count: int) -> str:
#         counter_question_map = {
#             0: "Did you pay someone to care for your dependent so that you and your spouse, if filing a joint return, could work or look for work? Qualifying dependents are a child under age 13 at the time of care or a spouse or adult dependent who cannot physically or mentally care for themselves?",
#             1: "Did the dependent live with you for more than half of last year?",
#             2: "Did you and your spouse, if filing jointly, earn income? These can be from wages, salaries, tips, other taxable employee money, or earnings from self-employment?",
#             3: "If you are married, do both you and your spouse work outside of the home? Or, does one of you work outside of the home while the other is a full-time student, has a disability, or is looking for work?",
#             4: "Do you live in temporary housing?",
#             5: "Do you receive HRA Cash Assistance?",
#             6: "Do you receive SSI (Supplemental Security Insurance)?",
#             7: "Are you enrolling a child who is in foster care?",
#             8: "If your household income is at or below these amounts: Family size and yearly income: 1 - $14,580, 2 - $19,720, 3 - $24,860, 4 - $30,000, 5 - $35,140, 6 - $40,280, 7 - $45,420, 8 - $50,560. For each additional person, add $5,140.",
#             9: "Do you have a child age 5 or younger?",
#             10: "Do both parents have an approved reason for care, such as working 10+ hours per week, being in an educational or vocational training program, or looking for work?",
#             11: "Are you receiving treatment for substance abuse or attending services for domestic violence?",
#             12: "Is your household income at or below these amounts? Family size, monthly income, and yearly income: 1 - $4,301 (monthly), $51,610 (yearly), 2 - $5,624 (monthly), $67,490 (yearly), 3 - $6,948 (monthly), $83,370 (yearly), etc.",
#             13: "Did you earn up to $200,000, or up to $400,000 if married filing jointly?",
#             14: "Are you claiming a child who is 16 or younger with a valid SSN, ATIN, or ITIN?",
#             15: "Did your child or dependent live with you for more than half the year in the U.S., and are you claiming them as a dependent on your tax return?",
#             16: "Are you 18 years old or older?",
#             17: "Is your name on the lease?",
#             18: "Is your combined household income $50,000 or less?",
#             19: "Do you spend more than one-third of your monthly income on rent?",
#             20: "Do you live in NYC in one of the eligible housing types (rent stabilized, controlled, Mitchell-Lama, etc.)?",
#             21: "Do you have income from SSI, SSDI, VA disability pension or compensation, or disability-related Medicaid?",
#             22: "Do you have a valid Social Security Number?",
#             23: "Is your income, marital, and parental status one of the eligible cases for EITC (e.g., Married with children and earning up to $63,398)?",
#             24: "If claiming EITC without children, are you between the ages of 25 and 64?",
#             25: "Do you have investment income of less than $11,000?",
#             26: "Do you have a child aged 3-4, and are you eligible based on your income, or are you receiving HRA Cash Assistance, SSI, or SNAP?",
#             27: "Is your family income below the threshold for Head Start? Household size and yearly income: 2 - $20,440, 3 - $25,820, 4 - $31,200, etc.",
#             28: "Are you enrolling a child in a Comprehensive After School System (COMPASS) program, and are they in grades K-12?",
#         }

#         return counter_question_map[cur_iter_count]

#     def predict_cq(self, history, cur_iter_count: int) -> str:
#         """
#         Function to generate clarifying question.
#         """
#         cq = self.get_next_question(cur_iter_count)
#         return cq


# class ChatBotPredictCQPromptLoose(ChatBot):
#     def predict_cq(self, history, cur_iter_count: int) -> str:
#         """
#         Function to generate clarifying question.
#         """
#         prompt = {
#             "role": "system",
#             "content": predict_cq_prompt_loose.format(eligibility_requirements=self.eligibility_requirements)
#         }
#         cq = self.lm_api.forward(history + [prompt])
#         return cq
