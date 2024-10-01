from models.model_utils import LanguageModelWrapper
from models.lm_backbone import LmBackboneModel
from typing import List, Optional
import re
import ast
import numpy as np

np.random.seed(42)

benefits_ready_prompt = {
    "role": "system",
    "content": "Is the information sufficient to determine eligibility of all programs? Answer only in one word True or False.",
}


def example_array(n):
    return str([bool(x % 2) for x in range(n)])


benefits_prediction_prompt = "Predict the programs for which the user is eligible. Return only a boolean array of length {num_programs}, e.g. {example_array}, where the value at index `i` is true iff the user is eligible for program `i`. Only return the array. Do not return anything else in the response. If a user's eligibility is unclear, make your best guess."
predict_cq_prompt = "Ask a clarifying question that will help you determine the eligibility of user for benefits as efficiently as possible. Only ask about one fact at a time."

predict_cq_prompt_1 = "Ask a clarifying question to determine the user's eligibility for benefits for each program one requirement at a time. Begin with the first program and ask about one specific requirement before moving to the next. Ensure that you ask questions in a clear, concise, and logical sequence, focusing on gathering the necessary information efficiently."

class ChatBot:
    def __init__(
        self, lm_wrapper: LanguageModelWrapper, no_of_programs: str, history: str
    ):
        """
        ChatBot class for keeping the history of user chat and other functions to determine eligbility for benefits
        """
        self.lm_wrapper = lm_wrapper
        self.lm_backbone = LmBackboneModel(self.lm_wrapper)
        self.num_programs = no_of_programs

    def get_next_question(self, cur_iter_count: int) -> str:
        counter_question_map = {
            0: "Did you pay someone to care for your dependent so that you and your spouse, if filing a joint return, could work or look for work? Qualifying dependents are a child under age 13 at the time of care or a spouse or adult dependent who cannot physically or mentally care for themselves?",
            1: "Did the dependent live with you for more than half of 2023?",
            2: "Did you and your spouse, if filing jointly, earn income? These can be from wages, salaries, tips, other taxable employee money, or earnings from self-employment?",
            3: "If you are married, do both you and your spouse work outside of the home? Or, does one of you work outside of the home while the other is a full-time student, has a disability, or is looking for work?",
            4: "Do you live in temporary housing?",
            5: "Do you receive HRA Cash Assistance?",
            6: "Do you receive SSI (Supplemental Security Insurance)?",
            7: "Are you enrolling a child who is in foster care?",
            8: "If your household income is at or below these amounts: Family size and yearly income: 1 - $14,580, 2 - $19,720, 3 - $24,860, 4 - $30,000, 5 - $35,140, 6 - $40,280, 7 - $45,420, 8 - $50,560. For each additional person, add $5,140.",
        }

        return counter_question_map[cur_iter_count]

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        lm_output = self.lm_backbone.forward(history + [benefits_ready_prompt])[0]
        return lm_output

    def predict_benefits_eligibility(self, history, programs) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """
        prompt = {
            "role": "system",
            "content": benefits_prediction_prompt.format(
                num_programs=self.num_programs,
                example_array=example_array(self.num_programs),
            ),
        }
        lm_output = self.lm_backbone.forward(history + [prompt])[0]
        # TODO - Ensure output is a list of boolean
        lm_output = self.extract_prediction(lm_output, programs)
        return lm_output

    def predict_cq(self, history, ask_cq_mode: int, cur_iter_count: int) -> str:
        """
        Function to generate clarifying question.
        """

        if ask_cq_mode == 0:
            prompt = {
                "role": "system",
                "content": predict_cq_prompt,
            }
            cq = self.lm_backbone.forward(history + [prompt])[0]
            return cq
        elif ask_cq_mode == 1:
            prompt = {
                "role": "system",
                "content": predict_cq_prompt_1,
            }
            cq = self.lm_backbone.forward(history + [prompt])[0]
            return cq
        else:
            cq = self.get_next_question(cur_iter_count)
            return cq

    def extract_prediction(
        self, prediction: str, programs: list[str]
    ) -> List[Optional[int]]:
        """
        Extract the prediction from the model output
        """
        # Regex to match a list-like structure in the string
        pattern = r"\[.*?\]"
        # Find the first list-like match in the string
        match = re.search(pattern, prediction)
        if match:
            # Extract the matched portion and safely evaluate it
            extracted_list_str = match.group(0)
            # Safely evaluate the string into a Python list
            try:
                bool_output = ast.literal_eval(extracted_list_str)
                assert isinstance(bool_output, list)
                assert len(bool_output) == len(programs)
            except (SyntaxError, NameError, ValueError, AssertionError):
                # If the string can't be evaluated as a list, return None
                output = None
            str_output = ["pass" if x else "fail" for x in bool_output]
            output = str_output
        else:
            output = None
        if output is None:
            # default to all fails
            print(
                "*** WARNING: Could not extract prediction from model output. Defaulting to all fails. ***"
            )
            output = ["fail"] * len(programs)
        output = [1 if x == "pass" else 0 for x in output]
        # convert to dict
        output = {programs[i]: output[i] for i in range(len(programs))}
        return output
