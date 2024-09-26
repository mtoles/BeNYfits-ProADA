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
predict_cq_prompt = "You are a language model trying to help user to determine eligbility of user for benefits. Ask a clarifying question that will help you determine the eligibility of user for benefits as efficiently as possible. Only ask about one fact at a time."


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

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        lm_output = self.lm_backbone.forward(history + [benefits_ready_prompt])[0]
        return lm_output

    def predict_benefits_eligibility(self, history) -> List[bool]:
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
        return lm_output

    def predict_cq(self, history) -> str:
        """
        Function to generate clarifying question.
        """

        prompt = {
            "role": "system",
            "content": predict_cq_prompt,
        }
        cq = self.lm_backbone.forward(history + [prompt])[0]
        return cq

    def extract_prediction(
        self, prediction: str, num_programs: int
    ) -> List[Optional[str]]:
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
                assert len(bool_output) == num_programs
            except (SyntaxError, NameError, ValueError, AssertionError):
                # If the string can't be evaluated as a list, return None
                return None
            str_output = ["pass" if x else "fail" for x in bool_output]
            return str_output
        else:
            return None
