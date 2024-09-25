from models.model_utils import LanguageModelWrapper
from models.cq_models import BaseClarifyingQuestionModel
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


benefits_prediction_prompt = "Return only a boolean array of length {num_programs}, e.g. {example_array} determining if the user or any member in its houehold is eligible for the benefits. Only return the array. Do not return anything else in the response."
predict_cq_prompt = "You are a language model trying to help user to determine eligbility of user for benefits. Ask a clarifying question that will help you determine the eligibility of user for benefits as quickly as possible."


class ChatBot:
    def __init__(
        self, lm_wrapper: LanguageModelWrapper, no_of_programs: str, history: str
    ):
        """
        ChatBot class for keeping the history of user chat and other functions to determine eligbility for benefits
        """
        # self.history = history
        self.lm_wrapper = lm_wrapper
        self.cq_model = BaseClarifyingQuestionModel(self.lm_wrapper)
        self.num_programs = no_of_programs

    # def _format_llama_prompt(self, question: str) -> str:
    #     formatted_user_messages = [
    #         {
    #             "role": "system",
    #             "content": f"{self.history}",
    #         },
    #         {
    #             "role": "user",
    #             "content": f"{question}",
    #         },
    #     ]
    #     return self.lm_wrapper.language_model._tokenizer.apply_chat_template(
    #         formatted_user_messages, tokenize=False, add_generation_prompt=True
    #     )

    # def _format_gpt_prompt(self, question: str) -> str:
    #     json_instruction = (
    #         # "Return the answer in JSON form, i.e. {{'answer': 'the answer here'}}."
    #         ""  # Not using JSON here, match with Llama
    #     )
    #     return f"Context: {self.history}\n\n{json_instruction}\n\nQuestion: {question}\n\nAnswer:"

    # def _format_default_prompt(self, question: str) -> str:
    #     json_instruction = (
    #         # "Return the answer in JSON form, i.e. {{'answer': 'the answer here'}}."
    #         ""  # Not using JSON here, match with Llama
    #     )
    #     return f"Context: {self.history}\n\n{json_instruction}\n\nQuestion: {question}\n\nAnswer:"

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        lm_output = self.cq_model.forward(history + [benefits_ready_prompt])[0]
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
        # sequences = list(self.cq_model.forward(history + [prompt]))
        lm_output = self.cq_model.forward(history + [prompt])[0]
        # TODO - Ensure output is a list of boolean
        return lm_output

    def predict_cq(self, history) -> str:
        """
        Function to generate clarifying question.
        """
        # cq = self.cq_model.forward_batch_generate_single_question(
        # [self.history], [task]
        # )[0]
        prompt = {
            "role": "system",
            "content": predict_cq_prompt,
        }
        cq = self.cq_model.forward(history + [prompt])[0]
        # cq = list(sequences)[0].completion_text

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
