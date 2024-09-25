from models.model_utils import load_lm, LanguageModelWrapper
from models.cq_models import BaseClarifyingQuestionModel
from typing import List, Optional
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
import re
import ast
import numpy as np

np.random.seed(42)


class ChatBot:
    def __init__(
        self, lm_wrapper: LanguageModelWrapper, no_of_programs: str, history: str
    ):
        """
        ChatBot class for keeping the history of user chat and other functions to determine eligbility for benefits
        """
        self.history = history
        self.lm_wrapper = lm_wrapper
        self.cq_model = BaseClarifyingQuestionModel(self.lm_wrapper)
        self.no_of_programs = no_of_programs

    def _format_llama_prompt(self, question: str) -> str:
        formatted_user_messages = [
            {
                "role": "system",
                "content": f"{self.history}",
            },
            {
                "role": "user",
                "content": f"{question}",
            },
        ]
        return self.lm_wrapper.language_model._tokenizer.apply_chat_template(
            formatted_user_messages, tokenize=False, add_generation_prompt=True
        )

    def _format_gpt_prompt(self, question: str) -> str:
        json_instruction = (
            # "Return the answer in JSON form, i.e. {{'answer': 'the answer here'}}."
            ""  # Not using JSON here, match with Llama
        )
        return f"Context: {self.history}\n\n{json_instruction}\n\nQuestion: {question}\n\nAnswer:"

    def _format_default_prompt(self, question: str) -> str:
        json_instruction = (
            # "Return the answer in JSON form, i.e. {{'answer': 'the answer here'}}."
            ""  # Not using JSON here, match with Llama
        )
        return f"Context: {self.history}\n\n{json_instruction}\n\nQuestion: {question}\n\nAnswer:"

    def benefits_ready(self) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        benefits_ready_question = "Is the information sufficient to determine eligibility of all programs? Answer only in one word True or False."
        format_func = {
            "llama": self._format_llama_prompt,
            "gpt": self._format_gpt_prompt,
            "gemma": self._format_default_prompt,
            "mistral": self._format_default_prompt,
        }.get(self.lm_wrapper.family, self._format_default_prompt)

        formatted_prompt = format_func(benefits_ready_question)

        # print("--"*20)
        # print(f"Prompt for Checking Benefits are Ready:")
        # print(formatted_prompt)
        # print("--"*20)

        sequences = list(
            self.lm_wrapper.language_model.predict_many(
                ([LmPrompt(formatted_prompt, cache=False, max_tokens=512)]),
                completion_window=CompletionWindow.ASAP,
            )
        )

        output = sequences[0].completion_text

        # print("--"*20)
        # print(f"RESULT: Are Benefits Ready? : {output}")
        # print("--"*20)

        return output

    def predict_benefits_eligibility(self) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """

        def example_array(n):
            return str([bool(x % 2) for x in range(n)])

        benefits_ready_question = f"Return only a boolean array of length {self.no_of_programs}, e.g. {example_array(self.no_of_programs)} determining if the user or any member in its houehold is eligible for the benefits. Only return the array. Do not return anything else in the response."
        format_func = {
            "llama": self._format_llama_prompt,
            "gpt": self._format_gpt_prompt,
        }.get(self.lm_wrapper.family, self._format_default_prompt)

        formatted_prompt = format_func(benefits_ready_question)

        print("--" * 20)
        print(f"Prompt for Predicting Benefits Eligbility:")
        print(formatted_prompt)
        print("--" * 20)

        sequences = list(
            self.lm_wrapper.language_model.predict_many(
                ([LmPrompt(formatted_prompt, cache=False, max_tokens=512)]),
                completion_window=CompletionWindow.ASAP,
            )
        )

        output = sequences[0].completion_text
        # TODO - Ensure output is a list of boolean
        return output

    def ask_cq(self) -> str:
        """
        Function to generate clarifying question.
        """
        task = "You are a language model trying to help user to determine eligbility of user for benefits. Ask a clarifying question that will help you determine the eligibility of user for benefits as quickly as possible."
        cq = self.cq_model.forward_batch_generate_single_question(
            [self.history], [task]
        )[0]
        return cq

    def append_chat_question_and_answer(self, clarifying_question: str, clarifying_answer: str):
        self.history = f"{self.history}\n\n{clarifying_question}: {clarifying_answer}"

    def print_chat_history(self):
        print("==" * 30)
        print("Chat History: ")
        print(self.history)
        print("==" * 30)

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
        #     # If no list-like structure is found, return None
        #     bool_output = [None] * num_programs
        # # extracted_list = ["pass" if x else "fail" for x in extracted_list]
        # str_output = []
        # for i, x in enumerate(bool_output):
        #     if x is None:
        #         str_output.append(None)
        #     else:
        #         str_output.append("pass" if x else "fail")
        # return str_output
