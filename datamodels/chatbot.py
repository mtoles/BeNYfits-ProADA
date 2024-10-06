from models.model_utils import LanguageModelWrapper
from models.lm_backbone import LmBackboneModel
from typing import List, Optional
import re
import ast
import numpy as np
import re

np.random.seed(42)

benefits_ready_prompt = {
    "role": "system",
    "content": "Is the information sufficient to determine eligibility of all programs? Answer only in one word True or False.",
}


def example_array(n):
    return str([bool(x % 2) for x in range(n)])


### Backbone Prompts ###
benefits_prediction_prompt = "Predict the programs for which the user is eligible. Return only a boolean array of length {num_programs}, e.g. {example_array}, where the value at index `i` is true iff the user is eligible for program `i`. Only return the array. Do not return anything else in the response. If a user's eligibility is unclear, make your best guess."
predict_cq_prompt = "Ask a clarifying question that will help you determine the eligibility of user for benefits as efficiently as possible. Only ask about one fact at a time."


class ChatBot:
    """ "Base class for chatbots. Serves as the simple backbone model."""

    def __init__(
        self,
        lm_wrapper: LanguageModelWrapper,
        no_of_programs: str,
        eligibility_requirements: str,
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
        lm_output = self.lm_backbone.forward(history + [benefits_ready_prompt])
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
        lm_output = self.lm_backbone.forward(history + [prompt])
        # TODO - Ensure output is a list of boolean
        lm_output = self.extract_prediction(lm_output, programs)
        return lm_output

    def predict_cq(self, history) -> str:
        """
        Function to generate clarifying question.
        """

        prompt = {
            "role": "system",
            "content": predict_cq_prompt,
        }
        cq = self.lm_backbone.forward(history + [prompt])
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
                str_output = ["pass" if x else "fail" for x in bool_output]
                output = str_output
            except (SyntaxError, NameError, ValueError, AssertionError):
                # If the string can't be evaluated as a list, return None
                output = None
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


### Notebook Prompts ###
initialize_notebook_prompt = (
    "Below is a set of requirements for a user's eligibility. Currently you know nothing about the user. Annotate each unknown fact in the eligibility document with <UNK>. For example:\n\n"
    "Original Document:\n"
    "For you to be eligible for the working plumber tax credit, both you and your spouse must be a plumber and have a combined income of less than $100,000.\n\n"
    "Annotated Document:\n"
    "For you to be eligible for the working plumber tax credit, both you <UNK> and your spouse <UNK> must be a plumber and have a combined income of less than $100,000 <UNK>.\n"
    "Return ONLY the annotated document. Do not return anything else in the response. Only change the <UNK> tags to the correct values. Do not remove or add new tags. Make sure there is at least one <UNK> per requirement.\n\n"
    "Here is your document:\n"
    "{eligibility_requirements}"
    "Your annotation:"
)
update_notebook_prompt = (
    "Below is an annotated notebook. You have just received new information about the user. Update the notebook with the new information by replacing <UNK> with values learned from the dialog turn. For example:\n\n"
    #
    "Dialog turn:\n"
    "Chatbot: Are you a plumber?\n"
    "User: Yes, I am a plumber.\n\n"
    #
    "Original Notebook:\n"
    "For you to be eligible for the working plumber tax credit, both you <UNK> and your spouse <UNK> must be a plumber and have a combined income of less than $100,000 <UNK>.\n\n"
    #
    "Updated Notebook:\n"
    "For you to be eligible for the working plumber tax credit, both you <True> and your spouse <UNK> must be a plumber and have a combined income of less than $100,000 <UNK>.\n\n"
    #
    "Valid values are: <True>, <False>, numbers, and <N/A - Reason> for requirements that are not applicable. For example, if the user does not have a spouse, you would replace <UNK> with <N/A - User has no spouse>.\n"
    "Do not update any <UNK> tags unless you know FOR SURE the correct value. It is OK to return a notebook with no changes. Return ONLY the notebook. Do not return anything else in the response.\n\n"
    #
    "Here is your dialog turn:\n"
    "Chatbot: {cq}\n"
    "User: {answer}\n\n"
    "Here is your notebook:\n"
    "{notebook_page}\n\n"
    "Your updated notebook:\n"
)
notebook_guidance_prompt = (
    "Notebooks are annotated with notes in <angle brackets>. The annotations indicate:\n"
    "<UNK> - The requirement is currently unknown\n"
    "<True> - The user meets the requirement\n"
    "<False> - The user does not meet the requirement\n"
    "<a number or string> - The user's value for the requirement\n"
    "<N/A - Reason> - The requirement is not applicable and the reason why\n"
)
notebook_guidance_turn = {
    "role": "system",
    "content": notebook_guidance_prompt,
}


class NotetakerChatBot(ChatBot):
    """ "Class for chatbots that can take notes."""

    def __init__(
        self,
        lm_wrapper: LanguageModelWrapper,
        no_of_programs: str,
        eligibility_requirements: str,
        notebook_only: bool,
    ):
        """
        ChatBot class for keeping the history of user chat and other functions to determine eligbility for benefits
        """
        self.lm_wrapper = lm_wrapper
        self.lm_backbone = LmBackboneModel(self.lm_wrapper)
        self.num_programs = no_of_programs
        self.notebook = [self.initialize_notebook(eligibility_requirements)]
        self.notebook_only = notebook_only

    def initialize_notebook(self, eligibility_requirements: str):
        """
        add a 'page' of notes to the 'notebook' based on the most recent dialog turn
        """
        prompt = {
            "role": "system",
            "content": initialize_notebook_prompt.format(
                eligibility_requirements=eligibility_requirements
            ),
        }
        lm_output = self.lm_backbone.forward([prompt])
        return lm_output

    def post_answer(self, history: List[dict]) -> None:
        """
        Update the notebook based on the most recent dialog turn
        """
        # prompt the lm to update the notebook until it returns a valid update
        for i in range(10):
            history = history.copy()
            prompts = [
                {
                    "role": "system",
                    "content": update_notebook_prompt.format(
                        notebook_page=self.notebook[-1],
                        cq=history[-2]["content"],
                        answer=history[-1]["content"],
                    ),
                },
                notebook_guidance_turn,
            ]
            lm_output = self.lm_backbone.forward(prompts, num_completions=i + 1)[-1]
            if self.validate_notebook_update(self.notebook[-1], lm_output):
                break
            else:
                print(
                    f"*** WARNING: Invalid notebook update. Attempting to update again. ***"
                )
        print(lm_output)
        self.notebook.append(lm_output)

    def validate_notebook_update(self, old_notebook: str, new_notebook: str) -> bool:
        """Validate the update to the notebook"""
        old_notebook, new_notebook = old_notebook.strip(), new_notebook.strip()
        # check the notebooks contain the same number of <...> tags
        old_notebook_tags = re.findall(r"<.*?>", old_notebook)
        new_notebook_tags = re.findall(r"<.*?>", new_notebook)
        if len(old_notebook_tags) != len(new_notebook_tags):
            return False
        # remove anything inside <...> tags
        old_notebook_ = re.sub(r"<.*?>", "", old_notebook)
        new_notebook_ = re.sub(r"<.*?>", "", new_notebook)
        # check the notebooks are different
        notebooks_are_same = old_notebook_ == new_notebook_
        return notebooks_are_same

    def predict_cq(self, history) -> str:
        """
        Function to generate clarifying question.
        """
        history = history.copy()
        prompts = [
            {
                "role": "system",
                "content": predict_cq_prompt,
            }
        ]

        # replace the eligibility requirements with the most recent notebook page
        history[0]["content"] = self.notebook[-1]
        history.insert(0, notebook_guidance_turn)
        cq = self.lm_backbone.forward(history + prompts)
        return cq

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        history = history.copy()
        history[0]["content"] = self.notebook[-1]
        # if the notebook contains no <UNK> tags, short circuit and return True
        if "<UNK>" not in self.notebook[-1]:
            return True
        history.insert(0, notebook_guidance_turn)
        lm_output = self.lm_backbone.forward(history + [benefits_ready_prompt])
        return str(lm_output)

    def predict_benefits_eligibility(
        self, history, programs
    ) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """
        history = history.copy()
        if self.notebook_only:
            prompts = [
                {"role": "system", "content": self.notebook[-1]},
                notebook_guidance_turn,
                {
                    "role": "system",
                    "content": benefits_prediction_prompt.format(
                        num_programs=self.num_programs,
                        example_array=example_array(self.num_programs),
                    ),
                },
                {
                    "role": "system",
                    "content": "Base your prediction on the <annotations> in the notebook.",
                },
            ]
            lm_output = self.lm_backbone.forward(prompts)
        else:
            prompts = [
                {
                    "role": "system",
                    "content": benefits_prediction_prompt.format(
                        num_programs=self.num_programs,
                        example_array=example_array(self.num_programs),
                    ),
                }
            ]
            history[0]["content"] = self.notebook[-1]
            history.insert(0, notebook_guidance_turn)
            lm_output = self.lm_backbone.forward(history + prompts)
        lm_output = self.extract_prediction(lm_output, programs)
        return lm_output
