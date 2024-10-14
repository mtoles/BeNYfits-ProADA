from models.model_utils import LanguageModelWrapper
from models.lm_backbone import LmBackboneModel
from typing import List, Optional
from copy import deepcopy
import re
import ast
import numpy as np
import re
from models.lm_logging import LmLogger
from inspect import currentframe

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
    """Get the last True or False mentioned in a string"""
    pattern = r"True|False"
    match = re.search(pattern, s.lower())
    if match:
        # return match.group()
        group = match.group()
        # capitalize the first letter
        return group[0].upper() + group[1:]
    else:
        return "False"


class ChatBot:
    """ "Base class for chatbots. Serves as the simple backbone model."""

    def __init__(
        self,
        lm_wrapper: LanguageModelWrapper,
        no_of_programs: str,
        eligibility_requirements: str,
        lm_logger: Optional[LmLogger] = None,
    ):
        """
        ChatBot class for keeping the history of user chat and other functions to determine eligbility for benefits
        """
        self.benefits_ready_prompt = {
            "role": "system",
            "content": "Is the information sufficient to determine eligibility of all programs? Answer only in one word True or False.",
        }
        self.benefits_prediction_prompt = "Predict the programs for which the user is eligible. Return only a boolean array of length {num_programs}, e.g. {example_array}, where the value at index `i` is true iff the user is eligible for program `i`. Only return the array. Do not return anything else in the response. If a user's eligibility is unclear, make your best guess."
        self.predict_cq_prompt = "Ask a clarifying question that will help you determine the eligibility of user for benefits as efficiently as possible. Only ask about one fact at a time."
        self.lm_wrapper = lm_wrapper
        self.lm_backbone = LmBackboneModel(self.lm_wrapper, lm_logger=lm_logger)
        self.num_programs = no_of_programs
    
    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        raw_lm_output = self.lm_backbone.forward(
            history + [self.benefits_ready_prompt],
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
            "role": "system",
            "content": self.benefits_prediction_prompt.format(
                num_programs=self.num_programs,
                example_array=example_array(self.num_programs),
            ),
        }
        lm_output = self.lm_backbone.forward(
            history + [prompt], logging_role="predict_benefits_eligibility"
        )
        # TODO - Ensure output is a list of boolean
        lm_output = self.extract_prediction(lm_output, programs)
        return lm_output

    def predict_cq(self, history, cur_iter_count: int) -> str:
        """
        Function to generate clarifying question.
        """
        prompt = {
            "role": "system",
            "content": self.predict_cq_prompt,
        }
        cq = self.lm_backbone.forward(history + [prompt], logging_role="predict_cq")
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

    def pre_conversation(self, eligibility_requirements: str = None):
        """Function called before a new conversation is started"""
        pass


class CotChatBot(ChatBot):
    """Class for chatbot that uses chain-of-thought"""

    def __init__(
        self,
        lm_wrapper: LanguageModelWrapper,
        no_of_programs: str,
        eligibility_requirements: str,
        lm_logger: Optional[LmLogger] = None,
    ):
        super().__init__(
            lm_wrapper, no_of_programs, eligibility_requirements, lm_logger
        )
        self.benefits_ready_prompt = {
            "role": "system",
            "content": "Is the information sufficient to determine eligibility of all programs? Think through your reasoning out loud. Then answer with True or False.",
        }
        self.benefits_prediction_prompt = "Predict the programs for which the user is eligible. Think through your reasoning out loud, then output a boolean array of length {num_programs}, e.g. {example_array}, where the value at index `i` is true iff the user is eligible for program `i`. If a user's eligibility is unclear, make your best guess."
        self.predict_cq_prompt = "Ask a clarifying question that will help you determine the eligibility of user for benefits as efficiently as possible. Only ask about one fact at a time."


class NotetakerChatBot(ChatBot):
    """Class for chatbots that can take notes"""

    def __init__(
        self,
        lm_wrapper: LanguageModelWrapper,
        no_of_programs: str,
        eligibility_requirements: str,
        notebook_only: bool,
        lm_logger: Optional[LmLogger] = None,
    ):
        """
        ChatBot class for keeping the history of user chat and other functions to determine eligibility for benefits
        """
        super().__init__(
            lm_wrapper, no_of_programs, eligibility_requirements, lm_logger
        )
        self.notebook_only = notebook_only

        # Prompts inside the class
        self.initialize_notebook_prompt = (
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

        self.update_notebook_prompt = (
            "Below is an annotated notebook. You have just received new information about the user. Update the notebook with the new information by replacing <UNK> with values learned from the dialog turn. For example:\n\n"
            "Dialog turn:\n"
            "Chatbot: Are you a plumber?\n"
            "User: Yes, I am a plumber.\n\n"
            "Original Notebook:\n"
            "For you to be eligible for the working plumber tax credit, both you <UNK> and your spouse <UNK> must be a plumber and have a combined income of less than $100,000 <UNK>.\n\n"
            "Updated Notebook:\n"
            "For you to be eligible for the working plumber tax credit, both you <True> and your spouse <UNK> must be a plumber and have a combined income of less than $100,000 <UNK>.\n\n"
            "Valid values are: <True>, <False>, numbers, and <N/A - Reason> for requirements that are not applicable. For example, if the user does not have a spouse, you would replace <UNK> with <N/A - User has no spouse>.\n"
            "Do not update any <UNK> tags unless you know FOR SURE the correct value. It is OK to return a notebook with no changes. Return ONLY the notebook. Do not return anything else in the response.\n\n"
            "Here is your dialog turn:\n"
            "Chatbot: {cq}\n"
            "User: {answer}\n\n"
            "Here is your notebook:\n"
            "{notebook_page}\n\n"
            "Your updated notebook:\n"
        )

        self.notebook_guidance_prompt = (
            "Notebooks are annotated with notes in <angle brackets>. The annotations indicate:\n"
            "<UNK> - The requirement is currently unknown\n"
            "<True> - The user meets the requirement\n"
            "<False> - The user does not meet the requirement\n"
            "<a number or string> - The user's value for the requirement\n"
            "<N/A - Reason> - The requirement is not applicable and the reason why\n"
        )

        self.notebook_guidance_turn = {
            "role": "system",
            "content": self.notebook_guidance_prompt,
        }

    def pre_conversation(self, eligibility_requirements: str = None):
        """
        Initialize the 'notebook'
        Add a 'page' of notes to the 'notebook' based on the most recent dialog turn
        """
        prompt = {
            "role": "system",
            "content": self.initialize_notebook_prompt.format(
                eligibility_requirements=eligibility_requirements
            ),
        }
        self.notebook = [
            self.lm_backbone.forward([prompt], logging_role="initialize_notebook")
        ]

    def post_answer(self, history: List[dict]) -> None:
        """
        Update the notebook based on the most recent dialog turn
        """
        # prompt the lm to update the notebook until it returns a valid update
        for i in range(10):
            history = deepcopy(history)
            prompts = [
                {
                    "role": "system",
                    "content": self.update_notebook_prompt.format(
                        notebook_page=self.notebook[-1],
                        cq=history[-2]["content"],
                        answer=history[-1]["content"],
                    ),
                },
                self.notebook_guidance_turn,
            ]
            lm_output = self.lm_backbone.forward(
                prompts, num_completions=i + 1, logging_role="post_answer"
            )[-1]
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
        history = deepcopy(history)
        prompts = [
            {
                "role": "system",
                "content": self.predict_cq_prompt,
            }
        ]

        # replace the eligibility requirements with the most recent notebook page
        history[0]["content"] = self.notebook[-1]
        history.insert(0, self.notebook_guidance_turn)
        cq = self.lm_backbone.forward(history + prompts, logging_role="predict_cq")
        return cq

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligibility of all benefits
        """
        history = deepcopy(history)
        history[0]["content"] = self.notebook[-1]
        # if the notebook contains no <UNK> tags, short circuit and return True
        if "<UNK>" not in self.notebook[-1]:
            return True
        history.insert(0, self.notebook_guidance_turn)
        lm_output = self.lm_backbone.forward(
            history + [benefits_ready_prompt], logging_role="predict_benefits_ready"
        )
        return str(lm_output)

    def predict_benefits_eligibility(self, history, programs) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """
        history = deepcopy(history)
        if self.notebook_only:
            prompts = [
                {"role": "system", "content": self.notebook[-1]},
                self.notebook_guidance_turn,
                {
                    "role": "system",
                    "content": self.benefits_prediction_prompt.format(
                        num_programs=self.num_programs,
                        example_array=example_array(self.num_programs),
                    ),
                },
                {
                    "role": "system",
                    "content": "Base your prediction on the <annotations> in the notebook.",
                },
            ]
            lm_output = self.lm_backbone.forward(
                prompts, logging_role="predict_benefits_eligibility"
            )
        else:
            prompts = [
                {
                    "role": "system",
                    "content": self.benefits_prediction_prompt.format(
                        num_programs=self.num_programs,
                        example_array=example_array(self.num_programs),
                    ),
                }
            ]
            history[0]["content"] = self.notebook[-1]
            history.insert(0, self.notebook_guidance_turn)
            lm_output = self.lm_backbone.forward(
                history + prompts, logging_role="predict_benefits_eligibility"
            )
        lm_output = self.extract_prediction(lm_output, programs)
        return lm_output


class ChatBotBackboneFixed(ChatBot):
    def get_next_question(self, cur_iter_count: int) -> str:
        counter_question_map = {
            0: "Did you pay someone to care for your dependent so that you and your spouse, if filing a joint return, could work or look for work? Qualifying dependents are a child under age 13 at the time of care or a spouse or adult dependent who cannot physically or mentally care for themselves?",
            1: "Did the dependent live with you for more than half of last year?",
            2: "Did you and your spouse, if filing jointly, earn income? These can be from wages, salaries, tips, other taxable employee money, or earnings from self-employment?",
            3: "If you are married, do both you and your spouse work outside of the home? Or, does one of you work outside of the home while the other is a full-time student, has a disability, or is looking for work?",
            4: "Do you live in temporary housing?",
            5: "Do you receive HRA Cash Assistance?",
            6: "Do you receive SSI (Supplemental Security Insurance)?",
            7: "Are you enrolling a child who is in foster care?",
            8: "If your household income is at or below these amounts: Family size and yearly income: 1 - $14,580, 2 - $19,720, 3 - $24,860, 4 - $30,000, 5 - $35,140, 6 - $40,280, 7 - $45,420, 8 - $50,560. For each additional person, add $5,140.",
            9: "Do you have a child age 5 or younger?",
            10: "Do both parents have an approved reason for care, such as working 10+ hours per week, being in an educational or vocational training program, or looking for work?",
            11: "Are you receiving treatment for substance abuse or attending services for domestic violence?",
            12: "Is your household income at or below these amounts? Family size, monthly income, and yearly income: 1 - $4,301 (monthly), $51,610 (yearly), 2 - $5,624 (monthly), $67,490 (yearly), 3 - $6,948 (monthly), $83,370 (yearly), etc.",
            13: "Did you earn up to $200,000, or up to $400,000 if married filing jointly?",
            14: "Are you claiming a child who is 16 or younger with a valid SSN, ATIN, or ITIN?",
            15: "Did your child or dependent live with you for more than half the year in the U.S., and are you claiming them as a dependent on your tax return?",
            16: "Are you 18 years old or older?",
            17: "Is your name on the lease?",
            18: "Is your combined household income $50,000 or less?",
            19: "Do you spend more than one-third of your monthly income on rent?",
            20: "Do you live in NYC in one of the eligible housing types (rent stabilized, controlled, Mitchell-Lama, etc.)?",
            21: "Do you have income from SSI, SSDI, VA disability pension or compensation, or disability-related Medicaid?",
            22: "Do you have a valid Social Security Number?",
            23: "Is your income, marital, and parental status one of the eligible cases for EITC (e.g., Married with children and earning up to $63,398)?",
            24: "If claiming EITC without children, are you between the ages of 25 and 64?",
            25: "Do you have investment income of less than $11,000?",
            26: "Do you have a child aged 3-4, and are you eligible based on your income, or are you receiving HRA Cash Assistance, SSI, or SNAP?",
            27: "Is your family income below the threshold for Head Start? Household size and yearly income: 2 - $20,440, 3 - $25,820, 4 - $31,200, etc.",
            28: "Are you enrolling a child in a Comprehensive After School System (COMPASS) program, and are they in grades K-12?"
        }

        return counter_question_map[cur_iter_count]
    
    def predict_cq(self, history, cur_iter_count: int) -> str:
        """
        Function to generate clarifying question.
        """
        cq = self.get_next_question(cur_iter_count)
        return cq
    
class ChatBotPredictCQPromptLoose(ChatBot):
    def predict_cq(self, history, cur_iter_count: int) -> str:
        """
        Function to generate clarifying question.
        """
        prompt = {
            "role": "system",
            "content": predict_cq_prompt_loose,
        }
        cq = self.lm_backbone.forward(history + [prompt])
        return cq