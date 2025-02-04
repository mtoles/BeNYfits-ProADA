import ast
from typing import List, Optional
from datamodels.chatbot import ChatBot, example_array, get_last_bool_in_str
from models.lm_logging import LmLogger


class HumanBot(ChatBot):
    """Class for chatbot that uses an actual human to give predictions"""

    def __init__(
        self,
        chat_model_id: str,
        no_of_programs: str,
        eligibility_requirements,
        use_cache: bool,
        random_seed: int,
        lm_logger: Optional[LmLogger] = None,
        target_programs: Optional[List[str]] = None,
    ):
        super().__init__(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_requirements,
            use_cache=use_cache,
            lm_logger=lm_logger,
            random_seed=random_seed,
        )
        self.target_programs = target_programs

        print("** Your task is to predict benefits eligibility for the following program names: ")

        print("*" * 20)

        for program in self.target_programs:
            print(f"* {program}")

        print("*" * 20)

    def predict_cq(self, history, chat_model_id) -> str:
        print("** Ask a clarifying question and solicit information from the user that will help you deterimine benefits eligibility ** \n")
        cq = str(input())

        return cq

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """

        while True:
            print("** Based on the information you have gathered so far, do you think you can determine eligibility for all programs in question? Say only a single character: Y/N **\n")
            c = str(input()).lower()
            if c in ["y", "n"]:
                break
        
        return "True" if c == 'y' else "False"

    def predict_benefits_eligibility(self, history, programs) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """
        output_dict = {}

        for program in programs:
            while True:
                print(f"** Based on the information you have gathered so far, do you think the user is eligible for the program named {program}? Say only a single character: Y/N: **\n")
                c = str(input()).lower()
                if c in ["y", "n"]:
                    break
        
            output_dict[program] = True if c == "y" else False
        
        return output_dict