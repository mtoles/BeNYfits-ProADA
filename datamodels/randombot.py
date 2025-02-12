import ast
from typing import List, Optional
from datamodels.chatbot import ChatBot, example_array, get_last_bool_in_str
from models.lm_logging import LmLogger
import random

class RandomBot(ChatBot):
    """Class for chatbot that uses an actual human to give predictions"""

    def __init__(
        self,
        chat_model_id: str,
        no_of_programs: str,
        eligibility_requirements,
        use_cache: bool,
        random_seed: int,
        lm_logger: Optional[LmLogger] = None,
    ):
        super().__init__(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_requirements,
            use_cache=use_cache,
            lm_logger=lm_logger,
            random_seed=random_seed,
        )
    
    def predict_cq(self, history, chat_model_id) -> str:
        return ""

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        
        return "True"

    def predict_benefits_eligibility(self, history, programs) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """
        output_dict = {}

        for program in programs:
            
            output_dict[program] = True if random.random() >=  0.5 else False
        
        return output_dict