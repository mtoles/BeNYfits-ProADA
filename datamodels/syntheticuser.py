from models.utils import LanguageModelWrapper
from models.oracle_models import BaseOracleModel
from datamodels.userprofile import UserProfile

class SyntheticUser:
    def __init__(self, user: UserProfile, lm_wrapper: LanguageModelWrapper):
        """
        The grund truth information about the user
        """
        self.user = user
        self.lm_wrapper = lm_wrapper
        self.nl_profile = self.get_nl_profile(user)
        # Model to answer clarifying question
        self.oracle_model = BaseOracleModel(self.lm_wrapper, 1)

    def get_nl_profile(self, user: UserProfile):
        """
        Function to return the description of the user profile in natural language
        """
        # TODO - Convert user to NL user
        return ""
    
    def answer_cq(self, cq: str):
        """
        Function to answer the question asked from the user using the user profile in natural language
        """
        cq_answer = self.oracle_model.forward_batch([self.nl_profile], [cq])[0]
        return cq_answer

