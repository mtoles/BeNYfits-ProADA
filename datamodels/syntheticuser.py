from models.model_utils import LanguageModelWrapper
from datamodels.userprofile import UserProfile
from models.lm_backbone import LmBackboneModel

class SyntheticUser:
    def __init__(self, user: UserProfile, hh_nl_desc: str, lm_wrapper: LanguageModelWrapper):
        """
        The grund truth information about the user
        """
        self.user = user
        self.lm_wrapper = lm_wrapper
        self.nl_profile = hh_nl_desc
        # Model to answer clarifying question
        # self.oracle_model = BaseOracleModel(self.lm_wrapper, 1)
        self.lm_backbone = LmBackboneModel(self.lm_wrapper)

    def answer_cq(self, cq: str):
        """
        Function to answer the question asked from the user using the user profile in natural language
        """
        prompt = [
            {
                "role": "system",
                "content": self.nl_profile,
            },
            {
                "role": "user",
                "content": cq,
            },
            {
                "role": "system",
                "content": "Use the context to answer the question. Use only the information given in context and do not add any additional information. Answer the question in the first person. Do not add any additional information beyond what is in the context. If you cannot answer the question from the context, respond with 'Sorry, I'm not sure.' Answer concisely. Answer only 'yes' or 'no' to yes/no questions.",
            }
        ]
        lm_output = self.lm_backbone.forward(prompt)[0]
        return lm_output

