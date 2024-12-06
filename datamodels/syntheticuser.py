from datamodels.userprofile import UserProfile
from models.lm_backbone import LmBackboneModel
from models.lm_logging import LmLogger
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from server.model_client import ModelAPIClient


class SyntheticUser:
    def __init__(
        self,
        hh_nl_desc: str,
        chat_model_id: str,
        use_cache: bool,
        lm_logger: LmLogger,
        top_k: int = 5,
    ):
        """
        The ground truth information about the user
        """
        # self.lm_wrapper = lm_wrapper
        self.nl_profile = hh_nl_desc
        self.chat_model_id = chat_model_id
        self.use_cache = use_cache
        # Model to answer clarifying question
        # self.oracle_model = BaseOracleModel(self.lm_wrapper, 1)
        # self.lm_backbone = LmBackboneModel(
        #     id_of_model, self.use_cache, lm_logger=lm_logger
        # )
        self.lm_api = ModelAPIClient("http://localhost:8000", lm_logger)

        # Initialize the sentence encoder model (e.g., SentenceTransformer)
        self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.profile_sentences = self.nl_profile.split("\n")
        self.profile_vectors = self.sentence_encoder.encode(self.profile_sentences)
        self.top_k = top_k

    def retrieve_relevant_context(self, question: str):
        """
        Retrieve top-k relevant sentences from the natural language profile
        based on the similarity to the question.
        """
        question_vector = self.sentence_encoder.encode(question)
        similarity_scores = util.cos_sim(question_vector, self.profile_vectors)[0]
        cur_top_k = min(
            self.top_k, len(similarity_scores)
        )  # Ensure top_k is valid and less than length of similarity score tensor
        sim_scores = np.argsort(similarity_scores)[-cur_top_k:]
        top_k_indices = torch.flip(sim_scores, dims=[0])
        relevant_sentences = [self.profile_sentences[idx] for idx in top_k_indices]
        return "\n".join(relevant_sentences)

    def answer_cq(self, cq: str):
        """
        Function to answer the question asked from the user using the user profile in natural language
        """

        if self.top_k != None:
            relevant_context = self.retrieve_relevant_context(cq)
        else:
            relevant_context = self.nl_profile

        # prompt = [
        #     {
        #         "role": "system",
        #         "content": relevant_context,
        #     },
        #     {
        #         "role": "user",
        #         "content": cq,
        #     },
        #     {
        #         "role": "system",
        #         "content": "Use the context to answer the question. Use only the information given in context and do not add any additional information. Answer the question in the first person. If you cannot answer the question from the context, respond with 'Sorry, I'm not sure.' Answer concisely. Answer only 'yes' or 'no' to yes/no questions. However, if the question assumes a fact that is not true, you should correct them.",
        #     },
        # ]
        prompt = [
            {
                "role": "system",
                "content": relevant_context,
            },
            {
                "role": "user",
                "content": "Use the context to answer the question. Use only the information given in context and do not add any additional information. Answer the question in the first person. If you cannot answer the question from the context, explain why you cannot answer the question. Answer concisely. Answer only 'yes' or 'no' to yes/no questions. However, if the question assumes a fact that is not true, you should correct them.\n\n"
                + f"Question: {cq}",
            },
        ]
        lm_output = self.lm_api.forward(
            prompt,
            chat_model_id=self.chat_model_id,
            use_cache=self.use_cache,
            logging_role="answer_cq",
        )
        return lm_output
