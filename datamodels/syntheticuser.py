from datamodels.userprofile import UserProfile
from models.lm_backbone import LmBackboneModel
from models.lm_logging import LmLogger
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import numpy as np
import torch
from dotenv import load_dotenv
import os
from server.model_client import ModelAPIClient
from utils import RoleEnum, rename_roles
from copy import deepcopy


class SyntheticUser:
    def __init__(
        self,
        # hh_nl_desc: str,
        # hh_nl_always_incldue: str,
        su_json: dict,
        chat_model_id: str,
        use_cache: bool,
        random_seed: int,
        lm_logger: LmLogger,
        top_k: int = 25,
    ):
        """
        The ground truth information about the user
        """
        # self.lm_wrapper = lm_wrapper
        # self.nl_profile = hh_nl_desc
        self.nl_profile = su_json["hh_nl_desc"]
        self.always_included = su_json["hh_nl_always_include"]
        self.user_name = su_json["hh"].members[0]["name"]
        self.chat_model_id = chat_model_id
        self.use_cache = use_cache
        self.random_seed = random_seed
        # Model to answer clarifying question
        # self.oracle_model = BaseOracleModel(self.lm_wrapper, 1)
        # self.lm_backbone = LmBackboneModel(
        #     id_of_model, self.use_cache, lm_logger=lm_logger
        # )
        load_dotenv(override=False)
        port = os.getenv("LM_PORT")
        self.lm_api = ModelAPIClient(
            f"http://localhost:{port}",
            random_seed=self.random_seed,
            lm_logger=lm_logger,
        )

        # Initialize the sentence encoder model (e.g., SentenceTransformer)
        # self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        # self.sentence_encoder = SentenceTransformer(
        #     "Alibaba-NLP/gte-Qwen2-7B-instruct", device="cuda:0"
        # )
        # self.tk = AutoTokenizer.from_pretrained(self.lm_api.id_of_model)
        self.profile_sentences = self.nl_profile.split("\n")
        # self.profile_vectors = self.sentence_encoder.encode(self.profile_sentences)
        self.top_k = top_k

    # def retrieve_relevant_context(self, cq: str, history: dict):
    #     """
    #     Retrieve top-k relevant sentences from the natural language profile
    #     based on the similarity to the question.
    #     """
    #     relevant_sentences = {
    #         "role": "system",
    #         "content": f"I am a user trying to determine my eligibility for certain government programs. This is some information about my household:\n{self.nl_profile}",
    #     }
    #     # invert labels on history
    #     h_ = history.copy()
    #     for i in range(len(h_)):
    #         if history[i]["role"] == "user":
    #             history[i]["role"] = "assistant"
    #         elif history[i]["role"] == "assistant":
    #             history[i]["role"] = "user"
    #     h2 = [relevant_sentences] + h_
    #     return h2
    # history_str = self.tk.apply_chat_template(h2, tokenize=False)
    # question_vector = self.sentence_encoder.encode(history_str, prompt_name="query")
    # similarity_scores = util.cos_sim(question_vector, self.profile_vectors)[0]
    # cur_top_k = min(
    #     self.top_k, len(similarity_scores)
    # )  # Ensure top_k is valid and less than length of similarity score tensor
    # sim_scores = np.argsort(similarity_scores)[-cur_top_k:]
    # top_k_indices = torch.flip(sim_scores, dims=[0])
    # relevant_sentences = [self.profile_sentences[idx] for idx in top_k_indices]
    # relevant_sentences = self.profile_sentences
    # return self.always_included + "\n" + "\n".join(relevant_sentences)

    def answer_cq(self, cq: str, history: dict):
        """
        Function to answer the question asked from the user using the user profile in natural language
        """
        if self.chat_model_id == "human":
            print()
            print()
            return input(f"*** Assistant: {cq} ***\n>>> User     : ")

        relevant_sentences = {
            "role": "system",
            "content": f"Your instructions are to play the role of a person trying to determine their eligibility for certain government programs. This is some information about my household:\n{self.nl_profile}",
        }
        # invert labels on history

        h_ = rename_roles(history, invert=False)

        h2 = [relevant_sentences] + h_[:-1]

        prompt = (
            h2
            + [
                {
                    "role": "user",
                    "content": "Use the context to answer the question. Use only the information given in context and do not add any additional information. Answer the question in the first person. If you cannot answer the question from the context, explain why you cannot answer the question. Answer concisely. Answer only 'yes' or 'no' to yes/no questions. However, if the question assumes a fact that is not true, you should correct them.\n\n"
                    + f"Question: {cq}",
                    # "content": cq,
                },
            ]
            # + h2[-1:]
        )
        lm_output = self.lm_api.forward(
            prompt,
            chat_model_id=self.chat_model_id,
            use_cache=self.use_cache,
            logging_role="answer_cq",
        )
        return lm_output

        # if self.top_k != None:
        #     relevant_context = self.retrieve_relevant_context(cq=cq, history=history)
        # else:
        #     relevant_context = self.nl_profile

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
