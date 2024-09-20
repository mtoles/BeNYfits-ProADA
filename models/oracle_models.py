from utils import *
from typing import List, Dict, Tuple, Union, Optional
from json import loads
import os
from huggingface_hub import login
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
from models.utils import ModelFamily


class BaseOracleModel:
    def __init__(self, lm_wrapper, batch_size):
        super().__init__()

        self.main_instruction = "Use the context to answer the question. Use only the information given in context and do not add any additional information. Answer the question in the first person. Do not add any additional information beyond what is in the context. If you cannot answer the question from the context, respond with 'Sorry, I'm not sure.' Answer concisely. Answer only 'yes' or 'no' to yes/no questions."

        self.lm_wrapper = lm_wrapper
        self.batch_size = batch_size

        # self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_api_key = os.getenv("HF_TOKEN")
        login(token=self.hf_api_key)

    def _format_llama_prompt(self, document: str, question: str) -> str:
        user_prompt = "{question_string}"
        formatted_user_messages = [
            {
                "role": "system",
                "content": f"{self.main_instruction}\n\nContext:\n\n{document}",
            },
            {
                "role": "user",
                "content": user_prompt.format(question_string=question),
            },
        ]
        return self.lm_wrapper.language_model._tokenizer.apply_chat_template(
            formatted_user_messages, tokenize=False, add_generation_prompt=True
        )

    def _format_gpt_prompt(self, document: str, question: str) -> str:
        # json_instruction = "Return the answer in JSON form, i.e. {{'answer': 'the answer here'}}."
        json_instruction = "Use the following context to answer the user's question: "  # no json instruction for GPT
        return f"Context: {document}\n\n{self.main_instruction} {json_instruction}\n\nQuestion: {question}\n\nAnswer:"

    def _format_gemma_prompt(self, document: str, question: str) -> str:
        return f"<start_of_turn>system\nUse the following context to answer the user's question: {document}\n<end_of_turn>\n<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n"

    def _format_mistral_prompt(self, document: str, question: str) -> str:
        return f"[INST] Context: {document}\n\n{self.main_instruction}\n\nQuestion: {question} [/INST]"

    def _format_default_prompt(self, document: str, question: str) -> str:
        # json_instruction = "Return the answer in JSON form, i.e. {{'answer': 'the answer here'}}."
        json_instruction = "Use the following context to answer the user's question: "  # no json instruction
        return f"Context: {document}\n\n{self.main_instruction} {json_instruction}\n\nQuestion:\n\n{question}"

    def forward_batch(self, documents: List[str], questions: List[str]) -> List[str]:
        format_func = {
            ModelFamily.LLAMA: self._format_llama_prompt,
            ModelFamily.GPT: self._format_gpt_prompt,
            ModelFamily.GEMMA: self._format_gemma_prompt,
            ModelFamily.MISTRAL: self._format_mistral_prompt,
        }.get(self.lm_wrapper.family, self._format_default_prompt)

        formatted_prompts = [
            format_func(doc, question) for doc, question in zip(documents, questions)
        ]

        # print("--"*20)
        # print(f"Prompt for Oracle Model:")
        # for p in formatted_prompts:
        #     print(p)
        # print("--"*20)

        sequences = self.lm_wrapper.language_model.predict_many(
            [LmPrompt(p, cache=False, max_tokens=512) for p in formatted_prompts],
            completion_window=CompletionWindow.ASAP,
        )

        outputs = [x.completion_text for x in sequences]
        return outputs


# testing
if __name__ == "__main__":
    document = (
        "My name is Matt. I wrote this code. I am a student at Columbia University."
    )
    question1 = "What is my name?"
    question2 = "What did I write?"
    question3 = "Where do I go to school?"
