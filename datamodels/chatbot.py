from models.utils import load_lm, LanguageModelWrapper
from models.cq_models import BaseClarifyingQuestionModel
from typing import List
from models.utils import ModelFamily
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow

class ChatBot:
    def __init__(self, lm_wrapper: LanguageModelWrapper, no_of_programs: str, history: str):
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
        json_instruction = "Return the answer in JSON form, i.e. {{'answer': 'the answer here'}}."
        return f"Context: {self.history}\n\n{json_instruction}\n\nQuestion: {question}\n\nAnswer:"

    def _format_default_prompt(self, question: str) -> str:
        json_instruction = "Return the answer in JSON form, i.e. {{'answer': 'the answer here'}}."
        return f"Context: {self.history}\n\n{json_instruction}\n\nQuestion: {question}\n\nAnswer:"

    def benefits_ready(self) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all beenfits
        """
        benefits_ready_question = "Is the information sufficient to determine eligibility of all programs? Answer only in one word True or False."
        format_func = {
            ModelFamily.LLAMA: self._format_llama_prompt,
            ModelFamily.GPT: self._format_gpt_prompt,
            ModelFamily.GEMMA: self._format_default_prompt,
            ModelFamily.MISTRAL: self._format_default_prompt
        }.get(self.lm_wrapper.family, self._format_default_prompt)

        formatted_prompt = format_func(benefits_ready_question)

        print("--"*20)
        print(f"Prompt for Checking Benefits are Ready:")
        print(formatted_prompt)
        print("--"*20)

        sequences = list(self.lm_wrapper.language_model.predict_many(
            ([LmPrompt(formatted_prompt, cache=False, max_tokens=512)]),
            completion_window=CompletionWindow.ASAP,
        ))

        output = sequences[0].completion_text

        print("--"*20)
        print(f"RESULT: Are Benefits Ready? : {output}")
        print("--"*20)

        return output
    
    def predict_benefits_eligibility(self) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """
        benefits_ready_question = f"Return only a boolean array of length {self.no_of_programs} determining if the user or any member in its houehold is eligible for the benefits. Do not return anything else in the response."
        format_func = {
            ModelFamily.LLAMA: self._format_llama_prompt,
            ModelFamily.GPT: self._format_gpt_prompt,
        }.get(self.lm_wrapper.family, self._format_default_prompt)

        formatted_prompt = format_func(benefits_ready_question)

        print("--"*20)
        print(f"Prompt for Predicting Benefits Eligbility:")
        print(formatted_prompt)
        print("--"*20)

        sequences = list(self.lm_wrapper.language_model.predict_many(
            ([LmPrompt(formatted_prompt, cache=False)]),
            completion_window=CompletionWindow.ASAP,
        ))

        output = sequences[0].completion_text
        # TODO - Ensure output is a list of boolean
        return output
        
    def ask_cq(self) -> str:
        """
        Function to generate clarifying question.
        """
        task = "You are a language model trying to help user to determine eligbility of user for benefits."
        cq = self.cq_model.forward_batch_generate_single_question([self.history], [task])[0]
        return cq
    
    def append_chat_history_with_cq_answer(self, cq_answer: str):
        self.history = self.history + cq_answer

    def print_chat_history(self):
        print("=="*30)
        print("Chat History: ")
        print(self.history)
        print("=="*30)
