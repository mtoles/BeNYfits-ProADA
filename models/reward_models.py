import random
from utils import cached_openai_call
import numpy as np


class RewardModel:
    def __init__(self):
        pass

    def forward(x):
        # subclass this method
        return x


class GPTRewardModel(RewardModel):
    def __init__(self):
        pass

    def forward(
        self,
        document_full: str,
        pm_answer_full: str,
        pm_answer_summ: str,
        prompt: str,
        temperature: float,
        model="gpt-4",
    ):
        # randomly shuffle the order of the answers to avoid bias
        randomize = bool(np.random.randint(2))
        answer_a = pm_answer_full if randomize == 0 else pm_answer_summ
        answer_b = pm_answer_summ if randomize == 0 else pm_answer_full
        lm_input = f"Which of the following answers is a better answer to the question? \n\nContext: {document_full} \n\n Question: {prompt} \n\n Answer A: {answer_a} \n\n Answer B: {answer_b}\n\n Which answer is better, A or B? "
        completion = cached_openai_call(
            x=lm_input,
            model=model,
            n=1,
            temperature=temperature,
        )
        openai_output = completion.choices[0].message.content.lower()[0]
        assert openai_output in ["a", "b"]
        selected_full_doc = (openai_output == "a") ^ randomize
        return "full" if selected_full_doc else "summ"
