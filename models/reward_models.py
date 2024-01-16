import random
from utils import *
import numpy as np
from alpaca_eval import evaluate
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd
import os
from json import loads


class RewardModel:
    """
    Model used to rank two outputs from a primary model. Subclass this class to implement a new reward model. Technically more of a ranking model right now.
    """

    def __init__(self):
        pass

    def forward(x: str) -> str:
        # subclass this method
        return x


class GPTRewardModel(RewardModel):
    def __init__(self, use_cache):
        self.use_cache = use_cache

    def forward(
        self,
        document_full: str,
        pm_answer_full: str,
        pm_answer_summ: str,
        prompt: str,
        temperature: float,
        model="gpt-4-1106-preview",
    ) -> str:
        """
        Use the OpenAI API to rank two outputs from a primary model. Return the selected output.

        Parameters:
            document_full (str): the full document
            pm_answer_full (str): the pm answer given the full document
            pm_answer_summ (str): the pm answer given the summary of the document
            prompt (str): the task prompt
            temperature (float): the temperature to use for the GPT model
            model (str): the name of the OpenAI model to use

        Returns:
            str: the selected output {"full", "summ"}
        """
        # randomly shuffle the order of the answers to avoid bias
        randomize = bool(np.random.randint(2))
        answer_a = pm_answer_full if randomize == 0 else pm_answer_summ
        answer_b = pm_answer_summ if randomize == 0 else pm_answer_full
        lm_input = f"Which of the following answers is a better answer to the question? \n\nContext: {document_full} \n\n Question: {prompt} \n\n Answer A: {answer_a} \n\n Answer B: {answer_b}\n\n Which answer is better, A or B? Consider how helpful, specific, and factually correct the response is with respect to the original document. Respond in JSON format, as in {{'choice': 'A'}}"
        completion = conditional_openai_call(
            x=lm_input,
            use_cache=self.use_cache,
            model=model,
            temperature=temperature,
            response_format="json",
        )
        openai_output = loads(completion.choices[0].message.content.lower())[
            "choice"
        ].upper()
        assert openai_output in ["A", "B"]
        selected_full_doc = (openai_output == "A") ^ randomize
        return "full" if selected_full_doc else "summ"


def run_alpaca_eval(
    model_outputs: pd.Series, reference_outputs: pd.Series, instruction: pd.Series
) -> pd.Series:
    """
    Run alpaca eval on the model outputs and reference outputs. Append a column containing the winner {1: model_outputs, 2: reference_outputs} to the model_outputs dataframe. Return the dataframe.}

    Parameters:
        model_outputs (pd.Series): the model outputs (e.g. full document pm outputs)
        reference_outputs (pd.Series): the reference outputs (e.g. summary pm outputs)
        instruction (pd.Series): prompts

    Returns:
        pd.Series: the winner for each annotation where 0 indicates reference_output and 1 indicates model_output
    """

    # form a dataframe from model_outputs and instructions
    model_outputs = pd.DataFrame({"output": model_outputs, "instruction": instruction})
    reference_outputs = pd.DataFrame(
        {"output": reference_outputs, "instruction": instruction}
    )

    # run alpaca_eval
    df_leaderboard, annotations = evaluate(
        model_outputs=model_outputs,
        reference_outputs=reference_outputs,
        is_return_instead_of_print=True,
        output_path="alpaca_eval/",
        annotators_config=os.path.join(os.getcwd(), "annotators_config.yaml"),
    )

    # get the winner for each annotation since I can't figure out
    # how to disable random output ordering in alpaca eval
    winners = []
    for i in range(len(annotations)):
        preference = annotations[i]["preference"]
        if preference is None:
            winners.append("None")
            continue
        choice_output = annotations[i][f"output_{preference}"]
        output_1 = annotations[i]["output_1"]
        output_2 = annotations[i]["output_2"]
        if choice_output == output_1:
            winning_model = 0
        elif choice_output == output_2:
            winning_model = 1
        else:
            raise ValueError("chosen output not found in input")
        winners.append(winning_model)
    return pd.Series(winners)
