from utils import *
import pandas as pd
from json import loads
from tqdm import tqdm
from typing import List

tqdm.pandas()


class Clarifying_Question_Model:
    """
    Model to generate a clarifying question given a document and a question. Subclass this model.
    """

    def __init__(self):
        pass

    def forward(document: str, task: str) -> str:
        # subclass this method
        return "What is six times seven?"


class GPTClarifyingQuestionModel(Clarifying_Question_Model):
    def __init__(self, use_cache, model_name="gpt-4-1106-preview"):
        self.use_cache = use_cache
        self.no_answer_str = "GPT-4 did not return a valid sentence"
        self.model_name = model_name

    # def forward(
    #     self,
    #     document: str,
    #     task: str,
    #     temperature: float = 0.7,
    # ) -> str:
    #     """
    #     Use the OpenAI API to answer a question given a document. Return the selected sentence.

    #     Parameters:
    #         document (str): the full document
    #         task (str): the task
    #         temperature (float): the temperature to use for the GPT model

    #     Returns:
    #         str: the selected sentence
    #     """
    #     lm_input = f"Context: {document}\n\nTask:{task}\n\n? You are trying to complete the task but do not have enough information from the document. Ask a question about the situation that can help you complete the task. Only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Return the question itself, exactly, in JSON format, as in {{'question': 'The question.'}}"
    #     completion = conditional_openai_call(
    #         x=lm_input,
    #         use_cache=self.use_cache,
    #         model=self.model_name,
    #         temperature=temperature,
    #         response_format="json",
    #     )
    #     # Tokenize the answer and return the first sentence
    #     question = loads(completion.choices[0].message.content)["question"]
    #     return question

    def forward(
        self,
        document: str,
        task: str,
        n_clarifying_questions: int,
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Use the OpenAI API to ask multiple questions about a document. Return the selected sentence.

        Parameters:
            document (str): the full document
            task (str): the task
            temperature (float): the temperature to use for the GPT model

        Returns:
            List[str]: the selected sentence
        """
        lm_input = f"Context: {document}\n\nTask:{task}\n\n? You are trying to complete the task but do not have enough information from the document. Ask {n_clarifying_questions} question{'s' if n_clarifying_questions > 1 else ''} about the situation that can help you complete the task. In each question, only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Return the questions as a list in JSON format, as in {{'questions': ['The first question?', 'The second question?']}}"
        completion = conditional_openai_call(
            x=lm_input,
            use_cache=self.use_cache,
            model=self.model_name,
            temperature=temperature,
            response_format="json",
        )
        # Tokenize the answer and return the first sentence
        questions = loads(completion.choices[0].message.content)["questions"]
        return questions

    # todo: add iterative question asking where the model accounts for the answer to each question before asking another one


if __name__ == "__main__":
    ### testing

    # load data
    df = pd.read_json("results/unsafe/llama2-13b_200_summ-preference_unsafe.json").head(
        20
    )
    # load model
    model = GPTClarifyingQuestionModel(use_cache=False)
    # apply model
    df["cq"] = df.progress_apply(
        lambda x: model.forward(x["doc_summ"], x["prompt"]), axis=1
    )
    # save results
    df_to_md(df, "results/cq/cq_test.md")
    print
