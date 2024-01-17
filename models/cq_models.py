from utils import *
import pandas as pd
from json import loads
from tqdm import tqdm

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
    def __init__(self, use_cache):
        self.use_cache = use_cache
        self.no_answer_str = "GPT-4 did not return a valid sentence"

    def forward(
        self,
        document: str,
        task: str,
        temperature: float = 0.7,
        model: str = "gpt-4-1106-preview",
    ) -> str:
        """
        Use the OpenAI API to answer a question given a document. Return the selected sentence.

        Parameters:
            document (str): the full document
            task (str): the task
            temperature (float): the temperature to use for the GPT model
            model (str): the name of the OpenAI model to use

        Returns:
            str: the selected sentence
        """
        lm_input = f"Context: {document}\n\nTask:{task}\n\n? You are trying to complete the task but do not have enough information from the document. Ask a question about the situation that can help you complete the task. Only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Return the question itself, exactly, in JSON format, as in {{'question': 'The question.'}}"
        completion = conditional_openai_call(
            x=lm_input,
            use_cache=self.use_cache,
            model=model,
            temperature=temperature,
            response_format="json",
        )
        # Tokenize the answer and return the first sentence
        question = loads(completion.choices[0].message.content)["question"]
        return question


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
