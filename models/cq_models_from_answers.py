import dotenv
dotenv.load_dotenv()

from utils import *
import pandas as pd
from json import loads
from tqdm import tqdm
from typing import List, Tuple
import os

from cq_prompt import cq_prompt

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
        answer: str,
        n_clarifying_questions: int = 5,
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
        lm_input = cq_prompt.format(reddit_post=document, answer=answer)

        # lm_input = f"Context: {document}\n\nTask:{task}\n\n? You are trying to complete the task but do not have enough information from the document. Ask {n_clarifying_questions} question{'s' if n_clarifying_questions > 1 else ''} about the situation that can help you complete the task. In each question, only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Return the questions as a list in JSON format, as in {{'questions': ['The first question?', 'The second question?']}}"
        completion = conditional_openai_call(
            x=lm_input,
            use_cache=self.use_cache,
            model=self.model_name,
            temperature=temperature,
            response_format="json",
        )
        # Tokenize the answer and return the first sentence
        questions = loads(completion.choices[0].message.content)["question"]
        return questions

    # todo: add iterative question asking where the model accounts for the answer to each question before asking another one


def df_to_md_grouped(df: pd.DataFrame, output_path: str, groupby: List = ["doc_orig", "doc_summ", "prompt"]):
    """
    Convert a dataframe to a markdown table and save to disk.

    Parameters:
        df (pd.DataFrame): the dataframe to convert. Must have columns "subreddit", "doc_orig", "doc_summ", "prompt", "pm_answer_full", "pm_answer_summ", "selection"
        output_path (str): the path to save the markdown file
    """
    # delete the existing file and create a new one
    with open(output_path, "w") as f:
        col_to_header = {
            "subreddit": "subreddit",
            "doc_orig": "original document",
            "doc_summ": "summary document",
            "prompt": "prompt",
            "pm_answer_full": "full answer",
            "pm_answer_summ": "summary answer",
            "cq": "clarifying question",
            "selection": "selection",
        }

        for _, group in df.groupby(groupby):
            substrs = []
            for col_name in groupby:
                if col_name in col_to_header:
                    substrs.append(f"## {col_to_header[col_name]}")
                else:
                    substrs.append(f"## {col_name}")
                substrs.append(str(group.iloc[0][col_name]))

            substrs.append(f"\n\n{'='*50}\n\n")
            for i, row in group.iterrows():
                for col_name, val in row.items():
                    if col_name in groupby:
                        continue

                    if col_name in col_to_header:
                        substrs.append(f"## {col_to_header[col_name]}")
                    else:
                        substrs.append(f"## {col_name}")
                    substrs.append(str(val))
                substrs.append(f"\n\n{'-'*50}\n\n")
            md_row = "\n\n".join(substrs)
            f.write(md_row)

if __name__ == "__main__":
    ### testing
    
    # print(os.getenv("OPENAI_API_KEY"))
    # print("HI")

    # load data
    df = pd.read_json("results/unsafe/llama2-13b_200_summ-preference_unsafe.json")
    # load model
    use_cache = True
    model = GPTClarifyingQuestionModel(use_cache=use_cache)
    # apply model

    if not os.path.exists("results/cq/llama2-13b_20_cq_fromanswer_sample.jsonl"):
        answers = df['doc_orig'].map(lambda x: x.split('.'))
        answers = answers.map(lambda x: [y.strip() for y in x])
        df["answer"] = answers
        df = df.explode("answer")

        df = df.sample(frac=1).head(100)
        print(f"Number of sampled answers {len(df)}")

        df.to_json("results/cq/llama2-13b_20_cq_fromanswer_sample.jsonl", orient="records", lines=True)
    else:
        df = pd.read_json("results/cq/llama2-13b_20_cq_fromanswer_sample.jsonl", lines=True)

    df["cq"] = df.progress_apply(
        lambda x: model.forward(x["doc_summ"], x["prompt"], x["answer"]), axis=1
    )

    # save as jsonl using pandas
    df.to_json("results/cq/llama2-13b_20_cq_fromanswer.jsonl", orient="records", lines=True)

    # save results
    df_to_md(df, "results/cq/llama2-13b_20_cq_new_fromanswer.md")

    short_df = df[["doc_orig", "doc_summ", "prompt", "answer", "cq"]]
    df_to_md_grouped(short_df, "results/cq/llama2-13b_20_cq_new_fromanswer_short.md")
