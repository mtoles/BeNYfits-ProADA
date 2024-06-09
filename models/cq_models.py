from utils import *
import pandas as pd
from json import loads
from tqdm import tqdm
from typing import List

from transformers import AutoTokenizer
import transformers
import os
from huggingface_hub import login
import torch

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
        # todo: unify with the llama mode
        lm_input = f"Context: {document}\n\nTask:{task}?\n\n You are trying to complete the task but do not have enough information from the document. Ask {n_clarifying_questions} question{'s' if n_clarifying_questions > 1 else ''} about the situation that can help you complete the task. In each question, only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Return the questions as a list in JSON format, as in {{'questions': ['The first question?', 'The second question?']}}"
        completion = conditional_openai_call(
            x=lm_input,
            use_cache=self.use_cache,
            model=self.model_name,
            temperature=temperature,
            response_format="json",
        )
        # Tokenize the answer and return the first sentence
        questions = loads(completion.choices[0].message.content)["questions"]
        # assert len(questions) == n_clarifying_questions
        if len(questions) > n_clarifying_questions:
            questions = questions[:n_clarifying_questions]
        elif len(questions) < n_clarifying_questions:
            questions += self.forward(
                document, task, n_clarifying_questions - len(questions)
            )
        return questions

    # todo: add iterative question asking where the model accounts for the answer to each question before asking another one

class GPTExperimentalClarifyingQuestionModel(Clarifying_Question_Model):
    def __init__(self, use_cache, model_name="gpt-4-1106-preview"):
        self.use_cache = use_cache
        self.no_answer_str = "GPT-4 did not return a valid sentence"
        self.model_name = model_name

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
        # todo: unify with the llama mode
        lm_input = f"Context: {document}\n\nTask:{task}?\n\n Instructions: You need to complete the task but lack sufficient information from the document. Follow these steps to ask one specific clarifying question:\n1. Identify the Gap: What specific information do you need to complete the task that is not in the document?\n2. Reference the Document: Can you refer to a particular section or detail in the document related to this gap?\n3. Formulate the Question: Create a clear, concise question that asks for the missing information. Ensure the question only seeks one fact at a time and does not merely rephrase the original task.\nReturn the questions as a list in JSON format, like this: {{'questions': ['The first question?', 'The second question?']}}"
        # lm_input = f"Context: {document}\n\nTask:{task}?\n\n You are trying to complete the task but do not have enough information from the document. Ask {n_clarifying_questions} question{'s' if n_clarifying_questions > 1 else ''} about the situation that can help you complete the task. In each question, only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Return the questions as a list in JSON format, as in {{'questions': ['The first question?', 'The second question?']}}"
        completion = conditional_openai_call(
            x=lm_input,
            use_cache=self.use_cache,
            model=self.model_name,
            temperature=temperature,
            response_format="json",
        )
        # Tokenize the answer and return the first sentence
        questions = loads(completion.choices[0].message.content)["questions"]
        # assert len(questions) == n_clarifying_questions
        if len(questions) > n_clarifying_questions:
            questions = questions[:n_clarifying_questions]
        elif len(questions) < n_clarifying_questions:
            questions += self.forward(
                document, task, n_clarifying_questions - len(questions)
            )
        return questions

    # todo: add iterative question asking where the model accounts for the answer to each question before asking another one

class Llama2PrimaryModel(Clarifying_Question_Model):
    """
    Llama2 chat primary model.
    """

    def __init__(self, model_size, batch_size):
        raise NotImplementedError # This model still needs a prompt template
        super().__init__()
        if model_size == "7b":
            self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        elif model_size == "13b":
            self.model_name = "meta-llama/Llama-2-13b-chat-hf"
        elif model_size == "70b":
            self.model_name = "meta-llama/Llama-2-70b-chat-hf"
        else:
            raise ValueError(f"Unknown llama2 model size {model_size}")
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        login(token=self.hf_api_key)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.pipeline.tokenizer.pad_token_id = 0
        self.pipeline.tokenizer.padding_side = "left"
        # self.system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
        self.system_prompt = "You are a helpful assistant. Always answer the question and be faithful to the provided document."

        self.batch_size = batch_size

    def process(
        self,
        document: pd.Series,
        task: pd.Series,
    ) -> pd.Series:
        llama_formatted_input = [
            f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"
            for instruction in instructions
        ]
        # wrap the pipeline so we can have a progress bar
        sequences = []
        for i in tqdm(range(0, len(llama_formatted_input), self.batch_size)):
            batch = llama_formatted_input[i : i + self.batch_size]
            sequences.extend(
                self.pipeline(
                    batch,
                    # do_sample=True,
                    # top_k=10,
                    # num_return_sequences=1,
                    # eos_token_id=self.tokenizer.eos_token_id,
                    # max_length=300,
                )
            )

        outputs = [sequence[0]["generated_text"] for sequence in sequences]
        # delete the prompt
        # outputs = [output[len(llama_formatted_input) :] for output in outputs]
        outputs = [x[len(y) :] for x, y in zip(outputs, llama_formatted_input)]
        return outputs


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
