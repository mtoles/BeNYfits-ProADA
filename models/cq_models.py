from utils import *
import pandas as pd
from json import loads
from tqdm import tqdm
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer
import transformers
import os
from huggingface_hub import login
import torch
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow
from models.utils import ModelFamily
from typing import List, Callable
from enum import Enum

tqdm.pandas()

### TEMPLATES ###

benchmark_template = "Context: {document}\n\nTask:{task}\n\nYou are trying to complete the task but do not have enough information from the document. Ask {n_clarifying_questions} question{plural} about the situation that can help you complete the task. In each question, only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Do not say anything other than the question. {json}"
bechmark_template_json = 'Return the questions as a list in JSON format, as in {{"questions": ["The first question?", "The second question?"]}}'

ambig_cot_template_1 = 'Context: {document}\n\nTask:{task}\n\n? You are trying to complete the task but do not have enough information from the document. Identify 5 sources of ambiguity in the situation. Return your answer as as a list in JSON format, as in {{"ambiguities": ["The first ambiguity", "The second ambiguity"]}}'
ambig_cot_template_2 = 'Task: {task}\n\nAmbiguities:\n\n{ambigs_str}\n\nYou are trying to complete the task but do not have enough information from the document. Identify the most important ambiguity in the situation. Return your answer as a json dict in the form {{"best_ambiguity": 1}}'
ambig_cot_template_3 = 'Context: {document}\n\nAmbiguity:\n\n{best_ambig}\n\nGenerate a clarifying question that will help you resolve the ambiguity in the context. Return the question itself, exactly, in JSON format, as in {{"question": "The question."}} Do not return anything besides the JSON.'

imagine_answers_template = "Context: {document}\n\nQuestion: {cq}\n\nAlthough you do not have enough information to answer the question, imagine you are the writer of the context and provide a plausible answer. Stay in character and only respond with the answer to the question."


### CLASSES ###
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
    def __init__(self, model_name, use_cache):
        self.use_cache = use_cache
        self.no_answer_str = "GPT-4 did not return a valid sentence"
        self.model_name = model_name

    def forward_single(
        self,
        document: str,
        task: str,
        # n_clarifying_questions: int,
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
        # plural = "s" if n_clarifying_questions > 1 else ""
        lm_input = benchmark_template.format(
            document=document,
            task=task,
            n_clarifying_questions=1,
            plural=False,
            json=bechmark_template_json,
        )
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

    def forward(
        self,
        documents: List[str],
        tasks: List[str],
        # n_clarifying_questions: int,
        temperature: float = 0.7,
    ) -> List[List[str]]:
        """
        Use the OpenAI API to ask multiple questions about a document. Return the selected sentence.

        Parameters:
            document (str): the full document
            task (str): the task
            temperature (float): the temperature to use for the GPT model

        Returns:
            List[str]: the selected sentence
        """
        # plural = "s" if n_clarifying_questions > 1 else ""
        lm_inputs = [
            benchmark_template.format(
                document=doc,
                task=task,
                n_clarifying_questions=1,
                plural=False,
                json=bechmark_template_json,
            )
            for doc, task in zip(documents, tasks)
        ]
        completions = []
        for lmi in lm_inputs:
            completions.append(
                conditional_openai_call(
                    x=lmi,
                    use_cache=self.use_cache,
                    model=self.model_name,
                    temperature=temperature,
                    response_format="json",
                )
            )
        questions = []
        for completion in completions:
            questions.append(loads(completion.choices[0].message.content)["questions"][0])
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
        # n_clarifying_questions: int,
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
        # if len(questions) > n_clarifying_questions:
        #     questions = questions[:n_clarifying_questions]
        # elif len(questions) < n_clarifying_questions:
        #     questions += self.forward(
        #         document, task, n_clarifying_questions - len(questions)
        #     )
        return questions

    # todo: add iterative question asking where the model accounts for the answer to each question before asking another one


class Llama3ClarifyingQuestionModel(Clarifying_Question_Model):
    """
    Llama3 CQ Model.
    """

    def __init__(self, model_name, batch_size, pipeline):
        super().__init__()
        assert model_name in [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
        ]
        self.model_name = model_name
        self.pipeline = pipeline
        # self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        # login(token=self.hf_api_key)
        # if pipeline:
        #     self.pipeline = pipeline
        # else:
        #     self.pipeline = transformers.pipeline(
        #         "text-generation",
        #         model=self.model_name,
        #         model_kwargs={"torch_dtype": torch.bfloat16},
        #         device_map="auto",
        #     )
        self.no_answer_str = "Llama3 did not return a valid sentence"
        # self.user_prompt = "{question_string}"

        self.batch_size = batch_size

    def forward(
        self, documents: List[str], tasks: List[str]
    ) -> List[str]:
        candidate_cqs = []
        # for i in range(N_QUESTIONS):
        benchmark_content = [
            benchmark_template.format(
                document=doc,
                task=task,
                n_clarifying_questions=1,
                plural="s",
                json="",
            )
            for doc, task in zip(documents, tasks)
        ]
        llama_formatted_prompts = self._format_and_apply_chat_template(
            benchmark_content
        )
        # use predict instead of predict_many until David adds batching to the lib
        for i in range(len(llama_formatted_prompts)):
            sequence = self.pipeline.predict(
                LmPrompt(llama_formatted_prompts[i], cache=False),
                # completion_window=CompletionWindow.ASAP,
            )
            candidate_cqs.append(sequence.completion_text)
        return candidate_cqs

    # def forward(self, documents: List[str], questions: List[str]) -> List[str]:
    #     assert len(documents) == len(
    #         questions
    #     ), "The length of the documents list must be equal to the length of the questions list."

    #     results = []
    #     n_batches = len(documents) // self.batch_size + (
    #         0 if len(documents) % self.batch_size == 0 else 1
    #     )

    #     for i in tqdm(range(n_batches)):
    #         batch_documents = documents[i * self.batch_size : (i + 1) * self.batch_size]
    #         batch_questions = questions[i * self.batch_size : (i + 1) * self.batch_size]

    #         batch_results = self.forward_batch(batch_documents, batch_questions, 1)
    #         results.extend(batch_results)

    #     return results

    # def forward_list(self, document: str, tasks: List[str]) -> List[str]:
    #     # forward but for a list of questions
    #     return self.forward([document] * len(questions), questions)

    def _format_and_apply_chat_template(self, content: List[str]) -> List[str]:
        formatted_messages = []
        for i, message in enumerate(content):
            formatted_message = [
                {
                    "role": "system",
                    "content": f"You are trying to help the user with the task below.",
                },
                {
                    "role": "user",
                    "content": message,
                },
            ]
            formatted_messages.append(formatted_message)
        llama_formatted_prompts = [
            self.pipeline._tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            for prompt in formatted_messages
        ]
        return llama_formatted_prompts


class Llama3ImagineClarifyingQuestionModel(Llama3ClarifyingQuestionModel):
    """
    Llama3 CQ Model.
    """

    def forward(
        self,
        documents: List[str],
        tasks: List[str],
    ) -> List[str]:
        # generate candidate questions
        N_QUESTIONS = 4
        N_ANSWERS = 4
        # IMAGINATION_TEMP = 1.0
        candidate_cqs = []
        # for i in range(N_QUESTIONS):
        benchmark_content = [
            benchmark_template.format(
                document=doc,
                task=task,
                n_clarifying_questions=1,
                plural="s",
                json="",
            )
            for doc, task in zip(documents, tasks)
        ]
        llama_formatted_prompts = self._format_and_apply_chat_template(
            benchmark_content
        )
        # use predict instead of predict_many until David adds batching to the lib
        for i in range(len(llama_formatted_prompts)):
            sequences = self.pipeline.predict(
                LmPrompt(
                    llama_formatted_prompts[i], num_completions=N_QUESTIONS, cache=False
                ),
                # completion_window=CompletionWindow.ASAP,
            )
            candidate_cqs.extend([x.completion_text for x in sequences])
        repeated_docs = [doc for doc in documents for _ in range(N_QUESTIONS)]
        imagine_answers_content = [
            imagine_answers_template.format(document=doc, cq=cq)
            for doc, cq in zip(repeated_docs, candidate_cqs)
        ]
        llama_formatted_imagine_prompts = self._format_and_apply_chat_template(
            imagine_answers_content
        )
        sequences = []
        for i, p in enumerate(llama_formatted_imagine_prompts):
            lm_prompt = LmPrompt(p, num_completions=N_ANSWERS, cache=False)
            sequences.extend(self.pipeline.predict(lm_prompt))
        completions = []
        completions.extend(sequences)
        imagined_answers = [x.completion_text for x in completions]
        # imagined_answers.extend(completions)
        imagined_answers_np = np.array(imagined_answers).reshape(
            len(documents), N_QUESTIONS, N_ANSWERS
        )
        q_similarity = np.zeros((len(documents), N_QUESTIONS))
        sbert = SentenceTransformer("all-mpnet-base-v2")
        for i in range(len(documents)):
            for j in range(N_QUESTIONS):
                embeddings = sbert.encode(imagined_answers_np[i, j, :])
                similarities = sbert.similarity(embeddings, embeddings)
                q_similarity[i, j] = similarities.mean()
        # get the question with the lowest answer similarity
        best_questions = []
        np_candidate_cqs = np.array(candidate_cqs).reshape(
            len(documents), N_QUESTIONS
        )
        for i in range(len(documents)):
            best_questions.append(np_candidate_cqs[i][q_similarity[i].argmin()])
        return best_questions

    def forward_list(self, document: str, questions: List[str]) -> List[str]:
        # forward but for a list of questions
        return self.forward([document] * len(questions), questions)

class GPTCOTClarifyingQuestionModel(Clarifying_Question_Model):
    def __init__(self, use_cache, model_name="gpt-4-1106-preview"):
        self.use_cache = use_cache
        self.no_answer_str = "Llama3 did not return a valid sentence"
        self.model_name = model_name

    def forward(
        self,
        document: str,
        task: str,
        n_ambiguities: int = 5,
        temperature: float = 0.0,
    ) -> List[str]:
        """ """
        # todo: unify with the llama mode
        # lm_input1 = f"Context: {document}\n\nTask:{task}\n\n? You are trying to complete the task but do not have enough information from the document. Identify 5 sources of ambiguity in the situation. Return your answer as as a list in JSON format, as in {{'ambiguities': ['The first ambiguity', 'The second ambiguity']}}"
        lm_input1 = ambig_cot_template_1.format(document=document, task=task)
        completion1 = conditional_openai_call(
            x=lm_input1,
            use_cache=self.use_cache,
            model=self.model_name,
            temperature=temperature,
            response_format="json",
        )
        # Tokenize the answer and return the first sentence
        ambigs = loads(completion1.choices[0].message.content)["ambiguities"]
        # assert len(questions) == n_clarifying_questions
        if len(ambigs) > n_ambiguities:
            ambigs = ambigs[:n_ambiguities]
        elif len(ambigs) < n_ambiguities:
            ambigs += self.forward(document, task, n_ambiguities - len(ambigs))
        ambigs_str = "\n\n".join(
            [str(i + 1) + ". " + ambig for i, ambig in enumerate(ambigs)]
        )
        lm_input2 = ambig_cot_template_2.format(task=task, ambigs_str=ambigs_str)

        completion2 = conditional_openai_call(
            x=lm_input2,
            use_cache=self.use_cache,
            model=self.model_name,
            temperature=temperature,
            response_format="json",
        )
        best_ambig = loads(completion2.choices[0].message.content)["best_ambiguity"]
        lm_input3 = ambig_cot_template_3.format(
            document=document, best_ambig=ambigs[best_ambig - 1]
        )
        completion3 = conditional_openai_call(
            x=lm_input3,
            use_cache=self.use_cache,
            model=self.model_name,
            temperature=temperature,
            response_format="json",
        )
        clarifying_question = loads(completion3.choices[0].message.content)["question"]
        return clarifying_question

    def forward_batch(
        self,
        documents: List[str],
        tasks: List[str],
        n_ambiguities: int = 5,
        temperature: float = 0.0,
    ) -> List[str]:
        """ """
        clarifying_questions = []
        for doc, task in zip(documents, tasks):
            clarifying_questions.append(
                self.forward(doc, task, n_ambiguities, temperature)
            )
        return clarifying_questions

class PromptMode(Enum):
    DEFAULT = "default"

class BaseClarifyingQuestionModel:
    def __init__(self, lm_wrapper, mode: PromptMode = PromptMode.DEFAULT):
        self.lm_wrapper = lm_wrapper
        self.mode = mode
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        login(token=self.hf_api_key)

    def _get_format_func(self) -> Callable:
        format_funcs = {
            (ModelFamily.LLAMA, PromptMode.DEFAULT): self._format_llama_prompt_default,
            (ModelFamily.GPT, PromptMode.DEFAULT): self._format_gpt_prompt_default,
            (ModelFamily.GEMMA, PromptMode.DEFAULT): self._format_gemma_prompt_default,
            (ModelFamily.MISTRAL, PromptMode.DEFAULT): self._format_mistral_prompt_default,
        }
        return format_funcs.get((self.lm_wrapper.family, self.mode), self._format_default_prompt)

    def _format_llama_prompt_default(self, document: str, task: str) -> str:
        llama_prompt = "Context: {document}\n\nTask:{task}\n\nYou are trying to complete the task but do not have enough information from the document. Ask {n_clarifying_questions} question{plural} about the situation that can help you complete the task. In each question, only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Do not say anything other than the question. {json}"
        llama_user_message = llama_prompt.format(document=document, task=task, n_clarifying_questions=1, plural="s", json="")

        formatted_user_messages = [
            {
                "role": "system",
                "content": f"You are trying to help the user with the task below.",
            },
            {
                "role": "user",
                "content": llama_user_message,
            },
        ]
        
        return self.lm_wrapper.language_model._tokenizer.apply_chat_template(
            formatted_user_messages, tokenize=False, add_generation_prompt=True
        )

    def _format_gpt_prompt_default(self, document: str, task: str) -> str:
        benchmark_template = "Context: {document}\n\nTask:{task}\n\nYou are trying to complete the task but do not have enough information from the document. Ask {n_clarifying_questions} question{plural} about the situation that can help you complete the task. In each question, only ask for one fact at a time. If you can, reference something specific in the document. Do not merely rephrase the original task. Do not say anything other than the question. {json}"
        bechmark_template_json = 'Return the questions as a list in JSON format, as in {{"questions": ["The first question?", "The second question?"]}}'
        return benchmark_template.format(document=document, task=task, n_clarifying_questions=1, plural=False, json="")

    def _format_gemma_prompt_default(self, document: str, task: str) -> str:
        pass

    def _format_mistral_prompt_default(self, document: str, task: str) -> str:
        pass

    def _format_default_prompt(self, document: str, task: str) -> str:
        pass
        
    def forward_batch(self, documents: List[str], tasks: List[str]) -> List[List[str]]:
        format_func = self._get_format_func()
        formatted_instructions = [format_func(document, task) for document, task in zip(documents, tasks)]
        
        sequences = self.lm_wrapper.language_model.predict_many(
            [LmPrompt(p, cache=False, max_tokens=512) for p in formatted_instructions],
            completion_window=CompletionWindow.ASAP,
        )

        outputs = [loads(x.completion_text)["questions"] for x in sequences]
        return outputs

    def forward_single_generate_multiple_questions(self, document: str, task: str) -> List[str]:
        format_func = self._get_format_func()
        formatted_instruction = format_func(document, task)

        sequences = self.lm_wrapper.language_model.predict_many(
            ([LmPrompt(formatted_instruction, cache=False)]),
            completion_window=CompletionWindow.ASAP,
        )

        output = sequences[0].completion_text
        return loads(output)["questions"]
    
    def forward_batch_generate_single_question(self, documents: List[str], tasks: List[str]) -> List[str]:
        format_func = self._get_format_func()
        formatted_instructions = [format_func(document, task) for document, task in zip(documents, tasks)]
        
        # print(f"Prompt for Clariying Question Model:")
        # for p in formatted_instructions:
        #     print(p)
        # print("--"*20)

        sequences = self.lm_wrapper.language_model.predict_many(
            [LmPrompt(p, cache=False, max_tokens=512) for p in formatted_instructions],
            completion_window=CompletionWindow.ASAP,
        )

        outputs = [x.completion_text for x in sequences]
        
        return outputs

if __name__ == "__main__":
    ### testing

    # load data
    df = pd.read_json("results/unsafe/llama2-13b_200_summ-preference_unsafe.json").head(
        20
    )
    # load model
    model = GPTClarifyingQuestionModel(use_cache=True)
    # apply model
    df["cq"] = df.progress_apply(
        lambda x: model.forward(x["doc_summ"], x["prompt"]), axis=1
    )
    # save results
    df_to_md(df, "results/cq/cq_test.md")
    print
