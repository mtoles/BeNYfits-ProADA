import re
import sys
import importlib.util
import traceback
import json
import ast

import black
import numpy as np

from tqdm import tqdm
from enum import Enum
from pydantic import BaseModel

from utils import extract_function_definitions, remove_raise_statements
from datamodels.chatbot import ChatBot
from utils import hist_to_str
from typing import Optional
from copy import deepcopy

np.random.seed(0)
list_regex = r'(\["[^"]+"(?: *, *"[^"]+")+\])'


class Options(BaseModel):
    options: list[str]


class ConstraintType:
    choice = "choice"
    regex = "regex"
    types = "types"
    none = "none"


class ImaginaryData:
    def __init__(self, index=None):
        super().__init__()
        self.members = {}
        self.tl_data = {}  # top level data i.e. household data
        self.index = index  # index of the current member

    def get(self, key, default=None):
        return self._get(key)

    def _get(self, key):
        if isinstance(key, int):
            if key in self.members:
                return self.members[key]
            else:
                self.members[key] = ImaginaryData(key, index=key)
                return self.members[key]
        elif key in self.tl_data:
            return self.tl_data[key]
        else:
            raise KeyError(key, self.index)

    def __setitem__(self, key, value):
        self.tl_data[key] = value

    def __getitem__(self, key):
        return self._get(key)

    def __eq__(self, other):
        if not isinstance(other, ImaginaryData):
            return False
        return self.tl_data == other.tl_data and self.members == other.members

    # def __len__(self):
    # raise KeyError("nuber of family members")


class CodeBot(ChatBot):
    gen_checker_prompt = """{attempt_no}\nEligibility Requirements:\n{eligibility_requirement}\n\nWrite a python function called `check_eligibility` that takes a dictionary `hh` containing relevant information and determines user eligibility. hh is a special dictionary connected to a language model that is conversing with the user. Any time it does not contain a key, it will determine that information from the user. As a result here are some requirements for interacting with `hh`:
    - DO NOT use `dict.get()` anywhere in the code. Key errors will be handled elsewhere.
    - Do not use default values.
    - Do not use any f-strings, curly brackets, or dynamically generated strings in your keys.
    - Use only literal strings in keys.
    - Do not use try-except blocks.
    - If you need to access data for individuals (rather than the household as a whole) you can use integer indexing. hh[0] is the head of the household. 
     
     `check_eligibility` returns a bool. All keys and values of `hh` are strings. If you write helper functions, keep them inside the `check_eligibility` function. Make your code as detailed as possible capturing every edge case. Remember that the household may have no relevant members, so be sure to ask about the composition of the household. For example, for childcare programs, check that the household has at least one child. Here is an example:

    def dummy_eligibility_program(hh: dict) -> bool:
        def _helper(individual):
            if individual["has_id"]=="yes":
                return True
            else:
                return False
        if hh["has_dependents"]=="yes":
            num_dependents = hh["num_dependents"]
            for i in range(len(num_dependents)):
            
                if float(hh[i]["dependent_age"]) < 18 and _helper(hh[i]):
                    return True
        return False

    Avoid using int() and use float() instead. Do not provide anything besides code in your response.
    """

    get_type_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nTraget key:\n{key}\n\nQuestion: Given the code and context above, what do you expect {key} to be an integer, a float, or one choice from a set of strings? Return ONLY int, float, or choice."""
    get_choices_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nTraget key:\n{key}\n\nQuestion: Given the code and context above, what are the possible choices of {key}? Return ONLY the list of possible values."""
    get_values_prompt = """Context:{eligibility_requirements}\n\nCode:\n{code}\n\nTraget key:\n{key}\n\nQuestion: Given the code and context above, what are the possible values of {key}? Return ONLY the list of possible values in a list of strings. For example, return `["a", "b", "c"]`."""

    extract_value_from_ans_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to extract the value of {key} from the following dialog:\n\nQuestion: {cq}\n\nAnswer:\n{answer}\n\nWhat should we set as the value of {key}? Return ONLY the value."""
    key_error_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to determine what value of {key} should be stored in the `hh` dictionary. Ask a question to the user that would get this value. Return ONLY the question."""

    type_error_prompt = """Context:\n{eligibility_requirements}\n\nDialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, cannot be cast to type {target_type}. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""
    str_error_prompt = """Context:\n{eligibility_requirements}\n\nDialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, is not one of {target_options}. Which option of {target_options} is most similar to {value}? Return ONLY the value."""

    value_error_find_key_prompt = """Attempt: {attempt_no}\nCode:\n{code}\n\nLine:\n{line}\n\nTraceback: {traceback}\n\nWhat key is responsible for the error in the traceback? The possible keys are {key_options}. Return ONLY the key."""
    update_val_gen_prompt = """Attempt no. {attempt_no}\n\nCode:\n{code}\n\nError message:\n{error_message}.\n\nThe code above checks whether the type of the input is correct. Update the code so that all defaults pass, but nonsensical inputs of the wrong type fail. If there are no defaults in a dict.get() call, add a default value. Only check the type of the input, not the range. Do not check that all values are present. Ensure that an empty dictionary is valid. Return ONLY the code."""
    schema_error_prompt = """Attempt: {attempt_no}\nContext:\n{eligibility_requirements}\n\Dialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, does not fit the following criteria: `{target_type}`. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""

    generate_edge_cases_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nQuestion: Given the code and context above, what are the edge cases that will fail the code? Return ONLY the edge cases as a JSON in the form ["The first case", "The second case", ...]"""
    make_unit_tests_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nQuestion: Given the code and context above, what is the test case that will fail the code? Return ONLY the test case as JSON in the form:
{
    "hh": {
        "key": "value"
    },
    "expected": "value"
}
    """

    def __init__(
        self,
        chat_model_id: str,
        no_of_programs: str,
        eligibility_requirements: str,
        use_cache: bool,
        lm_logger=None,
        code_model_id: Optional[str] = None,
    ):
        super().__init__(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_requirements,
            use_cache=use_cache,
            lm_logger=lm_logger,
        )
        self.code_model_id = code_model_id
        self.used_keys = []
        self.key_types = {}
        self.choices = {}

    def pre_conversation(
        self,
        code_file_handle,
        eligibility_requirements,
        code_model_id,
        use_cache,
    ):
        code = self.make_program(
            code_file_handle=code_file_handle,
            eligibility_requirements=eligibility_requirements,
            code_model_id=code_model_id,
            use_cache=use_cache,
        )
        edge_case_prompt = [
            {
                "role": "user",
                "content": self.generate_edge_cases_prompt.format(
                    eligibility_requirements=eligibility_requirements,
                    code=code,
                ),
            }
        ]
        # Optional code using edge_case_prompt would go here

    def make_program(
        self,
        code_file_handle,
        eligibility_requirements,
        code_model_id,
        use_cache,
    ):
        generated_checker_text = {}
        generated_val_text = {}

        for name, desc in tqdm(eligibility_requirements.items()):
            checker_attempt_no = -1
            clean_checker_output = ""
            while True:
                checker_attempt_no += 1
                checker_gen_prompt = [
                    {
                        "role": "user",
                        "content": self.gen_checker_prompt.format(
                            attempt_no=checker_attempt_no,
                            eligibility_requirement=desc,
                        ),
                    }
                ]

                dirty_checker_output = self.lm_api.forward(
                    checker_gen_prompt,
                    chat_model_id=code_model_id,
                    use_cache=use_cache,
                    logging_role="code_gen",
                ).strip("`")
                try:
                    extracted = extract_function_definitions(dirty_checker_output)
                    func_def = extracted["check_eligibility"]
                    func_def = func_def.replace("def check_eligibility", f"def {name}")
                    func_def = re.sub(
                        r'hh\.get\(f?(["\'])(.*?)\1\)',
                        r'hh["\2"]',
                        func_def,
                    )
                    func_def = black.format_str(
                        remove_raise_statements(func_def),
                        mode=black.FileMode(),
                    )
                    exec(func_def)
                    clean_checker_output = func_def
                    generated_checker_text[name] = clean_checker_output
                    break
                except Exception:
                    pass

            this_program_used_keys = re.findall(r'hh\["(.*?)"\]', clean_checker_output)
            self.used_keys.extend(this_program_used_keys)
            this_program_key_types = {}

            for key in this_program_used_keys:
                key_type_prompt = [
                    {
                        "role": "user",
                        "content": self.get_type_prompt.format(
                            eligibility_requirements=desc,
                            code=clean_checker_output,
                            key=key,
                        ),
                    },
                ]
                guessed_type = self.lm_api.forward(
                    key_type_prompt,
                    chat_model_id=code_model_id,
                    use_cache=use_cache,
                    logging_role="type_gen",
                    constraints=["int", "float", "choice"],
                    constraint_type="choice",
                ).strip()
                this_program_key_types[key] = guessed_type

            new_choices = {
                k: {} for k, v in this_program_key_types.items() if v == "choice"
            }

            for c in new_choices:
                if code_model_id and code_model_id.startswith("gpt"):
                    response_dict_raw = self.lm_api.forward(
                        [
                            {
                                "role": "user",
                                "content": self.get_choices_prompt.format(
                                    eligibility_requirements=desc,
                                    code=clean_checker_output,
                                    key=c,
                                ),
                            }
                        ],
                        chat_model_id=code_model_id,
                        use_cache=use_cache,
                        logging_role="choice_gen",
                        openai_response_format=Options,
                    )
                    response_dict = json.loads(response_dict_raw)
                    choices = response_dict.get("options", [])
                else:
                    choices = self.lm_api.forward(
                        [
                            {
                                "role": "user",
                                "content": self.get_choices_prompt.format(
                                    eligibility_requirements=desc,
                                    code=clean_checker_output,
                                    key=c,
                                ),
                            }
                        ],
                        chat_model_id=code_model_id,
                        use_cache=use_cache,
                        logging_role="choice_gen",
                        constraints=list_regex,
                        constraint_type="regex",
                    ).strip()
                    choices = ast.literal_eval(choices)
                new_choices[c] = choices

            self.key_types.update(this_program_key_types)
            self.choices.update(new_choices)

        with open("datamodels/template.py", "r") as template_file:
            template = template_file.read()

        eligibility_definition_block = "\n\n".join(generated_checker_text.values())
        eligibility_call_block = ",".join(
            [f"'{k}':{k}" for k in generated_checker_text.keys()]
        )
        val_definition_block = "\n\n".join(generated_val_text.values())
        val_call_block = ",".join([])  # no actual code validators in this example

        program = template.replace(
            "### ELIGIBILITY PROGRAMS PLACEHOLDER ###", eligibility_definition_block
        )
        program = program.replace("### CALLS PLACEHOLDER ###", eligibility_call_block)
        program = program.replace(
            "### ELIGIBILITY VALIDATORS PLACEHOLDER ###", val_definition_block
        )
        program = program.replace("### VALS PLACEHOLDER ###", val_call_block)

        code_file_handle.write(program)
        code_file_handle.flush()
        code_file_handle.close()

        sys.path.append(code_file_handle.name)
        return program

    def run_generated_code(
        self,
        code_file_path,
        synthetic_user,
        eligibility_requirements,
        program_names,
    ):
        program_outputs = {}
        sorted_program_names = sorted(program_names, key=len, reverse=True)
        for program_name in sorted_program_names:
            spo = self.run_single_program(
                program_name=program_name,
                code_file_path=code_file_path,
                synthetic_user=synthetic_user,
                eligibility_requirements=eligibility_requirements,
            )
            del spo["history"]
            del spo["hh"]
            program_outputs[spo["program_name"]] = spo
        return program_outputs

    def run_single_program(
        self,
        program_name,
        code_file_path,
        synthetic_user,
        eligibility_requirements,
    ):
        spec = importlib.util.spec_from_file_location("generated_code", code_file_path)
        generated_code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_code)

        history = []
        hh = ImaginaryData()
        attempt_no = 0
        prev_hh = None

        while True:
            if hh == prev_hh:
                # return {
                #     "program_name": program_name,
                #     "hh": hh,
                #     "history": history,
                #     "eligibility": None,
                #     "completed": False,
                # }
                print(f"warning: no progress for {program_name}")

            try:
                p_fn = generated_code.calls[program_name]
                eligibility = p_fn(hh=hh)
                break
            except Exception as e:
                error_var = e
                fn_name = program_name
                if fn_name not in eligibility_requirements.keys():
                    print("why is this here? should never happen")
                    raise NotImplementedError
                    return {
                        "program_name": program_name,
                        "hh": hh,
                        "history": history,
                        "eligibility": None,
                        "completed": False,
                    }
                relevant_program = eligibility_requirements[fn_name]

                if isinstance(e, KeyError):
                    key = error_var.args[0]
                    member_idx = error_var.args[1]
                    # line = [x for x in traceback.format_exc().split("\n") if key in x][0]
                    line = self.find_line(
                        traceback.extract_tb(error_var.__traceback__),
                        key,
                        generated_code.__file__,
                    )
                    cq = self.forward_generic(
                        prompt=self.key_error_prompt.format(
                            eligibility_requirements=relevant_program,
                            line=line,
                            key=key,
                        ),
                        logging_role="key_error",
                    )
                    history.append({"role": "assistant", "content": cq})
                    ca = synthetic_user.answer_cq(cq)
                    history.append({"role": "user", "content": ca})

                    key_type = self.key_types.get(key, "any")
                    if key_type == ConstraintType.choice:
                        constraint_type = ConstraintType.choice
                        constraint = self.choices[key]
                    elif key_type in ["int", "float"]:
                        constraint_type = ConstraintType.types
                        constraint = [eval(key_type)]
                    elif key_type == "any":
                        constraint_type = ConstraintType.none
                        constraint = None
                    else:
                        raise NotImplementedError

                    new_hh_value = self.forward_generic(
                        prompt=self.extract_value_from_ans_prompt.format(
                            eligibility_requirements=relevant_program,
                            line=line,
                            key=key,
                            cq=cq,
                            answer=ca,
                        ),
                        logging_role="extract_value_from_ans",
                        constraint_type=constraint_type,
                        constraints=constraint,
                    )
                    history.append({"role": "assistant", "content": new_hh_value})
                    prev_hh = deepcopy(hh)
                    if member_idx is None:
                        hh[key] = new_hh_value
                    else:
                        hh[member_idx][key] = new_hh_value
                    continue
                else:
                    return {
                        "program_name": program_name,
                        "hh": hh,
                        "history": history,
                        "eligibility": np.random.choice([True, False]),
                        "completed": False,
                    }

        return {
            "program_name": program_name,
            "hh": hh,
            "history": history,
            "eligibility": eligibility,
            "completed": True,
        }

    def find_line(self, fe, key, filename):
        for frame in fe[::-1]:
            if frame.filename == filename:
                if key in frame.line:
                    return frame.line
        raise Exception(f"Key {key} not found in traceback")

    def forward_generic(
        self,
        prompt: str,
        logging_role: str,
        constraints=None,
        constraint_type="none",
    ):
        prompt = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        lm_output = self.lm_api.forward(
            prompt,
            chat_model_id=self.chat_model_id,
            use_cache=self.use_cache,
            constraint_type=constraint_type,
            constraints=constraints,
            logging_role=logging_role,
        )
        for c in ["'", '"', "`"]:
            if lm_output.startswith(c) and lm_output.endswith(c):
                lm_output = lm_output.strip(c)
        return lm_output
