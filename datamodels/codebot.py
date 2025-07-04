from users import user_features
from users.user_features import PersonAttributeMeta
from .chatbot import *
from tqdm import tqdm
from server.model_client import ResponseFormat
import re
import sys
import importlib.util
import traceback
import json
import ast
import sys

import black
import numpy as np

from tqdm import tqdm
from pydantic import BaseModel
import json

from utils.utils import extract_function_definitions, remove_raise_statements, RoleEnum
from datamodels.chatbot import ChatBot
from utils.utils import hist_to_str
from typing import Optional
from copy import deepcopy

np.random.seed(0)
# list_regex = r'(\["[^"]+"(?: *, *"[^"]+")+\])'
list_regex = (
    r'\{\"options\": (\["[^"]+"(?: *, *"[^"]+")+\])\}'  # {"options": ["a", "b", "c"]}
)


def convert_keys_to_int(d):
    if not isinstance(d, dict):
        return d  # Return unchanged if not a dictionary

    new_dict = {}
    for key, value in d.items():
        new_key = int(key) if isinstance(key, str) and key.isdigit() else key
        new_dict[new_key] = (
            convert_keys_to_int(value) if isinstance(value, dict) else value
        )

    return new_dict


class ConstraintType:
    choice = "choice"
    regex = "regex"
    types = "types"
    none = "none"


class ImaginaryDataKeyError(Exception):
    pass


class ImaginaryData:
    def __init__(self, index=None):
        super().__init__()
        self.members = {}
        self.tl_data = {}  # top level data i.e. household data
        self.index = index  # index of the current member

    def get(self, key, default=None):
        return self._get(key)

    def _get(self, key):
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            int_key = int(key)
            if int_key in self.members:
                return self.members[int_key]
            else:
                if len(self.members) > 100:
                    print("infinite loop caused too many members")
                    raise NotImplementedError
                self.members[int_key] = ImaginaryData(index=int_key)
                return self.members[int_key]
        elif key in self.tl_data:
            return self.tl_data[key]
        else:
            raise ImaginaryDataKeyError(key, self.index)

    def __setitem__(self, key, value):
        self.tl_data[key] = value

    def __getitem__(self, key):
        return self._get(key)

    def __eq__(self, other):
        if not isinstance(other, ImaginaryData):
            return False
        return self.tl_data == other.tl_data and self.members == other.members

    def __contains__(self, item):
        return item in self.tl_data or item in self.members

    def __len__(self):
        return int(self.get("number of household members"))

    # def __len__(self):
    # raise KeyError("number of family members")


class CodeBot(ChatBot):
    gen_checker_prompt = """{attempt_no}\nEligibility Requirements:\n{eligibility_requirement}\n\nWrite a python function called `check_eligibility` that takes a dictionary `hh` containing relevant information and determines user eligibility. hh is a special dictionary connected to a language model that is conversing with the user. Any time it does not contain a key, it will determine that information from the user. As a result here are some requirements for interacting with `hh`:
    - DO NOT use `dict.get()` anywhere in the code. Key errors will be handled elsewhere.
    - Do not use default values.
    - Do not use any f-strings, curly brackets, or dynamically generated strings in your keys.
    - Use only literal strings in keys.
    - Do not use try-except blocks.
    - If you need to access data for individuals (rather than the household as a whole) you can use integer indexing. hh[0] is the head of the household. 
     
     `check_eligibility` returns a bool. All keys and values of `hh` are strings. If you write helper functions, keep them inside the `check_eligibility` function. Make your code as detailed as possible capturing every edge case. Remember that the household may have no relevant members, so be sure to ask about the composition of the household. For example, for childcare programs, check that the household has at least one child. After each new lookup in `hh`, write a comment suggesting a question to ask. Here is an example:

    def check_eligibility(hh: dict) -> bool:
        def _helper(individual):
            if individual["has_id"]=="yes": # "Does the individual have an ID?"
                return True
            else:
                return False
        if hh["has_dependents"]=="yes": # "How many dependents does the household have?"
            num_dependents = hh["num_dependents"]
            for i in range(len(num_dependents)):
                if float(hh[i]["dependent_age"]) < 18 and _helper(hh[i]): # "Is the first dependent under 18?"
                    return True
        return False

    The following is a set of preexisting keys and values in the `hh` dictionary; take care not to duplicate them.

    {preexisting_keys}

    Avoid using int() and use float() instead. Do not provide anything besides code in your response. Do not use `input` for user input.
    """

    get_type_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nTraget key:\n{key}\n\nQuestion: Given the code and context above, what do you expect {key} to be an integer, a float, or one choice from a set of strings? Return ONLY int, float, or choice."""
    # get_choices_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nTraget key:\n{key}\n\nQuestion: Given the code and context above, what are the possible choices of {key}? Return ONLY the list of possible values."""
    get_values_prompt = """Context:{eligibility_requirements}\n\nCode:\n{code}\n\nTraget key:\n{key}\n\nQuestion: Given the code and context above, what are the possible values of {key}? Return ONLY a JSON dict with they key "options" and the value being a list of possible values in a list of strings. For example, return `{{"options": ["a", "b", "c"]}}`."""

    extract_value_from_ans_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to extract the value of {key} from the following dialog:\n\nQuestion: {cq}\n\nAnswer:\n{answer}\n\nWhat should we set as the value of {key}? Return ONLY the value."""
    key_error_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to determine what value of {key} should be stored in the `hh` dictionary. Ask a question to the user that would get this value. For example, for age_i, ask "What is the age of person i?". Return ONLY the question."""
    response_not_found_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to determine what value of {key} should be stored in the `hh` dictionary. Ask a question to the user that would get this value. For example, for age_i, ask "What is the age of person i?". Return ONLY the question. The user responded "{answer}" to the previous question, which was"{cq}", so try to clarify the communication breakdown."""

    did_response_contain_answer_prompt = """Context:\nQuestion: {cq}\n\nResponse: {answer}\n\nDoes the response contain an answer to the question? Answer ONLY yes or no."""
    # type_error_prompt = """Context:\n{eligibility_requirements}\n\nDialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, cannot be cast to type {target_type}. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""
    # str_error_prompt = """Context:\n{eligibility_requirements}\n\nDialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, is not one of {target_options}. Which option of {target_options} is most similar to {value}? Return ONLY the value."""

    # value_error_find_key_prompt = """Attempt: {attempt_no}\nCode:\n{code}\n\nLine:\n{line}\n\nTraceback: {traceback}\n\nWhat key is responsible for the error in the traceback? The possible keys are {key_options}. Return ONLY the key."""
    # update_val_gen_prompt = """Attempt no. {attempt_no}\n\nCode:\n{code}\n\nError message:\n{error_message}.\n\nThe code above checks whether the type of the input is correct. Update the code so that all defaults pass, but nonsensical inputs of the wrong type fail. If there are no defaults in a dict.get() call, add a default value. Only check the type of the input, not the range. Do not check that all values are present. Ensure that an empty dictionary is valid. Return ONLY the code."""
    # schema_error_prompt = """Attempt: {attempt_no}\nContext:\n{eligibility_requirements}\n\Dialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, does not fit the following criteria: `{target_type}`. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""

    generate_edge_cases_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nQuestion: Given the code and context above, what are the edge cases that will fail the code? Return ONLY the edge cases as a JSON in the form ["The first case", "The second case", ...]"""
    generate_corrected_code_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nThe given code fails for {failed_test_case}. Please correct it. The output function name should strictly be "check_eligibility". Return ONLY the code."""
    make_unit_tests_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nQuestion: Given the code and context above, what are the test cases that will fail the code? DO NOT miss any JSON keys in the test cases. Return ONLY five test cases as JSON list in the form and DO NOT return an empty or null json:
    """
    example_unit_test = """[
    {
        "hh": [
            {
                "individual_level_key": "individual_level_value"
            },
            {
                "individual_level_key": "individual_level_value"
            }
        ],
        "expected": "value"
    }
]
"""

    def __init__(
        self,
        chat_model_id: str,
        no_of_programs: str,
        eligibility_requirements: str,
        use_cache: bool,
        random_seed: int,
        lm_logger=None,
        code_model_id: Optional[str] = None,
        max_code_gen_attempts: int = 1,
        max_code_rewrite_attempts: int = 0,
        data_user_index: int = 0,  # user data index used for tracking progress
    ):
        super().__init__(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_requirements,
            random_seed=random_seed,
            use_cache=use_cache,
            lm_logger=lm_logger,
        )
        self.code_model_id = code_model_id
        # self.used_keys = []
        # self.key_types = {name: {} for name, desc in eligibility_requirements.items()} # per-program key types
        self.key_types = {}  # merged key types
        # self.choices = {name: {} for name, desc in eligibility_requirements.items()} # per-program choices
        self.choices = {}  # merged choices
        self.max_code_gen_attempts = max_code_gen_attempts
        self.max_code_rewrite_attempts = max_code_rewrite_attempts
        self.total_questions = 0
        self.total_programs_completed = 0
        self.data_user_index = data_user_index

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
        # edge_case_prompt = [
        #     {
        #         "role": "user",
        #         "content": self.generate_edge_cases_prompt.format(
        #             eligibility_requirements=eligibility_requirements,
        #             code=code,
        #         ),
        #     }
        # ]
        # Optional code using edge_case_prompt would go here

    def _update_key_types_and_choices(
        self, input_keys, desc, clean_checker_output, code_model_id, use_cache
    ):
        # Identify keys used in the eligibility checker
        # this_program_used_keys = re.findall(r'hh\["(.*?)"\]', clean_checker_output)

        # drop hh to handle imaginary data not called hh
        input_keys = input_keys

        # Infer key types
        this_program_key_types = {}
        for key in input_keys:
            key_type_prompt = [
                {
                    "role": "user",
                    "content": self.get_type_prompt.format(
                        eligibility_requirements=desc,
                        code=clean_checker_output,
                        key=key,
                    ),
                }
            ]
            guessed_type = self.lm_api.forward(
                key_type_prompt,
                chat_model_id=code_model_id,
                use_cache=use_cache,
                logging_role="type_gen",
                constraints=["int", "float", "choice"],
                constraint_type="choice",
            ).strip()
            # find the last int/float/choice in case gpt fucks up
            guessed_type = re.findall(r"int|float|choice", guessed_type)[-1]
            assert guessed_type in ["int", "float", "choice"]
            # replace all '$' with '\$` as long as the $ is not already escaped
            # guessed_type = re.sub(r"[^\\](\$)")
            this_program_key_types[key] = guessed_type

        # Determine possible choices
        new_choices = {
            k: {} for k, v in this_program_key_types.items() if v == "choice"
        }
        for c in new_choices:
            if code_model_id and code_model_id.startswith("gpt"):
                response_dict_raw = self.lm_api.forward(
                    [
                        {
                            "role": "user",
                            "content": self.get_values_prompt.format(
                                eligibility_requirements=desc,
                                code=clean_checker_output,
                                key=c,
                            ),
                        }
                    ],
                    chat_model_id=code_model_id,
                    use_cache=use_cache,
                    logging_role="choice_gen",
                    # openai_response_format=ResponseFormat.options.value,
                    openai_response_format={"type": "json_object"},
                )
                response_dict = json.loads(response_dict_raw)
                choices = response_dict.get("options", [])
            elif code_model_id and code_model_id.startswith("claude"):

                def _is_valid(choices):
                    if choices is None:
                        return False
                    if len(choices) == 0:
                        return False
                    for c in choices:
                        if not isinstance(c, str):
                            return False
                    return True

                choices = None
                z = 0
                while not _is_valid(choices):
                    if z > 10:
                        print("too many attempts")
                        sys.exit(1)
                    print(f"attempt no {z}")
                    response_dict_raw = self.lm_api.forward(
                        [
                            {
                                "role": "user",
                                "content": f"attempt no {z}\n\n"
                                + self.get_values_prompt.format(
                                    eligibility_requirements=desc,
                                    code=clean_checker_output,
                                    key=c,
                                ),
                            }
                        ],
                        chat_model_id=code_model_id,
                        use_cache=use_cache,
                        logging_role="choice_gen",
                        claude_tool_def=[
                            {
                                "name": "define_values",
                                "description": "Define the possible values for a key",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {
                                        "options": {
                                            "type": "array",
                                            "description": "The possible values for the key",
                                            "items": {
                                                "type": "string",
                                            },
                                        },
                                    },
                                    "required": ["options"],
                                },
                            }
                        ],
                    )
                    # Handle Claude tool call response more robustly
                    response_dict = json.loads(response_dict_raw)
                    choices = response_dict.get("options", [])
                    z += 1

            else:
                choices = self.lm_api.forward(
                    [
                        {
                            "role": "user",
                            "content": self.get_values_prompt.format(
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
            # for i, c in enumerate(choices):  # escape all '$'
            #     choices[i] = re.sub(r"(?<!\\)\$", r"\\$", c)

            new_choices[c] = [x.strip("\"'") for x in choices]
        # do the $ escape substitution for all choices
        for k, cs in new_choices.items():
            for i, c in enumerate(cs):
                new_choices[k][i] = re.sub(r"(?<!\\)\$", r"\\$", c)
        return this_program_key_types, new_choices

    def make_program(
        self,
        code_file_handle,
        eligibility_requirements,
        code_model_id,
        use_cache,
    ):
        self.clean_checker_outputs = {}
        generated_checker_text = {}
        generated_val_text = {}

        for name, desc in tqdm(eligibility_requirements.items()):
            failed_test_case = None
            failed_code = None
            error_trace = None

            checker_attempt_no = 0
            code_rewrite_attempt_no = 0
            rewritten = False

            self.max_code_gen_attempts = self.max_code_gen_attempts
            while (
                checker_attempt_no < self.max_code_gen_attempts
                or code_rewrite_attempt_no < self.max_code_rewrite_attempts
            ):
                oai_seed_no = checker_attempt_no + 1000 * self.random_seed
                print(f"attempting to generate checker, attempt {oai_seed_no}")

                if failed_test_case is None:
                    prompt_content = self.gen_checker_prompt.format(
                        attempt_no=oai_seed_no,
                        eligibility_requirement=desc,
                        preexisting_keys=self.get_pek_str(),
                    )
                    rewritten = False
                else:

                    prompt_content = self.generate_corrected_code_prompt.format(
                        eligibility_requirements=desc,
                        code=failed_code,
                        failed_test_case=failed_test_case,
                    )

                    if error_trace:
                        prompt_content += f"\nThrows an error:\n {error_trace}"
                    else:
                        prompt_content += "Throws no errors but generates wrong output."

                    failed_test_case = None
                    failed_code = None
                    error_trace = None
                    rewritten = True

                dirty_checker_output = self.lm_api.forward(
                    [{"role": "user", "content": prompt_content}],
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

                    exec(func_def)  # test if it runs
                    self.clean_checker_outputs[name] = func_def
                    generated_checker_text[name] = func_def

                    this_program_used_keys = re.findall(
                        r'\["(.*?)"\]', self.clean_checker_outputs[name]
                    )

                except Exception as e:
                    import traceback

                    traceback.print_exc()

                    if not rewritten:
                        checker_attempt_no += 1

                    # If we exceeded attempts, skip this requirement
                    if (
                        not rewritten
                        and checker_attempt_no > self.max_code_gen_attempts
                    ):
                        raise Exception(
                            f"Failed to generate checker for {name} after {checker_attempt_no} attempts."
                        )

                try:
                    if code_rewrite_attempt_no < self.max_code_rewrite_attempts:
                        code_rewrite_attempt_no += 1
                        edge_case_prompt = [
                            {
                                "role": "user",
                                "content": self.make_unit_tests_prompt.format(
                                    eligibility_requirements=eligibility_requirements,
                                    code=func_def,
                                )
                                + self.example_unit_test,
                            }
                        ]

                        edge_case_output = (
                            self.lm_api.forward(
                                edge_case_prompt,
                                chat_model_id=code_model_id,
                                use_cache=use_cache,
                                logging_role="code_gen",
                            )
                            .strip("`")
                            .strip("json\n")
                        )

                        edge_case_outputs = json.loads(edge_case_output)
                        matches = True

                        for case in edge_case_outputs:
                            hh = convert_keys_to_int(case["hh"])
                            expected = case["expected"]
                            try:
                                result = locals()[name](hh)
                            except Exception as err:
                                import traceback

                                error_trace = "".join(
                                    traceback.format_exception(*sys.exc_info())
                                )
                                failed_code = func_def
                                failed_test_case = case
                                matches = False

                                break

                            if result != expected:
                                matches = False
                                failed_test_case = case
                                failed_code = func_def
                                break

                        if not matches:
                            continue

                    # success: neither code gen nor code rewrite need to run again
                    (
                        this_program_key_types,
                        new_choices,
                    ) = self._update_key_types_and_choices(
                        this_program_used_keys,
                        desc,
                        self.clean_checker_outputs[name],
                        code_model_id,
                        use_cache,
                    )
                    self.key_types.update(this_program_key_types)
                    self.choices.update(new_choices)
                    break
                except Exception as e:
                    import traceback

                    traceback.print_exc()

                    continue

        with open("datamodels/template.py", "r") as template_file:
            template = template_file.read()

        eligibility_definition_block = "\n\n".join(generated_checker_text.values())
        eligibility_call_block = ",".join(
            [f"'{k}':{k}" for k in generated_checker_text.keys()]
        )
        val_definition_block = "\n\n".join(generated_val_text.values())
        val_call_block = ",".join([])  # no validators in this example

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
        hh = ImaginaryData()
        for program_name in sorted_program_names:

            spo, hh = self.run_single_program(
                hh=hh,
                program_name=program_name,
                code_file_path=code_file_path,
                synthetic_user=synthetic_user,
                eligibility_requirements=eligibility_requirements,
            )
            del spo["history"]
            del spo["hh"]
            program_outputs[spo["program_name"]] = spo
            self.total_programs_completed += 1
        return program_outputs

    def run_single_program(
        self,
        hh,
        program_name,
        code_file_path,
        synthetic_user,
        eligibility_requirements,
    ):
        spec = importlib.util.spec_from_file_location("generated_code", code_file_path)
        generated_code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_code)

        history = []
        # hh = ImaginaryData()
        attempt_no = 0
        prev_hh = None
        this_program_questions = 0

        while True:
            if hh == prev_hh:

                print(f"warning: no progress for {program_name}")
                return {
                    "program_name": program_name,
                    "hh": hh,
                    "history": history,
                    "eligibility": np.random.choice([True, False]),
                    "completed": False,
                }, hh

            try:
                p_fn = generated_code.calls[program_name]
                eligibility = p_fn(hh=hh)
                break
            except ImaginaryDataKeyError as e:
                error_var = e
                fn_name = program_name
                assert fn_name in eligibility_requirements
                relevant_program = eligibility_requirements[fn_name]

                # if isinstance(e, ImaginaryDataKeyError):
                key = error_var.args[0]
                member_idx = error_var.args[1]
                # line = [x for x in traceback.format_exc().split("\n") if key in x][0]
                line = self.find_line(
                    traceback.extract_tb(error_var.__traceback__),
                    key,
                    generated_code.__file__,
                )
                if line is None:
                    # sometimes the key is not a literal string so it's not caught in the first pass
                    new_key_types, new_choices = self._update_key_types_and_choices(
                        [key],
                        eligibility_requirements[program_name],
                        self.clean_checker_outputs[program_name],
                        self.code_model_id,
                        self.use_cache,
                    )
                    self.key_types.update(new_key_types)
                    # self.choices.update(new_choices)
                    self.update_choices(new_choices)
                    line = "\n".join(
                        [
                            x.line
                            for x in list(traceback.extract_tb(error_var.__traceback__))
                        ]
                    )
                    assert line

                cq = self.forward_generic(
                    prompt=self.key_error_prompt.format(
                        eligibility_requirements=relevant_program,
                        line=line,
                        key=key,
                    ),
                    logging_role="key_error",
                )
                for clarification_attempt_no in range(3):
                    # check that the cq doesn't appear in the history more than 12 times
                    past_cqs = np.array([x["content"] for x in history])
                    if np.sum(past_cqs == cq) > 12:
                        # print("too many cq repeats")
                        return {
                            "program_name": program_name,
                            "hh": hh,
                            "history": history,
                            "eligibility": np.random.choice([True, False]),
                            "completed": False,
                        }, hh

                    this_program_questions += 1
                    self.total_questions += 1
                    print(f"user index: {self.data_user_index}")
                    print(
                        f"{this_program_questions} questions on program {program_name}"
                    )
                    print(
                        f"{self.total_questions} total questions on {self.total_programs_completed} programs"
                    )
                    history.append({"role": RoleEnum.CQ_MODEL.value, "content": cq})
                    ca = synthetic_user.answer_cq(cq=cq, history=history)
                    history.append(
                        {"role": RoleEnum.SYNTHETIC_USER.value, "content": ca}
                    )

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
                        print(f"unknown key_type: {key_type}")
                        raise NotImplementedError
                    # try:
                    # need a loop here to check if answer is solid
                    mini_history = deepcopy(history)
                    question_was_answered = self.forward_generic(
                        prompt=self.did_response_contain_answer_prompt.format(
                            cq=cq,
                            answer=ca,
                        ),
                        logging_role="did_response_contain_answer",
                        constraint_type=ConstraintType.choice,
                        constraints=["yes", "no"],
                    )
                    if "yes" in question_was_answered.lower():
                        break
                    else:
                        cq = self.forward_generic(
                            prompt=self.response_not_found_prompt.format(
                                cq=cq,
                                answer=ca,
                                eligibility_requirements=relevant_program,
                                line=line,
                                key=key,
                            ),
                            logging_role="response_not_found",
                        )
                    # except Exception as e:
                    #     print(f"failed to extract value from answer: {e}")
                    #     new_hh_value = "0"
                try:
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
                except Exception as e:
                    print(
                        f"failed to extract value from answer: {e}"
                    )  # not sure whats wrong, fix later, @nikhil?
                    new_hh_value = "0"
                # history.append({"role": "assistant", "content": new_hh_value})
                prev_hh = deepcopy(hh)
                if member_idx is None:
                    hh[key] = new_hh_value
                else:
                    hh[member_idx][key] = new_hh_value
                continue
                # else:
                # except Exception as e:
            except Exception as e:
                print(e)
                return {
                    "program_name": program_name,
                    "hh": hh,
                    "history": history,
                    "eligibility": np.random.choice([True, False]),
                    "completed": False,
                }, hh

        return {
            "program_name": program_name,
            "hh": hh,
            "history": history,
            "eligibility": eligibility,
            "completed": True,
        }, hh

    def find_line(self, fe, key, filename):
        for frame in fe[::-1]:
            if frame.filename == filename:
                if key in frame.line:
                    return frame.line
        return None
        # raise Exception(f"Key {key} not found in traceback")

    def forward_generic(
        self,
        prompt: str,
        logging_role: str,
        constraints=None,
        constraint_type="none",
    ):
        prompt = [
            {
                "role": "user",
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

    def update_choices(self, new_choices: dict):
        for k, v in new_choices.items():
            if k in self.choices:
                new = [c for c in v if c not in self.choices[k]]
                if new:
                    print(
                        f"updating choices for {k}.\noriginal: {self.choices[k]}\nnew: {new}"
                    )
                    self.choices[k].extend(new)
            else:
                self.choices[k] = v

    def get_pek_str(self):
        """Preexisting keys"""
        pek = {}
        for k, v in self.key_types.items():
            val = self.choices.get(k, str(v))
            pek[k] = val
        return "\n".join([f"{k}: {v}" for k, v in pek.items()])
