from users import user_features
from users.user_features import PersonAttributeMeta
from .chatbot import *
from tqdm import tqdm
import re
import sys
import importlib.util
from utils import extract_function_definitions
import traceback
import inspect
from utils import hist_to_str, remove_raise_statements
from datamodels.template import SchemaError
import black
from enum import Enum
from pydantic import BaseModel
import json
from schema import And

import numpy as np


np.random.seed(0)
list_regex = r'(\["[^"]+"(?: *, *"[^"]+")+\])'


class Options(BaseModel):
    options: list[str]


class ConstraintType:
    choice = "choice"
    regex = "regex"
    types = "types"
    none = "none"


class ImaginaryData(dict):
    def get(self, key, default=None):
        raise KeyError


class CodeBot(ChatBot):
    gen_checker_prompt = """{attempt_no}\nEligibility Requirements:\n{eligibility_requirement}\n\nWrite a python function called `check_eligibility` that takes a dictionary `hh` containing relevant information and determines user eligibility. All keys and values of `hh` are strings. `check_eligibility` returns a bool. If you write helper functions, keep them inside the `check_eligibility` function. Make your code as detailed as possible capturing every edge case. Remember that the household may have no relevant members, so be sure to ask about the composition of the household. For example, for childcare programs, check that the household has at least one child. Here is an example:

def dummy_eligibility_program(hh: dict) -> bool:
    def _helper(hh):
        if hh["has_id"]=="yes":
            return True
        else:
            return False
    if hh["has_dependents"]=="yes":
        if int(hh["dependent_age"]) < 18 and _helper(hh):
            return True
    return False

DO NOT use `dict.get()` anywhere in the code. Key errors will be handled elsewhere. Do not use default values. Do not raise any exceptions. Do not provide anything besides code in your response."""

    get_type_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nTraget key:\n{key}\n\nQuestion: Given the code and context above, what do you expect {key} to be an integer, a float, or one choice from a set of strings? Return ONLY int, float, or choice."""
    get_choices_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nTraget key:\n{key}\n\nQuestion: Given the code and context above, what are the possible choices of {key}? Return ONLY the list of possible values."""
    get_values_prompt = """Context:{eligibility_requirements}\n\nCode:\n{code}\n\nTraget key:\n{key}\n\nQuestion: Given the code and context above, what are the possible values of {key}? Return ONLY the list of possible values in a list of strings. For example, return `["a", "b", "c"]`."""

    extract_value_from_ans_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to extract the value of {key} from the following dialog:\n\nQuestion: {cq}\n\nAnswer:\n{answer}\n\nWhat should we set as the value of {key}? Return ONLY the value."""
    key_error_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to determine what value of {key} should be stored in the `hh` dictionary. Ask a question to the user that would get this value. Return ONLY the question."""

    type_error_prompt = """Context:\n{eligibility_requirements}\n\nDialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, cannot be cast to type {target_type}. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""
    str_error_prompt = """Context:\n{eligibility_requirements}\n\nDialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, is not one of {target_options}. Which option of {target_options} is most similar to {value}? Return ONLY the value."""  # same as above but for when the target is a list of strings

    value_error_find_key_prompt = """Attempt: {attempt_no}\nCode:\n{code}\n\nLine:\n{line}\n\nTraceback: {traceback}\n\nWhat key is responsible for the error in the traceback? The possible keys are {key_options}. Return ONLY the key."""
    update_val_gen_prompt = """Attempt no. {attempt_no}\n\nCode:\n{code}\n\nError message:\n{error_message}.\n\nThe code above checks whether the type of the input is correct. Update the code so that all defaults pass, but nonsensical inputs of the wrong type fail. If there are no defaults in a dict.get() call, add a default value. Only check the type of the input, not the range. Do not check that all values are present. Ensure that an empty dictionary is valid. Return ONLY the code. """
    schema_error_prompt = """Attempt: {attempt_no}\nContext:\n{eligibility_requirements}\n\Dialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, does not fit the following criteria: `{target_type}`. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""

    generate_edge_cases_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nQuestion: Given the code and context above, what are the edge cases that will fail the code? Return ONLY the edge cases as a JSON in the form ["The first case", "The second case", ...]"""
    generate_corrected_code_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nThe given code fails for {failed_test_case}. Please correct it. Return ONLY the code."""
    make_unit_tests_prompt = """Context:\n{eligibility_requirements}\n\nCode:\n{code}\n\nQuestion: Given the code and context above, what are the test cases that will fail the code? Return ONLY five test cases as JSON list in the form:
    """
    example_unit_test = """[
    {
        "hh": {
            "key": "value"
        },
        "expected": "value"
    }
]
"""

    def __init__(
        self,
        # chat_model_id: str,
        # no_of_programs: str,
        # eligibility_requirements: str,
        # use_cache: bool,
        # lm_logger: Optional[LmLogger] = None,
        # code_model_id: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.used_keys = []
        self.key_types = {}
        self.choices = {}

    def pre_conversation(self, local_scope: dict):

        code = self.make_program(local_scope)


        # edge_cases_str = self.lm_api.forward(
        #     edge_case_prompt,
        #     chat_model_id=local_scope["args"].code_model_id,
        #     use_cache=local_scope["args"].use_cache,
        #     logging_role="code_gen",
        #     constraint_type="regex",
        #     constraints=list_regex,
        # )
        # print
        # generate unit tests

    def make_program(self, local_scope: dict):
        eligibility_requirements = local_scope["eligibility_requirements"]
        tf = local_scope["tf"]
        ### Write checker code
        generated_checker_text = {}
        generated_val_text = {}
        for name, desc in tqdm(eligibility_requirements.items()):
            
            failed_test_case = None
            failed_code = None

            checker_attempt_no = -1
            while True:
                checker_attempt_no += 1
                print(f"attempting to generate checker, attempt {checker_attempt_no}")
                
                if failed_test_case == None:
                    checker_gen_prompt = [
                        {
                            "role": "user",
                            "content": self.gen_checker_prompt.format(
                                attempt_no=checker_attempt_no, eligibility_requirement=desc
                            ),
                        }
                    ]

                    dirty_checker_output = self.lm_api.forward(
                        checker_gen_prompt,
                        chat_model_id=local_scope["args"].code_model_id,
                        use_cache=local_scope["args"].use_cache,
                        logging_role="code_gen",
                        # constraints=r"(?!.*\.get\().*", # prevent dict.get()
                        # constraint_type="regex",
                    ).strip("`")
                else:
                    checker_gen_prompt = [
                        {
                            "role": "user",
                            "content": self.generate_corrected_code_prompt.format(
                                eligibility_requirements=desc, code=failed_code, failed_test_case=failed_test_case
                            ),
                        }
                    ]

                    dirty_checker_output = self.lm_api.forward(
                        checker_gen_prompt,
                        chat_model_id=local_scope["args"].code_model_id,
                        use_cache=local_scope["args"].use_cache,
                        logging_role="code_gen",
                        # constraints=r"(?!.*\.get\().*", # prevent dict.get()
                        # constraint_type="regex",
                    ).strip("`")
                    failed_test_case = None
                    failed_code = None
                try:
                    clean_checker_output = extract_function_definitions(
                        dirty_checker_output
                    )["check_eligibility"]
                    clean_checker_output = clean_checker_output.replace(
                        "def check_eligibility", f"def {name}"
                    )
                    # replace all hh.get(...) with hh["..."]
                    clean_checker_output = re.sub(
                        r'hh\.get\(f?(["\'])(.*?)\1\)', r'hh["\2"]', clean_checker_output
                    )
                    clean_checker_output = black.format_str(
                        remove_raise_statements(clean_checker_output),
                        mode=black.FileMode(),
                    )
                    # check if the code is a valid python function
                    exec(clean_checker_output)
                    generated_checker_text[name] = clean_checker_output
                    
                    edge_case_prompt = [
                        {
                            "role": "user",
                            "content": self.make_unit_tests_prompt.format(
                                eligibility_requirements=local_scope["eligibility_requirements"],
                                code=clean_checker_output,
                            ) + self.example_unit_test,
                        }
                    ]

                    edge_case_output = self.lm_api.forward(
                        edge_case_prompt,
                        chat_model_id=local_scope["args"].code_model_id,
                        use_cache=local_scope["args"].use_cache,
                        logging_role="code_gen",
                    ).strip("`").strip("json\n")
                    
                    edge_case_outputs = json.loads(edge_case_output)

                    matches = True

                    for edge_case_output in edge_case_outputs:
                        hh = edge_case_output["hh"]
                        expected = edge_case_output["expected"]

                        result = locals()[name](hh)

                        if result != expected:
                            # New attempt
                            matches = False
                            failed_test_case = edge_case_output
                            failed_code = clean_checker_output
                            # print(failed_code, failed_test_case)
                            break
                    
                    if not matches:
                        continue



                    break
                except Exception as e:
                    pass

            assert clean_checker_output
            this_program_used_keys = re.findall(
                r"hh\[\"(.*?)\"\]", clean_checker_output
            )
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
                this_program_key_types[key] = self.lm_api.forward(
                    key_type_prompt,
                    chat_model_id=local_scope["args"].code_model_id,
                    use_cache=local_scope["args"].use_cache,
                    logging_role="type_gen",
                    constraints=["int", "float", "choice"],
                    constraint_type="choice",
                ).strip()
            # self.choices = {k: {} for k, v in self.key_types.items() if v == "choice"}
            # self.choices.update(
            new_choices = {
                k: {} for k, v in this_program_key_types.items() if v == "choice"
            }
            # )
            for c in new_choices:
                if local_scope["args"].code_model_id.startswith("gpt"):

                    openai_response_format = Options
                    response = self.lm_api.forward(
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
                        chat_model_id=local_scope["args"].code_model_id,
                        use_cache=local_scope["args"].use_cache,
                        logging_role="choice_gen",
                        openai_response_format=openai_response_format,
                    ).strip()

                    response_dict = json.loads(response)
                    choices = response_dict.get("options")
                else:
                    constraints = list_regex
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
                        chat_model_id=local_scope["args"].code_model_id,
                        use_cache=local_scope["args"].use_cache,
                        logging_role="choice_gen",
                        constraints=constraints,
                        constraint_type="regex",
                    ).strip()
                    choices = ast.literal_eval(choices)
                new_choices[c] = choices

            self.key_types.update(this_program_key_types)
            self.choices.update(new_choices)


            print

        with open("datamodels/template.py", "r") as template_file:
            template = template_file.read()

        eligibility_definition_block = "\n\n".join(generated_checker_text.values())
        eligibility_call_block = ",".join(
            [f"'{k}':{k}" for k in generated_checker_text.keys()]
        )
        val_definition_block = "\n\n".join(generated_val_text.values())
        val_call_block = ",".join(
            [f"'{k}':get_{k}_type_dict" for k in generated_val_text.keys()]
        )

        program = template.replace(
            "### ELIGIBILITY PROGRAMS PLACEHOLDER ###", eligibility_definition_block
        )
        program = program.replace("### CALLS PLACEHOLDER ###", eligibility_call_block)
        program = program.replace(
            "### ELIGIBILITY VALIDATORS PLACEHOLDER ###", val_definition_block
        )
        program = program.replace("### VALS PLACEHOLDER ###", val_call_block)

        tf.write(program)
        tf.flush()
        tf.close()

        sys.path.append(tf.name)
        return program

    def run_generated_code(self, locals):
        program_outputs = {}
        program_names = locals["program_names"]
        sorted_program_names = sorted(program_names, key=len, reverse=True)
        for program_name in sorted_program_names:
            spo = self.run_single_program(program_name, locals)  # single program output
            # drop history and hh
            del spo["history"]
            del spo["hh"]
            program_outputs[spo["program_name"]] = spo

        return program_outputs

    def run_single_program(self, program_name, locals):
        gen_code_path = locals["tf"].name
        synthetic_user = locals["synthetic_user"]
        # import the generated code from the temp file
        spec = importlib.util.spec_from_file_location("generated_code", gen_code_path)
        generated_code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_code)

        history = []
        locals["hh"] = ImaginaryData()
        # run the generated code until we get a valid output
        attempt_no = 0
        prev_hh = None
        while True:
            if locals["hh"] == prev_hh:
                print("returning None eligibility due to no change in locals")
                return {
                    "program_name": program_name,
                    "hh": locals["hh"],
                    "history": history,
                    "eligibility": None,
                    "completed": False,
                }

            ### Run the actual generated code ###
            try:
                p_fn = generated_code.calls[program_name]
                # eligibility = generated_code.calls[program_name](local_scope=locals)
                eligibility = p_fn(hh=locals["hh"])
                break
            except Exception as e:
                error_var = e
                # get the stack trace
                line = "\n".join(
                    traceback.format_exc().split("\n")[-3:-2]
                )  # TODO: check accuracy
                fn_name = program_name
                # fn_name = traceback.extract_tb(e.__traceback__)[
                #     1
                # ].name  # TODO: get without hard coding index
                assert fn_name in list(locals["eligibility_requirements"].keys())
                fn_code = inspect.getsource(eval("generated_code" + "." + fn_name))
                relevant_program = locals["eligibility_requirements"][
                    fn_name.lstrip("validate_")
                ]
                # relevant_val_dict = generated_code.vals[program_name]()

                # if there is a key error, ask a question to get the value
                if type(e) == KeyError:
                    error_var = e
                    key = error_var.args[0]
                    cq = self.forward_generic(
                        prompt=self.key_error_prompt.format(
                            eligibility_requirements=relevant_program,
                            line=line,
                            key=key,
                        ),
                        logging_role="key_error",
                    )
                    print(cq)
                    history.append({"role": "assistant", "content": cq})
                    ca = synthetic_user.answer_cq(cq)
                    print(ca)
                    history.append({"role": "user", "content": ca})
                    if key in self.key_types:
                        key_type = self.key_types[key]
                    else:
                        key_type = "any"
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
                        # not sure if anything else matters
                        raise NotImplementedError
                    # elif constraint_type == ConstraintType.types:
                    # constraint = [x.__name__ for x in constraint]
                    new_hh_value = self.forward_generic(
                        prompt=self.extract_value_from_ans_prompt.format(
                            eligibility_requirements=relevant_program,
                            line=line,
                            key=key,
                            cq=cq,
                            answer=ca,
                        ),
                        constraint_type=constraint_type,
                        constraints=constraint,
                        logging_role="extract_value_from_ans",
                    )
                    history.append({"role": "assistant", "content": new_hh_value})
                    prev_hh = deepcopy(locals["hh"])
                    locals["hh"][key] = new_hh_value
                    continue
                # elif type(e) == ValueError:
                #     error=e
                #     raise NotImplementedError
                    # key_options = set(locals["hh"].keys()).intersection(
                    #     set(relevant_val_dict.keys())
                    # )
                    # key = None
                    # attempt_no = 0
                    # while key not in relevant_val_dict.keys():
                    #     raw_key = self.forward_generic(
                    #         prompt=self.value_error_find_key_prompt.format(
                    #             attempt_no=attempt_no,
                    #             code=fn_code,
                    #             line=line,
                    #             traceback=str(e),
                    #             key_options=str(key_options),
                    #         ),
                    #         logging_role="value_error_find_key",
                    #     )
                    #     key = raw_key.strip("'\"` \n.")
                    #     attempt_no += 1
                    # # target_type = e.args[0]
                    # # print(e)
                    # target_type = relevant_val_dict[key]
                    # original_value = locals["hh"][key]
                    # # original_value = str(e)[str(e).find('"') + 1 :].strip('"')
                    # if type(target_type) == type:
                    #     prompt = self.type_error_prompt.format(
                    #         eligibility_requirements=relevant_program,
                    #         line=line,
                    #         value=original_value,
                    #         target_type=[target_type],
                    #         dialog=hist_to_str(history),
                    #     )
                    # else:  # type(target_type) == list[str]
                    #     prompt = self.str_error_prompt.format(
                    #         eligibility_requirements=relevant_program,
                    #         line=line,
                    #         value=original_value,
                    #         target_options=target_type,
                    #         dialog=hist_to_str(history),
                    #     )
                    # retyped_val = self.forward_generic(
                    #     prompt=prompt,
                    #     logging_role="type_error",
                    # )
                    # prev_hh = deepcopy(locals["hh"])
                    # locals["hh"][key] = retyped_val
                    # continue
                else:
                    print(
                        f"returning None because of error in generated code. Error: {error_var}"
                    )
                    return {
                        "program_name": program_name,
                        "hh": locals["hh"],
                        "history": history,
                        "eligibility": np.random.choice([True, False]),
                        "completed": False,
                    }
        return {
            "program_name": program_name,
            "hh": locals["hh"],
            "history": history,
            "eligibility": eligibility,
            "completed": True,
        }

    def forward_generic(
        self, prompt: str, logging_role: str, constraints=None, constraint_type="none"
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
        # if lm_output starts and ends with '"`, remove them
        for c in ["'", '"', "`"]:
            if lm_output.startswith(c) and lm_output.endswith(c):
                lm_output = lm_output.strip(c)

        return lm_output
