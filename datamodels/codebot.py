from .chatbot import *
from tqdm import tqdm
from tempfile import NamedTemporaryFile
import re
import sys
import importlib.util
from utils import extract_function_definitions
import traceback
import inspect
from utils import hist_to_str, remove_raise_statements
from datamodels.template import SchemaError


class CodeBot(ChatBot):
    gen_checker_prompt = """Eligibility Requirements:\n{eligibility_requirement}\n\nWrite a python function called `check_eligibility` that takes a dictionary `hh` containing relevant information in string form and determines user eligibility. `check_eligibility` returns a bool. If you write helper functions, keep them inside the `check_eligibility` function. Make your code as detailed as possible capturing every edge case. Here is an example:

def dummy_eligibility_program(hh: dict) -> bool:
    def _helper(hh):
        if hh["has_id"]:
            return True
    if hh["age"] < 18 and _helper(hh):
        return True
    return False

DO NOT use `dict.get()` anywhere in the code. Key errors will be handled elsewhere. Do not use default values. Do not raise any exceptions. Do not provide anything besides code in your response."""

    gen_validator_prompt = """Eligibility Requirements:\n{eligibility_requirement}\n\ncode:\n{code}\n\nWrite a python function called `validate_{name}` that takes a dictionary `hh` containing relevant information and outputs whether all values in `hh` are of correct type. A value is of correct type if it will cause no errors in the code above, EVEN IF the code above would return `False`. Keys and values of `hh` are strings. For each key that appears in the code, raise a ValueError exception if the value is not valid. If all checks pass, return None. For example:

def eligibility_program(hh: dict) -> bool:
    if hh["age"] =< 18 and hh["filing_status"] == "single":
        return True
    return False

def validate_user_data(hh: dict):
    # note that we DO NOT check whether all keys are present
    try: 
        _ = int(hh.get("age", 0)) # note that we DO NOT check the range of age
    except ValueError:
        raise ValueError("Age must be an integer.")
    try:
        hh.get("filing_status", "single") in ["single", "married"]:
    except:
        raise ValueError("Filing status must be 'single' or 'married'.")
    return None

Always use `dict.get()` to access values in `hh`. Make sure the default value of `.get()` is valid. Do not check that all values are present. Ensure that an empty dictionary is valid. Do not provide anything besides code in your response."""

    extract_value_from_ans_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to extract the value of {key} from the following dialog:\n\nQuestion: {cq}\n\nAnswer:\n{answer}\n\nWhat should we set as the value of {key}? Return ONLY the value."""
    key_error_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to determine what value of {key} should be stored in the `hh` dictionary. Ask a question to the user that would get this value. Return ONLY the question."""
    type_error_prompt = """Context:\n{eligibility_requirements}\n\nDialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, cannot be cast to type {target_type}. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""
    value_error_find_key_prompt = """Code:\n{code}\n\nLine:\n{line}\n\nTraceback: {traceback}\n\nWhat key is responsible for the error in the traceback? Return ONLY the key."""
    update_val_gen_prompt = """Attempt no. {attempt_no}\n\nCode:\n{code}\n\nError message:\n{error_message}.\n\nThe code above checks whether the type of the input is correct. Update the code so that all defaults pass, but nonsensical inputs of the wrong type fail. If there are no defaults in a dict.get() call, add a default value. Only check the type of the input, not the range. Do not check that all values are present. Ensure that an empty dictionary is valid. Return ONLY the code. """

    def pre_conversation(self, local_scope: dict):
        eligibility_requirements = local_scope["eligibility_requirements"]
        tf = local_scope["tf"]
        ### Write checker code
        generated_checker_text = {}
        generated_val_text = {}
        for name, desc in tqdm(eligibility_requirements.items()):
            checker_gen_prompt = [
                {
                    "role": "system",
                    "content": self.gen_checker_prompt.format(
                        eligibility_requirement=desc
                    ),
                }
            ]
            dirty_checker_output = self.lm_backbone.forward(
                checker_gen_prompt,
                logging_role="code_gen",
            ).strip()
            clean_checker_output = extract_function_definitions(dirty_checker_output)[
                "check_eligibility"
            ]
            clean_checker_output = clean_checker_output.replace(
                "def check_eligibility", f"def {name}"
            )
            # replace all hh.get(...) with hh["..."]
            clean_checker_output = re.sub(
                r'hh\.get\((["\'])(.*?)\1\)', r'hh["\2"]', clean_checker_output
            )
            clean_checker_output = remove_raise_statements(clean_checker_output)
            # check if the code is a valid python function
            try:
                exec(clean_checker_output)
                generated_checker_text[name] = clean_checker_output
            except SyntaxError:
                raise ValueError(
                    f"Generated code is not a valid python function: {clean_checker_output[0]}"
                )
            error_var = None
            attempt_no = 0
            while True:
                # generate the validator
                try:
                    if error_var is None:
                        val_gen_prompt = [
                            {
                                "role": "system",
                                "content": self.gen_validator_prompt.format(
                                    eligibility_requirement=desc,
                                    code=clean_checker_output,
                                    name=name,
                                ),
                            }
                        ]
                    else:
                        val_gen_prompt = [
                            {
                                "role": "system",
                                "content": self.update_val_gen_prompt.format(
                                    attempt_no=attempt_no,
                                    code=dirty_val_output,
                                    error_message=error_var.args[0],
                                ),
                            }
                        ]
                    dirty_val_output = self.lm_backbone.forward(
                        val_gen_prompt,
                        logging_role="val_gen",
                    ).strip()
                    clean_val_output = extract_function_definitions(dirty_val_output)[
                        f"validate_{name}"
                    ]
                    # clean_val_output = clean_val_output.replace(
                    #     "def validate_{name}", f"def validate_{name}"
                    # )
                    clean_val_output = clean_val_output.replace(
                        "raise ValueError", "raise SchemaError"
                    )
                    exec(clean_val_output)  # define the val_{name} fn
                    val_fn = eval(f"validate_{name}")

                    # check if the validator passes on empty input
                    attempt_no += 1
                    if val_fn({}) is None:
                        break
                    # clean_val_output = remove_raise_statements(clean_val_output)

                except Exception as e:
                    error_var = e
                    print(f"{type(e).__name__} generating validator: {e}")
                    
                # if we succeeded, add the validator
            generated_val_text[name] = clean_val_output
        ### run the validator on empty input and rebuild it if empty input errors out ###

        # copy the template into a string
        with open("datamodels/template.py", "r") as template_file:
            template = template_file.read()

        # replace the template with the generated code
        # program_texts = []
        # program_dict_text = []
        # for name, code in generated_fns_text.items():
        eligibility_definition_block = "\n\n".join(generated_checker_text.values())
        eligibility_call_block = ",".join(
            [f"'{k}':{k}" for k in generated_checker_text.keys()]
        )
        val_definition_block = "\n\n".join(generated_val_text.values())
        val_call_block = ",".join(
            [f"'{k}':validate_{k}" for k in generated_val_text.keys()]
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

    def run_generated_code(self, locals):
        gen_code_path = locals["tf"].name
        synthetic_user = locals["synthetic_user"]
        # import the generated code from the temp file
        spec = importlib.util.spec_from_file_location("generated_code", gen_code_path)
        generated_code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_code)

        history = []
        locals["hh"] = dict()
        # run the generated code until we get a valid output
        while True:
            ### validate all inputs ##
            generated_validators = generated_code.vals
            for name, val in generated_validators.items():
                try:
                    val(locals["hh"])
                except SchemaError as e:
                    # get the stack trace
                    line = "\n".join(
                        traceback.format_exc().split("\n")[-3:-2]
                    )  # TODO: check accuracy
                    fn_name = traceback.extract_tb(e.__traceback__)[-1].name
                    fn_code = inspect.getsource(eval("generated_code" + "." + fn_name))
                    relevant_program = locals["eligibility_requirements"][
                        fn_name.lstrip("validate_")
                    ]

            ### Run the actual generated code ###
            try:
                eligibility = generated_code.run(local_scope=locals)
                break
            except Exception as e:
                # get the stack trace
                line = "\n".join(
                    traceback.format_exc().split("\n")[-3:-2]
                )  # TODO: check accuracy
                fn_name = traceback.extract_tb(e.__traceback__)[
                    2
                ].name  # TODO: get without hard coding index
                assert fn_name in list(locals["eligibility_requirements"].keys())
                fn_code = inspect.getsource(eval("generated_code" + "." + fn_name))
                relevant_program = locals["eligibility_requirements"][
                    fn_name.lstrip("validate_")
                ]

                # if there is a key error, ask a question to get the value
                if type(e) == KeyError:
                    key = e.args[0]
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
                    new_hh_value = self.forward_generic(
                        prompt=self.extract_value_from_ans_prompt.format(
                            eligibility_requirements=relevant_program,
                            line=line,
                            key=key,
                            cq=cq,
                            answer=ca,
                        ),
                        logging_role="extract_value_from_ans",
                    )
                    history.append({"role": "assistant", "content": new_hh_value})
                    locals["hh"][key] = new_hh_value
                    continue
                if type(e) == ValueError:
                    key = self.forward_generic(
                        prompt=self.value_error_find_key_prompt.format(
                            code=fn_code,
                            line=line,
                            traceback=str(e),
                        ),
                        logging_role="value_error_find_key",
                    ).strip("'\"` \n")
                    # target_type = e.args[0]
                    print(e)
                    match = re.search(
                        r"invalid literal for (.*)\(\) with base 10: \'(.*)\'", str(e)
                    ) or re.search(
                        r"could not convert string to (.*): \'(.*)\'", str(e)
                    )
                    target_type = match.group(1)
                    original_value = match.group(2)
                    # original_value = str(e)[str(e).find('"') + 1 :].strip('"')
                    retyped_val = self.forward_generic(
                        self.type_error_prompt.format(
                            eligibility_requirements=relevant_program,
                            line=line,
                            value=original_value,
                            target_type=target_type,
                            dialog=hist_to_str(history),
                        ),
                        logging_role="type_error",
                    )
                    try:
                        _ = eval(target_type)(retyped_val)
                        final_val = retyped_val
                    except ValueError:
                        final_val = {
                            "str": "None",
                            "int": "0",
                            "float": "0.0",
                            "bool": "False",
                        }[target_type]
                        print(
                            f"type_error retyping failed, setting val {original_value} to {final_val}"
                        )
                    # history.append({"role": "assistant", "content": retyped_val})
                    locals["hh"][key] = final_val
                    continue

                # check if the dialog can be used to update hh

                # code =
                # if we don't get a valid output, check the dialog to see if we can update hh
                # if not, ask another question
                continue
        return eligibility

    def forward_generic(self, prompt: str, logging_role: str):
        prompt = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        lm_output = self.lm_backbone.forward(prompt, logging_role=logging_role)
        # if lm_output starts and ends with '"`, remove them
        for c in ["'", '"', "`"]:
            if lm_output.startswith(c) and lm_output.endswith(c):
                lm_output = lm_output.strip(c)

        return lm_output
