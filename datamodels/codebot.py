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

    gen_validator_prompt = """{attempt_no}\nEligibility Requirements:\n{eligibility_requirement}\n\ncode:\n{code}\n\nWrite a python function called `get_{name}_type_dict` that takes no inputs and returns a dictionary that describe the data used in the code above. The keys of the dictionary are keys used in the input dictionary `hh`. The values of the dictionary are the data type that the key should be, or, if the key should take one of a few values, the list of possible values. Here is an example:

def example_input_program(hh: dict) -> bool:
    if int(hh["age"]) =< 18 and hh["filing_status"] == "single":
        return True
    if hh["has_id"] == "no" or float(hh["income"]) > 10000/12 or hh["address"] == "Boston":
        return False

you should output the function:    

def get_example_input_type_dict():
    kv = {{
        "age": int,
        "filing_status": ["single", "married"],
        "has_id": ["yes", "no"],
        "income": float,
        "address": str,
    }}
    return kv

return ONLY your function."""

    extract_value_from_ans_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to extract the value of {key} from the following dialog:\n\nQuestion: {cq}\n\nAnswer:\n{answer}\n\nWhat should we set as the value of {key}? Return ONLY the value."""
    key_error_prompt = """Context:\n{eligibility_requirements}\n\nLine:\n```{line}```\n\nWe need to determine what value of {key} should be stored in the `hh` dictionary. Ask a question to the user that would get this value. Return ONLY the question."""

    type_error_prompt = """Context:\n{eligibility_requirements}\n\nDialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, cannot be cast to type {target_type}. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""
    str_error_prompt = """Context:\n{eligibility_requirements}\n\nDialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, is not one of {target_options}. Which option of {target_options} is most similar to {value}? Return ONLY the value."""  # same as above but for when the target is a list of strings

    value_error_find_key_prompt = """Code:\n{code}\n\nLine:\n{line}\n\nTraceback: {traceback}\n\nWhat key is responsible for the error in the traceback? The possible keys are {key_options}. Return ONLY the key."""
    update_val_gen_prompt = """Attempt no. {attempt_no}\n\nCode:\n{code}\n\nError message:\n{error_message}.\n\nThe code above checks whether the type of the input is correct. Update the code so that all defaults pass, but nonsensical inputs of the wrong type fail. If there are no defaults in a dict.get() call, add a default value. Only check the type of the input, not the range. Do not check that all values are present. Ensure that an empty dictionary is valid. Return ONLY the code. """
    schema_error_prompt = """Context:\n{eligibility_requirements}\n\Dialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, does not fit the following criteria: `{target_type}`. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""

    def pre_conversation(self, local_scope: dict):
        eligibility_requirements = local_scope["eligibility_requirements"]
        tf = local_scope["tf"]
        ### Write checker code
        generated_checker_text = {}
        generated_val_text = {}
        for name, desc in tqdm(eligibility_requirements.items()):
            checker_attempt_no = -1
            while True:
                checker_attempt_no += 1
                print(f"attempting to generate checker, attempt {checker_attempt_no}")
                checker_gen_prompt = [
                    {
                        "role": "system",
                        "content": self.gen_checker_prompt.format(
                            attempt_no=checker_attempt_no, eligibility_requirement=desc
                        ),
                    }
                ]
                dirty_checker_output = self.lm_backbone.forward(
                    checker_gen_prompt,
                    logging_role="code_gen",
                ).strip()
                try:
                    clean_checker_output = extract_function_definitions(
                        dirty_checker_output
                    )["check_eligibility"]
                    clean_checker_output = clean_checker_output.replace(
                        "def check_eligibility", f"def {name}"
                    )
                    # replace all hh.get(...) with hh["..."]
                    clean_checker_output = re.sub(
                        r'hh\.get\((["\'])(.*?)\1\)', r'hh["\2"]', clean_checker_output
                    )
                    clean_checker_output = remove_raise_statements(clean_checker_output)
                    # check if the code is a valid python function
                    exec(clean_checker_output)
                    generated_checker_text[name] = clean_checker_output
                    break
                except Exception as e:
                    pass
            error_var = None
            val_attempt_no = -1
            while True:
                val_attempt_no += 1
                print(f"attempting to generate validator, attempt {val_attempt_no}")
                # generate the validator
                try:
                    # if error_var is None:
                    val_gen_prompt = [
                        {
                            "role": "system",
                            "content": self.gen_validator_prompt.format(
                                attempt_no=val_attempt_no,
                                eligibility_requirement=desc,
                                code=clean_checker_output,
                                name=name,
                            ),
                        }
                    ]
                    # else:
                    #     val_gen_prompt = [
                    #         {
                    #             "role": "system",
                    #             "content": self.update_val_gen_prompt.format(
                    #                 attempt_no=attempt_no,
                    #                 code=dirty_val_output,
                    #                 error_message=error_var.args[0],
                    #             ),
                    #         }
                    #     ]
                    dirty_val_output = self.lm_backbone.forward(
                        val_gen_prompt,
                        logging_role="val_gen",
                    ).strip()
                    clean_val_output = extract_function_definitions(dirty_val_output)[
                        f"get_{name}_type_dict"
                    ]
                    # exec(clean_val_output)  # define the val_{name} fn
                    # val_dict = eval(f"get_{name}_type_dict()")
                    break  # just one iteration, assume it's right

                    # check if the validator passes on empty input
                    # if val_fn({}) is None:
                    #     break
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
        prev_hh = None
        while True:
            if locals["hh"] == prev_hh:
                return {
                    "hh": locals["hh"],
                    "history": history,
                    "eligibility": eligibility,
                    "completed": False
                }
            ### validate all inputs ##
            val_result = generated_code.validate_user_data(locals["hh"])
            if val_result is not None:
                key, criterion = val_result
                val = locals["hh"][key]
                print(
                    f"val '{val}' for key: '{key}' does not fit criterion: `{criterion}`"
                )
                # TODO: use lm to cast it to the type or retry the question
                #     schema_error_prompt = """Context:\n{eligibility_requirements}\n\Dialog:{dialog}\n\nLine:\n```{line}```\n\nThe string value, {value}, does not fit the following criteria: `{target_type}`. What string can we use instead that can be cast to type {target_type}? Return ONLY the value."""

                fix_val = self.forward_generic(
                    self.schema_error_prompt.format(
                        eligibility_requirements=relevant_program,
                        dialog=hist_to_str(history),
                        line=line,
                        value=val,
                        target_type=criterion,
                    ),
                    logging_role="fix_val",
                )
                prev_hh = deepcopy(locals["hh"])
                locals["hh"][key] = fix_val

                # # get the stack trace
                # line = "\n".join(
                #     traceback.format_exc().split("\n")[-3:-2]
                # )  # TODO: check accuracy
                # fn_name = traceback.extract_tb(e.__traceback__)[-1].name
                # fn_code = inspect.getsource(eval("generated_code" + "." + fn_name))
                # relevant_program = locals["eligibility_requirements"][
                #     fn_name.lstrip("validate_")
                # ]

            ### Run the actual generated code ###
            try:
                eligibility = generated_code.run(local_scope=locals)
                break
            except Exception as e:
                error_var = e
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
                relevant_val_dict = generated_code.val_dict_getters[fn_name]()

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
                    key_options = set(locals["hh"].keys()).intersection(
                        set(relevant_val_dict.keys())
                    )
                    raw_key = self.forward_generic(
                        prompt=self.value_error_find_key_prompt.format(
                            code=fn_code,
                            line=line,
                            traceback=str(e),
                            key_options=str(key_options),
                        ),
                        logging_role="value_error_find_key",
                    )
                    key = raw_key.strip("'\"` \n.")
                    # target_type = e.args[0]
                    # print(e)
                    target_type = relevant_val_dict[key]
                    original_value = locals["hh"][key]
                    # original_value = str(e)[str(e).find('"') + 1 :].strip('"')
                    if type(target_type) == type:
                        prompt = self.type_error_prompt.format(
                            eligibility_requirements=relevant_program,
                            line=line,
                            value=original_value,
                            target_type=target_type,
                            dialog=hist_to_str(history),
                        )
                    else:  # type(target_type) == list[str]
                        prompt = self.str_error_prompt.format(
                            eligibility_requirements=relevant_program,
                            line=line,
                            value=original_value,
                            target_options=target_type,
                            dialog=hist_to_str(history),
                        )
                    retyped_val = self.forward_generic(
                        prompt=prompt,
                        logging_role="type_error",
                    )
                    prev_hh = deepcopy(locals["hh"])
                    locals["hh"][key] = retyped_val
                    continue
                    # try:
                    #     _ = target_type(retyped_val)
                    #     final_val = retyped_val
                    # except ValueError:
                    #     final_val = {
                    #         "str": "None",
                    #         "int": "0",
                    #         "float": "0.0",
                    #         "bool": "False",
                    #     }[target_type]
                    #     print(
                    #         f"type_error retyping failed, setting val {original_value} to {final_val}"
                    #     )
                    # # history.append({"role": "assistant", "content": retyped_val})

                # check if the dialog can be used to update hh

                # code =
                # if we don't get a valid output, check the dialog to see if we can update hh
                # if not, ask another question
                continue
        return {
            "hh": locals["hh"],
            "history": history,
            "eligibility": eligibility,
            "completed": True
        }

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
