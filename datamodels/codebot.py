from .chatbot import *
from tqdm import tqdm
from tempfile import NamedTemporaryFile
import re
import sys


class CodeBot(ChatBot):
    code_gen_prompt = """{eligibility_requirement}. Write a python function called `check_eligibility` that takes a dictionary `hh` containing relevant information in string form and determines user eligibility. `check_eligibility` returns a bool. Make your code as detailed as possible capturing every edge case. Do not provide anything besides code in your response."""
    ask_question_from_code_prompt = """We are writing code to check if a user is eligible for the following program:\n\n{program_text}\n\nGiven this line of code:\n\n```{key}```\n\nwhere `hh` represents data on the user's household. What question should we ask to the user about their household to determine the value to store in `hh`? Try your best even if you are unsure. Only respond with the question and enclose it in "double quotes"."""
    extract_value_from_ans_prompt = """Given this line of code:\n\n{key}\n\nwhere `hh` represents data on the user's household and the following dialog:\n\nBot: {cq}\nUser: {answer}\n\nWhat value would we expect in the `hh` dictionary? Give your answer last and enclose it in "double quotes"."""
    compare_prompt = """Although the types may not match, determine whether the following expression should be True or False:\n\n{a} {expression} {b}\n\nReturn only True or False."""
    cast_value_prompt = """Question: {cq}\n\nAnswer: {answer}\n\nFollowup Instruction: Convert the answer to a single value of type {target_type}. Give only the value, enclosed in "double quotes"."""

    def pre_conversation(self, local_scope: dict):
        eligibility_requirements = local_scope["eligibility_requirements"]
        tf = local_scope["tf"]
        ### Write checker code
        generated_fns_text = {}
        for name, desc in tqdm(eligibility_requirements.items()):
            prompt = [
                {
                    "role": "system",
                    "content": self.code_gen_prompt.format(
                        eligibility_requirement=desc
                    ),
                }
            ]
            lm_output = self.lm_backbone.forward(
                prompt,
                logging_role="code_gen",
            ).strip()
            # extract code between ```
            clean_code_output_matches = re.findall(
                r"(def check_eligibility.*?)```", str(lm_output), re.DOTALL
            )
            clean_code_output = (
                clean_code_output_matches[0]
                if len(clean_code_output_matches) > 0
                else lm_output
            )

            clean_code_output = clean_code_output.replace(
                "def check_eligibility(hh):", f"def {name}(hh):"
            )
            # check if the code is a valid python function
            try:
                exec(clean_code_output)
            except SyntaxError:
                raise ValueError(
                    f"Generated code is not a valid python function: {clean_code_output[0]}"
                )

            generated_fns_text[name] = clean_code_output

        # copy the template into a string
        with open("datamodels/template.py", "r") as template_file:
            template = template_file.read()

        # replace the template with the generated code
        # program_texts = []
        # program_dict_text = []
        # for name, code in generated_fns_text.items():
        definition_block = "\n\n".join(generated_fns_text.values())
        call_block = ",".join([f"'{k}':{k}" for k in generated_fns_text.keys()])

        program = template.replace("### FUNCTIONS PLACEHOLDER ###", definition_block)
        program = program.replace("### CALLS PLACEHOLDER ###", call_block)

        tf.write(program)
        tf.flush()
        tf.close()

        sys.path.append(tf.name)
        import generated_code

        eligibility = generated_code.run(local_scope)

        # super().pre_conversation(eligibility_requirements)

        return eligibility

    def ask_question_from_code(self, eligibility_requirements: str, key: str):
        prompt = [
            {
                "role": "system",
                "content": self.ask_question_from_code_prompt.format(
                    program_text=eligibility_requirements, key=key
                ),
            }
        ]
        lm_output = self.lm_backbone.forward(
            prompt, logging_role="ask_question_from_code"
        )
        # clean up the lm_output
        found_questions = re.findall(r'"(.*\?)"', lm_output)

        if len(found_questions) == 0:
            # nothing matching the regex
            return lm_output

        return found_questions[-1]

    def extract_value_from_ans(self, key: str, cq: str, answer: str):
        prompt = [
            {
                "role": "system",
                "content": self.extract_value_from_ans_prompt.format(
                    key=key, cq=cq, answer=answer
                ),
            }
        ]
        lm_output = self.lm_backbone.forward(
            prompt, logging_role="extract_value_from_ans"
        )

        # clean up the lm_output
        try:
            clean_value = re.findall(r"\"(.*)\"", lm_output)[-1]
        except IndexError:
            try:
                clean_value = re.findall(r"\'(.*)\'", lm_output)[-1]
            except IndexError:
                try:
                    clean_value = re.findall(r"\`(.*)\`", lm_output)[-1]
                except IndexError:
                    clean_value = lm_output

        return clean_value

    def compare_with_lm(self, a, b, operator):
        prompt = [
            {
                "role": "system",
                "content": self.compare_prompt.format(a=a, expression=operator, b=b),
            }
        ]
        lm_output = self.lm_backbone.forward(prompt, logging_role="compare_with_lm")
        # clean up the lm_output
        found_values = re.findall(r"True|False|true|false", lm_output)

        if len(found_values) == 0:
            raise ValueError(
                f"Could not find value in the following output: {lm_output}"
            )
        return found_values[-1]

    def cast_with_lm(self, cq, answer, target_type):
        assert target_type in ["int", "float", "bool"]
        # cast = {"int": int, "float": float, "bool": bool}[target_type]
        prompt = [
            {
                "role": "system",
                "content": self.cast_value_prompt.format(
                    cq=cq, answer=answer, target_type=target_type
                ),
            }
        ]
        lm_output = self.lm_backbone.forward(
            prompt, logging_role="cast_with_lm"
        ).strip()
        # Find anything in backticks

        # handle bools
        if "true" in lm_output.lower():
            return True
        if "false" in lm_output.lower():
            return False

        # handle floats
        patterns_without = r"|".join(
            [
                r"(?<!\S)(\d+)(?!\S)",
                r"(?<!\S)(.\d+)(?!\S)",
                r"(?<!\S)(\d+.)(?!\S)",
                r"(?<!\S)(\d+.\d+)(?!\S)",
            ]
        )
        pattern = "|".join(
            [
                r"(?<!\S)[\"'`](\d+)[\"'`](?!\S)",
                r"(?<!\S)[\"'`](.\d+)[\"'`](?!\S)",
                r"(?<!\S)[\"'`](\d+.)[\"'`](?!\S)",
                r"(?<!\S)[\"'`](\d+.\d)+[\"'`](?!\S)",
            ]
        )
        try:
            reduced = re.findall(
                f'{pattern}',  # double quotes
                lm_output.replace(",", "").replace("$", ""),
            )[-1]
        except:
            try:
                reduced = re.findall(
                    f"{patterns_without}",  # anything
                    lm_output.replace(",", "").replace("$", ""),
                )[-1]
            except:
                return 0
        # get the captured group that isn't empty
        actual_reduced = [x for x in reduced if x][-1]
        float_output = float(actual_reduced)
        if target_type == float:
            return float_output
        # handle ints in case integer is represented with a .
        return int(float_output)

