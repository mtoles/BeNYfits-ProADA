from .chatbot import *
from tqdm import tqdm
from tempfile import NamedTemporaryFile
import re
import sys


class CodeBot(ChatBot):
    code_gen_prompt = """{eligibility_requirement}. Write a python function called `check_eligibility` that takes a dictionary `hh` containing relevant information in string form and determines user eligibility. `check_eligibility` returns a bool. Make your code as detailed as possible capturing every edge case. Do not provide anything besides code in your response."""
    ask_question_from_code_prompt = """We are writing code to check if a user is eligible for the following program:\n\n{program_text}\n\nGiven this line of code:\n\n{key}\n\nwhere `hh` represents data on the user's household. What question should we ask to determine the value to store in `hh`? Give your question last and enclose it in "double quotes"."""
    extract_value_from_ans_prompt = """Given this line of code:\n\n{key}\n\nwhere `hh` represents data on the user's household and the following dialog:\n\nBot: {cq}\nUser{answer}\n\nWhat value would we expect in the `hh` dictionary? Give your answer last and enclose it in "double quotes"."""

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
            )
            # extract code between ```
            clean_code_output = re.findall(
                r"(def check_eligibility.*?)```", str(lm_output), re.DOTALL
            )[0]

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

        hh = {}
        eligibility = generated_code.run(local_scope)

        ### Wrap checker code

        ### Run unsafely with `exec`

        super().pre_conversation(eligibility_requirements)

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
        found_values = re.findall(r"\"(.*)\"", lm_output)

        if len(found_values) == 0:
            raise ValueError(
                f"Could not find value in the following output: {lm_output}"
            )
        return found_values[-1]
