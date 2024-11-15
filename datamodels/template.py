from users import Household
import traceback
from copy import deepcopy


class ImaginaryData:
    def __init__(self, chatbot, synthetic_user, program_desc):
        self.value = None
        self.features = {}
        self.chatbot = chatbot
        self.synthetic_user = synthetic_user
        self.program_desc = program_desc

    def __iter__(self):
        return iter(self.features)

    def __getitem__(self, key):
        if type(key) == ImaginaryData:
            key = key.value
        if key in self.features:
            return self.features[key]
        else:
            self.features[key] = ImaginaryData(
                self.chatbot, self.synthetic_user, self.program_desc
            )
            return self.features[key]

    def __setitem__(self, key, value):
        # self.features[key] = value
        pass

    def __str__(self):
        self.establish_value()
        return self.value

    def __contains__(self, key):
        return True

    def __call__(cls, *args, **kwargs):
        return cls.__init__(*args, **kwargs)

    def __lt__(self, other):
        self.establish_value()
        try:
            return type(other)(self.value) < other
        except (ValueError, TypeError):
            return self.chatbot.compare_with_lm(self.value, other, "<")

    def __gt__(self, other):
        self.establish_value()
        try:
            return type(other)(self.value) > other
        except (ValueError, TypeError):
            return self.chatbot.compare_with_lm(self.value, other, ">")

    def __eq__(self, other):
        self.establish_value()
        try:
            return type(other)(self.value) == other
        except (ValueError, TypeError):
            return self.chatbot.compare_with_lm(self.value, other, "==")

    def __le__(self, other):
        self.establish_value()
        try:
            return type(other)(self.value) <= other
        except (ValueError, TypeError):
            return self.chatbot.compare_with_lm(self.value, other, "<=")

    def __ge__(self, other):
        self.establish_value()
        try:
            return type(other)(self.value) >= other
        except (ValueError, TypeError):
            return self.chatbot.compare_with_lm(self.value, other, ">=")

    def __getattr__(self, name):
        self.establish_value()
        return getattr(self.value, name)

    def get(self, key, default=None):
        return self.__getitem__(key)

    def __int__(self):
        self.establish_value()
        try:
            return int(self.value)
        except (ValueError, TypeError):
            return self.chatbot.cast_with_lm(self.cq, self.answer, "int")

    def __float__(self):
        self.establish_value()
        try:
            return float(self.value)
        except (ValueError, TypeError):
            return self.chatbot.cast_with_lm(self.cq, self.answer, "float")

    def __bool__(self):
        self.establish_value()
        try:
            return bool(self.value)
        except (ValueError, TypeError):
            return self.chatbot.cast_with_lm(self.cq, self.answer, "bool")

    # def __hash__(self):
    #     if self.value is not None:

    def establish_value(self):
        if self.value is None:
            # parse the ast to get the line of code we are at from `generated_code.py`
            stack = list(reversed(traceback.extract_stack()))
            # relevant frame should always be frame 2
            self.line = stack[2].line
            assert "hasattr" not in self.line
            print("line:   ", self.line)
            ### Pranay's dialog lookup call goes here ###
            self.cq = self.chatbot.ask_question_from_code(
                program_text=self.program_desc, key=self.line
            )
            print("cq:     ", self.cq)
            self.answer = self.synthetic_user.answer_cq(self.cq)
            print("answer: ", self.answer)
            self.value = self.chatbot.extract_value_from_ans(
                key=self.line, cq=self.cq, answer=self.answer
            )
            print("value:  ", self.value)
            print


def dummy_eligibility_program(hh: dict) -> bool:
    """
    Dummy function overwritten by the generated code
    """
    if hh[0]["age"] < 18:
        return True
    return False


### FUNCTIONS PLACEHOLDER ###


def run(local_scope: dict) -> bool:
    synthetic_user = local_scope["synthetic_user"]
    chatbot = local_scope["chatbot"]
    eligibility_requirements = local_scope["eligibility_requirements"]
    calls = {
        # "dummy_eligibility_program": dummy_eligibility_program
    }
    calls.update(
        {
            # "dummy": dummy_eligibility_program
            ### CALLS PLACEHOLDER ###
        }
    )
    eligibility = {}
    for name, p in calls.items():
        eligibility[name] = p

    outputs = {}
    # run each program
    for name, p in eligibility.items():
        outputs[name] = p(ImaginaryData(chatbot, synthetic_user, eligibility_requirements[name]))
    return outputs
