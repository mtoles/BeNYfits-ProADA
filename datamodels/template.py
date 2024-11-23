from users.users import Household
import traceback
from copy import deepcopy



def dummy_eligibility_program(hh: dict) -> bool:
    """
    Dummy function overwritten by the generated code
    """
    if hh[0]["age"] < 18:
        return True
    return False


### FUNCTIONS PLACEHOLDER ###


def run(local_scope: dict) -> bool:
    hh = local_scope["hh"]
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
        outputs[name] = p(
            hh
        )
    return outputs
