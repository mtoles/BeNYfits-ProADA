from copy import deepcopy

class SchemaError(Exception):
    def __init__(self, message):
        super().__init__(message)

def dummy_eligibility_program(hh: dict) -> bool:
    """
    Dummy function ignored by the generated code
    """
    if hh[0]["age"] < 18:
        return True
    return False

def validate_dummy_eligibility_program(hh: dict) -> bool:
    """
    Dummy validation function for `dummy_eligibility_program`
    """
    try: 
        _ = float(hh.get("age", 0))
    except ValueError:
        raise ValueError("Age must be an number.")
    return None


### ELIGIBILITY PROGRAMS PLACEHOLDER ###

### ELIGIBILITY VALIDATORS PLACEHOLDER ###

vals = {
    # "dummy_eligibility_program": validate_dummy_eligibility_program
    ### VALS PLACEHOLDER ###
}

def run(local_scope: dict) -> bool:
    hh = local_scope["hh"]

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
