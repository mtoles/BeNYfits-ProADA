from copy import deepcopy
from typing import Union


class SchemaError(Exception):
    def __init__(self, message):
        super().__init__(message)


def check_single_key(key, criterion: Union[type, list]):
    # If criteria is a list, check if key is in the list
    if type(criterion) == list:
        return key in criterion
    # If criteria is a type, check if key can be cast to that type
    elif type(criterion) == type:
        try:
            criterion(key)
            return True
        except:
            return False
    # No other criteria are supported
    else:
        return False


def check_all_keys(criteria: dict, hh: dict):
    for k, v in hh.items():
        if not check_single_key(k, criteria[k]):
            raise SchemaError(f"Schema error: {k} must be of type {criteria[k]}")
    return True


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

val_dict_getters = {
    # "dummy_eligibility_program": {"age", int},
    ### VALS PLACEHOLDER ###
}


def validate_user_data(program_name: str, hh: dict):
    # for p, vdg in val_dict_getters.items():
    vd = val_dict_getters[program_name]
    for k, criterion in vd().items():
        if k in hh.keys():
            if not check_single_key(hh[k], criterion):
                # return k and criterion on error
                return (k, criterion)
    # return None on success
    return None


vals = {k: v for k, v in val_dict_getters.items()}

calls = {
    # "dummy_eligibility_program": dummy_eligibility_program
}
calls.update(
    {
        # "dummy": dummy_eligibility_program
        ### CALLS PLACEHOLDER ###
    }
)

# def run(program_name: str, local_scope: dict) -> bool:
#     hh = local_scope["hh"]

#     eligibility = {}
#     for name, p in calls.items():
#         eligibility[name] = p

#     outputs = {}
#     # run each program
#     for name, p in eligibility.items():
#         outputs[name] = p(hh)
#     return outputs

# def run(local_scope: dict) -> bool:
#     hh = local_scope["hh"]

#     calls = {
#         # "dummy_eligibility_program": dummy_eligibility_program
#     }
#     calls.update(
#         {
#             # "dummy": dummy_eligibility_program
#             ### CALLS PLACEHOLDER ###
#         }
#     )
#     eligibility = {}
#     for name, p in calls.items():
#         eligibility[name] = p

#     outputs = {}
#     # run each program
#     for name, p in eligibility.items():
#         outputs[name] = p(hh)
#     return outputs
