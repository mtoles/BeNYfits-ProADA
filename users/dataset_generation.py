from users.users import Household, Person, Household
from users.benefits_programs import *  # unused but import to register benefits programs in the metaclass
from users.user_features import *  # unused but import to register user features in the metaclass
import pandas as pd


def unit_test_dataset():
    hh1 = Household([Person.default_unemployed(is_self=True)])
    hh2 = Household(
        [
            Person.default_employed(is_self=True),
            Person.default_child(),
        ]
    )
    hh3 = Household(
        [
            Person.random_person(is_self=True),
            Person.random_person(),
            Person.random_person(),
        ]
    )
    hhs = [hh1, hh2, hh3]
    # nl_descs = []
    rows = []
    for hh in hhs:
        row = {}
        for name, bp in BenefitsProgramMeta.registry.items():
            row[name] = bp.__call__(hh)
        row["nl_desc"] = hh.nl_household_profile()
        rows.append(row)

    df = pd.DataFrame(rows)
    return df