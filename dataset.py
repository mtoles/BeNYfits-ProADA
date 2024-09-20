# Temporary small, static dataset for ChatBot testing
import numpy as np
from names import get_full_name
from users import (
    default_unemployed,
    default_child,
    default_employed,
    nl_household_profile,
    household_schema,
)
import pandas as pd

np.random.seed(42)

top_8_programs = [
    "ChildAndDependentCareTaxCredit",
    "EarlyHeadStartPrograms",
    "InfantToddlerPrograms",
    "ChildTaxCredit",
    "DisabilityRentIncreaseExemption",
    "EarnedIncomeTaxCredit",
    "HeadStart",
    "ComprehensiveAfterSchool",
]

### ex0 ###
dad = default_unemployed()
dad["work_income"] = 500
dad["work_hours_per_week"] = 40
kid = default_child()
# validate

ex0 = {
    # Single parent, low income, one kid,
    "programs": top_8_programs,
    "labels": ["fail", "fail", "fail", "pass", "fail", "pass", "pass", "fail"],
    "hh": {"members": [dad, kid]},
    "note": "Single parent, low income, one kid",
}

### ex1 ###
mom = default_employed()
mom["work_income"] = 12000
mom["name_is_on_lease"] = True
mom["monthly_rent_spending"] = 900
mom["receives_snap"] = True
mom["lives_in_rent_stabilized_apartment"] = True
mom["receives_ssi"] = True
kid = default_child()
kid["age"] = 10
kid["current_school_level"] = 5
kid["has_paid_caregiver"] = True
kid["lives_in_rent_stabilized_apartment"] = True
ex1 = {
    "programs": top_8_programs,
    "labels": ["pass", "fail", "fail", "pass", "pass", "pass", "pass", "fail"],
    "hh": {"members": [mom, kid]},
    "note": "Single parent, low income, one kid, paid caregiver, rent stabilized apartment",
}

### ex2 ###
mom = default_employed()
mom["work_income"] = 250000
mom["name_is_on_lease"] = True
mom["filing_jointly"] = True
dad = default_employed()
dad["relation"] = "spouse"
dad["work_income"] = 250000
dad["name_is_on_lease"] = True
dad["filing_jointly"] = True
kid1 = default_child()
kid1["age"] = 10
kid1["current_school_level"] = 5
kid2 = default_child()
kid2["age"] = 3
kid2["current_school_level"] = None
kid2["has_paid_caregiver"] = True
kid3 = default_employed()  # adult child
kid3["relation"] = "child"
ex2 = {
    "programs": top_8_programs,
    "labels": ["pass", "fail", "fail", "fail", "fail", "fail", "fail", "pass"],
    "hh": {"members": [mom, dad, kid1, kid2, kid3]},
    "note": "Two parent, high income, two young kids, one adult child, paid caregiver, filing jointly",
}


### Dataset ###

if __name__ == "__main__":
    dataset = [ex0, ex1, ex2]
    # validate all the households
    for i, ex in enumerate(dataset):
        household_schema.validate(ex["hh"])
    df = pd.DataFrame(dataset)
    df["hh_nl_desc"] = df.apply(nl_household_profile, axis=1)
    df.to_json("dataset_v0.1.0.jsonl", lines=True, orient="records")


# ex_a = (
#     {
#         ### ChildAndDependentCareTaxCredit ###
#         "programs": ["ChildAndDependentCareTaxCredit"],
#         "labels": ["pass"],
#         "hh": {
#             "members": [
#                 {
#                     "relation": "self",
#                     "works_outside_home": True,
#                     "work_income": 10000,
#                     "filing_jointly": True,
#                 },
#                 {
#                     "relation": "spouse",
#                     "student": True,
#                     "works_outside_home": True,
#                 },
#                 {
#                     "relation": "child",
#                     "age": 12,
#                     "has_paid_caregiver": True,
#                     "duration_more_than_half_prev_year": True,
#                 },
#             ]
#         },
#     },
# )

# ex_b = (
#     {
#         ### EarlyHeadStartPrograms ###
#         "programs": ["EarlyHeadStartPrograms"],
#         "labels": ["pass"],
#         "hh": {
#             "members": [
#                 {
#                     "relation": "self",
#                     "lives_in_temp_housing": True,
#                     "receives_hra": True,
#                     "receives_ssi": True,
#                     "in_foster_care": False,
#                     "work_income": 10000,
#                 },
#             ]
#         },
#     },
# )
# ex_c = (
#     {
#         "programs": ["EarlyHeadStartPrograms"],
#         "labels": ["fail"],
#         "hh": {
#             "members": [
#                 {
#                     "relation": "self",
#                     "lives_in_temp_housing": False,
#                     "receives_hra": False,
#                     "receives_ssi": False,
#                     "in_foster_care": False,
#                     "work_income": 100000,
#                 },
#             ]
#         },
#     },
# )
