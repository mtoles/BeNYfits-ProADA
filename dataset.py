# Temporary small, static dataset for ChatBot testing
import numpy as np
from names import get_full_name
from users import get_default_user, get_default_child

ex1 = (
    {
        ### ChildAndDependentCareTaxCredit ###
        "programs": ["ChildAndDependentCareTaxCredit"],
        "labels": ["pass"],
        "hh": {
            "members": [
                {
                    "relation": "self",
                    "works_outside_home": True,
                    "work_income": 10000,
                    "filing_jointly": True,
                },
                {
                    "relation": "spouse",
                    "student": True,
                    "works_outside_home": True,
                },
                {
                    "relation": "child",
                    "age": 12,
                    "has_paid_caregiver": True,
                    "duration_more_than_half_prev_year": True,
                },
            ]
        },
    }
)

ex2 = (
    {
        ### EarlyHeadStartPrograms ###
        "programs": ["EarlyHeadStartPrograms"],
        "labels": ["pass"],
        "hh": {
            "members": [
                {
                    "relation": "self",
                    "lives_in_temp_housing": True,
                    "receives_hra": True,
                    "receives_ssi": True,
                    "in_foster_care": False,
                    "work_income": 10000,
                },
            ]
        },
    }
)
ex3 = (
    {
        "programs": ["EarlyHeadStartPrograms"],
        "labels": ["fail"],
        "hh": {
            "members": [
                {
                    "relation": "self",
                    "lives_in_temp_housing": False,
                    "receives_hra": False,
                    "receives_ssi": False,
                    "in_foster_care": False,
                    "work_income": 100000,
                },
            ]
        },
    }
)

ex4 = (
    {
        "programs": ["InfantsAndToddlersPrograms"],
        "labels": ["fail"],
        "hh": {
            "members": [
                {
                    "relation": "self",
                    "work_hours_per_week": 9,
                    "enrolled_in_educational_training": False,
                    "enrolled_in_vocational_training": False,
                    "looking_for_work": False,
                    "days_looking_for_work": 5,
                    "lives_in_temp_housing": False,
                    "attending_services_for_domestic_violence": False,
                    "receiving_treatment_for_substance_abuse": False,
                    "work_income": 180000,
                },
            ]
        },
    }
)


### Dataset ###

dataset = [
    ex1,
    ex2,
    ex3,
    ex4,
]
