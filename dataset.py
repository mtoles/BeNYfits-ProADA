# Temporary small, static dataset for ChatBot testing
import numpy as np
from names import get_full_name
from users import get_default_person, get_default_child

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
    },
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
    },
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
    },
)

# ex4
"""
Single parent, low income, one kid aged 2, 

Child and Dependent Care Tax Credit:    fail
Early Head Start Programs:              pass
Infant/Toddler Programs:                pass
Child Tax Credit:                       pass
Disability Rent Increase Exemption:     fail
Earned Income Tax Credit:               pass
Head Start:                             pass
Comprehensive After School:             fail
"""
dad = get_default_person()
dad["work_income"] = 500
dad["work_hours_per_week"] = 40

kid = get_default_child()

ex4 = (
    {
        # Single parent, low income, one kid,
        "programs": [
            "EarlyHeadStartPrograms",
            "InfantToddlerPrograms",
            "ChildTaxCredit",
            "DisabilityRentIncreaseExemption",
            "EarnedIncomeTaxCredit",
            "HeadStart",
            "ComprehensiveAfterSchool",
        ],
        "labels": ["fail", "pass", "pass", "pass", "fail", "pass", "pass", "fail"],
        "hh": {[dad, kid]},
    },
)

### Dataset ###

dataset = [
    ex1,
    ex2,
    ex3,
    ex4,
]
