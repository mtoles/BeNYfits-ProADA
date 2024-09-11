# Temporary small, static dataset for ChatBot testing

dataset = [
    {
        ### ChildAndDependentCareTaxCredit ###
        "program": "ChildAndDependentCareTaxCredit",
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
        "label": "pass",
    },
    {
        ### EarlyHeadStartPrograms ###
        "program": "EarlyHeadStartPrograms",
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
        "label": "pass",
    },
    {
        "program": "EarlyHeadStartPrograms",
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
        "label": "fail",
    },
]
