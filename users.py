from schema import Schema, And, Or, Use, Optional, SchemaError
from names import get_full_name
import numpy as np

person_schema = Schema(
    {
        # Demographic Info
        Optional("name"): And(str, len),
        Optional("age"): And(Use(int), lambda n: n >= 0),
        Optional("disabled"): Use(bool),
        Optional("has_ssn"): Use(bool),
        Optional("has_atin"): Use(bool),
        Optional("has_itin"): Use(bool),
        Optional("can_care_for_self"): Use(bool),
        # Financial Info
        Optional("work_income"): And(int, lambda n: n >= 0),
        Optional("investment_income"): And(int, lambda n: n >= 0),
        Optional("provides_over_half_of_own_financial_support"): Use(bool),
        Optional("receives_hra"): Use(bool),
        Optional("receives_ssi"): Use(bool),
        # School Info
        Optional("student"): Use(bool),
        Optional("current_school_level"): Or(
            "pk",
            "k",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            None,
        ),
        # Work Info
        Optional("works_outside_home"): Use(bool),
        Optional("looking_for_work"): Use(bool),
        Optional("work_hours_per_week"): And(int, lambda n: n >= 0),
        Optional("days_looking_for_work"): And(int, lambda n: n >= 0),
        # Family Info
        Optional("in_foster_care"): Use(bool),
        Optional("attending_service_for_domestic_violence"): Use(bool),
        Optional("has_paid_caregiver"): Use(bool),
        # Housing Info
        Optional("lives_in_temp_housing"): Use(bool),
        Optional("name_is_on_lease"): Use(bool),
        Optional("monthly_rent_spending"): And(int, lambda n: n >= 0),
        Optional("lives_in_rent_stabilized_apartment"): Use(bool),
        Optional("lives_in_rent_controlled_apartment"): Use(bool),
        Optional("lives_in_mitchell-lama"): Use(bool),
        Optional("lives_in_limited_dividend_development"): Use(bool),
        Optional("lives_in_redevelopment_company_development"): Use(bool),
        Optional("lives_in_hdfc_development"): Use(bool),
        Optional("lives_in_section_213_coop"): Use(bool),
        Optional("lives_in_rent_regulated_hotel"): Use(bool),
        Optional("lives_in_rent_regulated_single"): Use(bool),
        # Relation Info
        Optional("relation"): Or(  # relation to user
            "self",
            "spouse",
            "child",
            "stepchild",
            "grandchild",
            "foster_child",
            "adopted_child",
            "sibling_niece_nephew",
            "other_family",
            "other_non_family",
        ),
        Optional("duration_more_than_half_prev_year"): Use(bool),
        Optional("lived_together_last_6_months"): Use(bool),
        Optional("filing_jointly"): Use(bool),
        Optional("dependent"): Use(bool),
    }
)


# random person generator
def get_random_person():
    person = {
        "name": get_full_name(),
        "age": np.random.randint(0, 100),
        "disabled": np.random.choice([True, False]),
        "has_ssn": np.random.choice([True, False]),
        "has_atin": np.random.choice([True, False]),
        "has_itin": np.random.choice([True, False]),
        "can_care_for_self": np.random.choice([True, False]),
        "work_income": np.random.randint(0, 100000),
        "investment_income": np.random.randint(0, 100000),
        "provides_over_half_of_own_financial_support": np.random.choice([True, False]),
        "student": np.random.choice([True, False]),
        "current_school_level": np.random.choice(
            [
                "pk",
                "k",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "other",
            ]
        ),
        "works_outside_home": np.random.choice([True, False]),
        "looking_for_work": np.random.choice([True, False]),
        "work_hours_per_week": np.random.randint(0, 60),
        "days_looking_for_work": np.random.randint(0, 365),
        "in_foster_care": np.random.choice([True, False]),
        "attending_service_for_domestic_violence": np.random.choice([True, False]),
        "lives_in_temp_housing": np.random.choice([True, False]),
        "name_is_on_lease": np.random.choice([True, False]),
        "monthly_rent_spending": np.random.randint(0, 10000),
        "lives_in_rent_stabilized_apartment": np.random.choice([True, False]),
        "lives_in_rent_controlled_apartment": np.random.choice([True, False]),
        "lives_in_mitchell-lama": np.random.choice([True, False]),
        "lives_in_limited_dividend_development": np.random.choice([True, False]),
        "lives_in_redevelopment_company_development": np.random.choice([True, False]),
        "lives_in_hdfc_development": np.random.choice([True, False]),
        "lives_in_section_213_coop": np.random.choice([True, False]),
        "lives_in_rent_regulated_hotel": np.random.choice([True, False]),
        "lives_in_rent_regulated_single": np.random.choice([True, False]),
        "relation": np.random.choice(
            [
                "spouse",
                "child",
                "stepchild",
                "grandchild",
                "foster_child",
                "adopted_child",
                "sibling_niece_nephew",
                "other_family",
                "other_non_family",
            ]
        ),
        "duration_more_than_half_prev_year": np.random.choice([True, False]),
        "lived_together_last_6_months": np.random.choice([True, False]),
        "filing_jointly": np.random.choice([True, False]),
        "dependent": np.random.choice([True, False]),
    }
    return person


def _one_self(hh):
    # check the household has exactly one self and that it is member 0
    if hh["members"][0]["relation"] != "self":
        raise SchemaError("Household must have exactly one `self`")
    for member in hh["members"][1:]:
        if "relation" in member.keys():
            if member["relation"] == "self":
                raise SchemaError("Household cannot have more than one `self`")
    return True


household_schema = Schema(
    And(
        {
            "members": [person_schema],
        },
        _one_self,
    )
)


def get_random_self_person():
    self_person = get_random_person()
    self_person["relation"] = "self"
    return self_person


if __name__ == "__main__":
    for i in range(10):
        members = [get_random_self_person()]
        for n in range(3):  # num family members
            members.append(get_random_person())
        household = {"members": members}
        household_schema.validate(household)

    print("Households are valid")


print
