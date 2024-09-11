from schema import Schema, And, Or, Use, Optional, SchemaError
from names import get_full_name
import numpy as np
import pandas as pd

# fmt: off
# DefaultName is a 20 year old NEET who 
# - qualifies for basically nothing 
# - has SSN
# - does not pay rent
person_struct = [
    # Demographic Info
    ("name", And(str, len), get_full_name(), "DefaultName"),
    ("age", And(Use(int), lambda n: n >= 0), np.random.randint(0, 100), 20),
    ("disabled", Use(bool), np.random.choice([True, False]), False),
    ("has_ssn", Use(bool), np.random.choice([True, False]), True),
    ("has_atin", Use(bool), np.random.choice([True, False]), False),
    ("has_itin", Use(bool), np.random.choice([True, False]), False),
    ("can_care_for_self", Use(bool), np.random.choice([True, False]), True),
    
    # Financial Info
    ("work_income", And(int, lambda n: n >= 0), np.random.randint(0, 100000), 0),
    ("investment_income", And(int, lambda n: n >= 0), np.random.randint(0, 100000), 0),
    ("provides_over_half_of_own_financial_support", Use(bool), np.random.choice([True, False]), True),
    ("receives_hra", Use(bool), np.random.choice([True, False]), False),
    ("receives_ssi", Use(bool), np.random.choice([True, False]), False),
    
    # School Info
    ("student", Use(bool), np.random.choice([True, False]), False),
    ("current_school_level", Or("pk", "k", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", None), 
     np.random.choice(["pk", "k", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", None]), None),
    
    # Work Info
    ("works_outside_home", Use(bool), np.random.choice([True, False]), False),
    ("looking_for_work", Use(bool), np.random.choice([True, False]), False),
    ("work_hours_per_week", And(int, lambda n: n >= 0), np.random.randint(0, 60), 0),
    ("days_looking_for_work", And(int, lambda n: n >= 0), np.random.randint(0, 365), 0),
    
    # Family Info
    ("in_foster_care", Use(bool), np.random.choice([True, False]), False),
    ("attending_service_for_domestic_violence", Use(bool), np.random.choice([True, False]), False),
    ("has_paid_caregiver", Use(bool), np.random.choice([True, False]), False),
    
    # Housing Info
    ("lives_in_temp_housing", Use(bool), np.random.choice([True, False]), False),
    ("name_is_on_lease", Use(bool), np.random.choice([True, False]), False),
    ("monthly_rent_spending", And(int, lambda n: n >= 0), np.random.randint(0, 10000), 0),
    ("lives_in_rent_stabilized_apartment", Use(bool), np.random.choice([True, False]), False),
    ("lives_in_rent_controlled_apartment", Use(bool), np.random.choice([True, False]), False),
    ("lives_in_mitchell-lama", Use(bool), np.random.choice([True, False]), False),
    ("lives_in_limited_dividend_development", Use(bool), np.random.choice([True, False]), False),
    ("lives_in_redevelopment_company_development", Use(bool), np.random.choice([True, False]), False),
    ("lives_in_hdfc_development", Use(bool), np.random.choice([True, False]), False),
    ("lives_in_section_213_coop", Use(bool), np.random.choice([True, False]), False),
    ("lives_in_rent_regulated_hotel", Use(bool), np.random.choice([True, False]), False),
    ("lives_in_rent_regulated_single", Use(bool), np.random.choice([True, False]), False),
    
    # Relation Info
    ("relation", Or("self", "spouse", "child", "stepchild", "grandchild", "foster_child", "adopted_child", 
                    "sibling_niece_nephew", "other_family", "other_non_family"), 
     np.random.choice(["spouse", "child", "stepchild", "grandchild", "foster_child", "adopted_child", 
                       "sibling_niece_nephew", "other_family", "other_non_family"]), "self"),
    ("duration_more_than_half_prev_year", Use(bool), np.random.choice([True, False]), True),
    ("lived_together_last_6_months", Use(bool), np.random.choice([True, False]), True),
    ("filing_jointly", Use(bool), np.random.choice([True, False]), False),
    ("dependent", Use(bool), np.random.choice([True, False]), False),
]
# fmt: on

person_schema = Schema({Optional(f[0]): f[1] for f in person_struct})

person_schema_df = pd.DataFrame(
    person_struct, columns=["field", "schema", "random", "default"]
).set_index("field")


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


def get_default_user():
    return person_schema_df["default"].to_dict()


def get_random_person():
    return (
        person_schema_df["random"].apply(lambda x: x() if callable(x) else x).to_dict()
    )


def get_default_child():
    child = get_default_user()
    child["relation"] = "child"
    child["provides_over_half_of_own_financial_support"] = False
    child["can_care_for_self"] = False
    child["age"] = 4
    child["student"] = True
    child["current_school_level"] = "pk"
    child["dependent"] = True
    return child


def get_random_self_person():
    self_person = get_random_person()
    self_person["relation"] = "self"
    return self_person


if __name__ == "__main__":
    #
    for i in range(10):
        members = [get_random_self_person()]
        for n in range(3):  # num family members
            members.append(get_random_person())
        household = {"members": members}
        household_schema.validate(household)

    # check default person
    default_person = get_default_user()
    household = {"members": [default_person]}
    print("Households are valid")


print
