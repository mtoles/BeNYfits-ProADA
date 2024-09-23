from schema import Schema, And, Or, Use, Optional, SchemaError
from names import get_full_name
import numpy as np
import pandas as pd


def _one_self(hh):
    # check the household has exactly one self and that it is member 0
    if hh["members"][0]["relation"] != "self":
        raise SchemaError("Household must have exactly one `self`")
    for member in hh["members"][1:]:
        if "relation" in member.keys():
            if member["relation"] == "self":
                raise SchemaError("Household cannot have more than one `self`")
    return True


def default_unemployed(random_name=True):
    person = person_schema_df["default"].to_dict()
    if random_name:
        person["name"] = get_full_name()
    return person


def default_employed(random_name=True):
    person = default_unemployed(random_name=random_name)
    person["works_outside_home"] = True
    person["work_income"] = 50000
    person["work_hours_per_week"] = 40
    return person


def random_person():
    return (
        person_schema_df["random"].apply(lambda x: x() if callable(x) else x).to_dict()
    )


def default_child(random_name=True):
    child = default_unemployed(random_name=random_name)
    child["relation"] = "child"
    child["provides_over_half_of_own_financial_support"] = False
    child["can_care_for_self"] = False
    child["age"] = 4
    child["student"] = True
    child["current_school_level"] = "pk"
    child["dependent"] = True
    return child


def random_self_person():
    self_person = random_person()
    self_person["relation"] = "self"
    return self_person


def nl_person_profile(person: dict) -> str:
    name = person["name"]
    sentences = []
    for field, schema, random, default, fn in person_struct:
        if field in person.keys():
            sentences.append(fn(name, person[field]))
    return "\n".join(sentences).strip()


def nl_household_profile(hh_df: pd.DataFrame) -> str:
    members = hh_df["hh"]["members"]
    user = members[0]
    user_name = user["name"]
    sentences = [f"You are {user_name}.", "You are the head of your household."]
    user_profile = nl_person_profile(user) + "\n=============="
    member_profiles = [nl_person_profile(member) for member in members[1:]]
    member_profiles = [x + "\n==============" for x in member_profiles]
    return "\n".join(
        sentences
        + [user_profile]
        + [
            f"Your household consists of the following {len(member_profiles)} additional members:"
        ]
        + member_profiles
    ).strip()


# fmt: off
grade_dict = {'pk':'preschool', 'k':'kindergarten', 1:'1st grade', 2:'2nd grade', 3:'3rd grade', 4:'4th grade', 5:'5th grade', 6:'6th grade', 7:'7th grade', 8:'8th grade', 9:'9th grade', 10:'10th grade', 11:'11th grade', 12:'12th grade'}
# DefaultName is a 20 year old NEET who 
# - qualifies for basically nothing 
# - has SSN
# - does not pay rent
person_struct = [
    #| Field Name | Schema | Random | Default | NL Function |
    # Demographic Info
    ("name", And(str, len), get_full_name(), "DefaultName", lambda n, x: f"Name: {n}"),
    ("age", And(Use(int), lambda n: n >= 0), np.random.randint(0, 100), 20, lambda n, x: f"{n} is {x} years old."),
    ("disabled", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is disabled." if x else f"{n} is not disabled."),
    ("has_ssn", Use(bool), np.random.choice([True, False]), True, lambda n, x: f"{n} has a social security number (SSN)." if x else f"{n} does not have a social security number (SSN)."),
    ("has_atin", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} has an adoption taxpayer ID number (ATIN)." if x else f"{n} does not have adoption taxpayer ID number (ATIN)."), # John has a TIN
    ("has_itin", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} has an individual taxpayer ID number (ITIN)." if x else f"{n} does not have an individual taxpayer ID number (ITIN)."), # John has a TIN
    ("can_care_for_self", Use(bool), np.random.choice([True, False]), True, lambda n, x: f"{n} can care for themselves." if x else f"{n} cannot care for themselves."),


    # Training Info
    ("enrolled_in_educational_training", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is enrolled in educational training." if x else f"{n} is not enrolled in educational training."),
    ("enrolled_in_vocational_training", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is enrolled in vocational training." if x else f"{n} is not enrolled in vocational training."),
    
    # Financial Info
    ("work_income", And(int, lambda n: n >= 0), np.random.randint(0, 100000), 0, lambda n, x: f"{n} makes {x} per year working."), # annual
    ("investment_income", And(int, lambda n: n >= 0), np.random.randint(0, 100000), 0, lambda n, x: f"{n} makes {x} per year from investments."), # annual
    ("provides_over_half_of_own_financial_support", Use(bool), np.random.choice([True, False]), True, lambda n, x: f"{n} provides over half of their own financial support." if x else f"{n} does not provide over half of their own financial support."),
    ("receives_hra", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} receives Health Reimbursement Arrangement (HRA)." if x else f"{n} does not receive Health Reimbursement Arrangement (HRA)."),
    ("receives_ssi", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} receives Supplemental Security Income (SSI)." if x else f"{n} does not receive Supplemental Security Income (SSI)."),
    ("receives_snap", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} receives Supplemental Nutrition Assistance Program (SNAP)." if x else f"{n} does not receive Supplemental Nutrition Assistance Program (SNAP)."),
    
    # School Info
    ("student", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is a student." if x else f"{n} is not a student."),
    ("current_school_level", 
        Or("pk", "k", 1,2,3,4,5,6,7,8,9,10,11,12, None),  # john is in 9th grade
        np.random.choice(["pk", 1,2,3,4,5,6,7,8,9,10,11,12, None]), 
        None, 
        lambda n, x: f"{n} is in {grade_dict[x]}." if x else f"{n} is not in school."
    ),
    
    # Work Info
    ("works_outside_home", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} works outside the home." if x else f"{n} does not work outside the home."),
    ("looking_for_work", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is looking for work." if x else f"{n} is not looking for work."),
    ("work_hours_per_week", And(int, lambda n: n >= 0), np.random.randint(0, 60), 0, lambda n, x: f"{n} works {x} hours per week."), # weekly
    ("days_looking_for_work", And(int, lambda n: n >= 0), np.random.randint(0, 365), 0, lambda n, x: f"{n} has been looking for work for {x} days." if x else f"{n} is not looking for work."), # daily
    
    # Family Info
    ("in_foster_care", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is in foster care." if x else f"{n} is not in foster care."),
    ("attending_service_for_domestic_violence", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is attending a service for domestic violence." if x else f"{n} is not attending a service for domestic violence."),
    ("has_paid_caregiver", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} has a paid caregiver." if x else f"{n} does not have a paid caregiver."),
    
    # Housing Info
    ("lives_in_temp_housing", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in temporary housing." if x else f"{n} does not live in temporary housing."),
    ("name_is_on_lease", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is on the household lease." if x else f"{n} is not on the household lease."),
    ("monthly_rent_spending", And(int, lambda n: n >= 0), np.random.randint(0, 10000), 0, lambda n, x: f"{n} spends {x} per month on rent."),
    ("lives_in_rent_stabilized_apartment", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in a rent stabilized apartment." if x else f"{n} does not live in a rent stabilized apartment."),
    ("lives_in_rent_controlled_apartment", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in a rent controlled apartment." if x else f"{n} does not live in a rent controlled apartment."),
    ("lives_in_mitchell-lama", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in a Mitchell-Lama development." if x else f"{n} does not live in a Mitchell-Lama development."),
    ("lives_in_limited_dividend_development", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in a limited dividend development." if x else f"{n} does not live in a limited dividend development."),
    ("lives_in_redevelopment_company_development", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in a redevelopment company development." if x else f"{n} does not live in a redevelopment company development."),
    ("lives_in_hdfc_development", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in a Housing Development Fund Corporation (HDFC) development." if x else f"{n} does not live in a Housing Developtment Fund Corporation (HDFC) development."),
    ("lives_in_section_213_coop", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in a Section 213 coop." if x else f"{n} does not live in a Section 213 coop."),
    ("lives_in_rent_regulated_hotel", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in a rent regulated hotel." if x else f"{n} does not live in a rent regulated hotel."),
    ("lives_in_rent_regulated_single", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} lives in a rent regulated single room occupancy (SRO)." if x else f"{n} does not live in a rent regulated single room occupancy (SRO)."),
    
    # Relation Info
    ("relation", 
        Or("self", "spouse", "child", "stepchild", "grandchild", "foster_child", "adopted_child", "sibling_niece_nephew", "other_family", "other_non_family"), 
        np.random.choice(["spouse", "child", "stepchild", "grandchild", "foster_child", "adopted_child", "sibling_niece_nephew", "other_family", "other_non_family"]), "self",
        lambda n, x: f"You are {n}" if x == "self" else f"{n} is your {x}"
    ),
    ("duration_more_than_half_prev_year", Use(bool), np.random.choice([True, False]), True, lambda n, x: f"{n} lived with you more than half of the previous year." if x else f"{n} did not live with you more than half of the previous year."),
    ("lived_together_last_6_months", Use(bool), np.random.choice([True, False]), True, lambda n, x: f"{n} lived with you for the last 6 months." if x else f"{n} did not live with you for the last 6 months."),
    ("filing_jointly", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is filing taxes jointly with you." if x else f"{n} is not filing taxes jointly with you."),
    ("dependent", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is your dependent." if x else f"{n} is not your dependent."),
    # Miscellaneous
    ("receiving_treatment_for_substance_abuse", Use(bool), np.random.choice([True, False]), False, lambda n, x: f"{n} is receiving treatment for substance abuse." if x else f"{n} is not receiving treatment for substance abuse."),
]
# fmt: on

person_schema = Schema({Optional(f[0]): f[1] for f in person_struct})

person_schema_df = pd.DataFrame(
    person_struct, columns=["field", "schema", "random", "default", "nl_fn"]
).set_index("field")


household_schema = Schema(
    And(
        {
            "members": [person_schema],
        },
        _one_self,
    )
)

if __name__ == "__main__":
    #
    for i in range(10):
        members = [random_self_person()]
        for n in range(3):  # num family members
            members.append(random_person())
        household = {"members": members}
        household_schema.validate(household)

    # check default person
    default_person = default_unemployed()
    household = {"members": [default_person]}
    print("Households are valid")

print
