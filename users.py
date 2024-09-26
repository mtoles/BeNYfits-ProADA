from schema import Schema, And, Or, Use, Optional, SchemaError
from names import get_full_name
import numpy as np
import pandas as pd
from typing import List, Dict, Union

### FUNCTIONS ###


def _one_self(hh):
    # check the household has exactly one self and that it is member 0
    if hh["members"][0]["relation"] != "self":
        raise SchemaError("Household must have exactly one `self`")
    for member in hh["members"][1:]:
        if "relation" in member.features.keys():
            if member["relation"] == "self":
                raise SchemaError("Household cannot have more than one `self`")
    return True


def default_unemployed(random_name=True):
    person_features = person_schema_df["default"].to_dict()
    if random_name:
        person_features["name"] = get_full_name()
    person = Person()
    for field, value in person_features.items():
        person.features[field] = value
    return person


def default_employed(random_name=True):
    person = default_unemployed(random_name=random_name)
    person["works_outside_home"] = True
    person["work_income"] = 50000
    person["work_hours_per_week"] = 40
    return person


def random_person():
    person = default_unemployed(random_name=True)
    for field, schema, random, default, fn in person_features:
        person.features[field] = random()
    return person


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


# TODO: update w/ person class
def nl_person_profile(person: dict) -> str:
    name = person["name"]
    sentences = []
    for field, schema, random, default, fn in person_features:
        if field in person.features.keys():
            sentences.append(fn(name, person[field]))
    return "\n".join(sentences).strip()


# TODO: update w/ person class
def nl_household_profile(hh_df: pd.DataFrame) -> str:
    members = hh_df["hh"]["members"]
    user = members[0]
    user_name = user["name"]
    sentences = [f"You are {user_name}.", "You are the head of your household."]
    user_profile = nl_person_profile(user) + "\n=============="
    member_profiles = [nl_person_profile(member) for member in members[1:]]
    member_profiles = [x + "\n==============" for x in member_profiles]
    num_members = len(members)
    num_children = len([member for member in members if member["age"] < 18])
    return "\n".join(
        sentences
        + [user_profile]
        + [
            f"Your household consists of the following {len(member_profiles)} additional members:"
        ]
        + member_profiles
        + [
            f"There are {num_members} members in your household, of which {num_children} are children."
        ]
    ).strip()


# fmt: off

### CLASSES ###
class Person:
    """
    A data class to represent a person
    Features must be added after creation; a person is not valid until all features are added
    """
    def __init__(self):
        self.features = {}
        
    def __getitem__(self, key):
        return self.features[key]
    
    def validate(self):
        for field, schema, random, default, nl_fn in person_features:
            feature_val = self.features[field]
            Schema(schema).validate(feature_val)
    # support assignment
    def __setitem__(self, key, value):
        self.features[key] = value
    def get(self, key, default=None):
        return self.features.get(key, default)

    @classmethod
    def from_dict(cls, person_dict):
        person = cls()
        for field, value in person_dict.items():
            person.features[field] = value
        return person
    def total_income(self):
        return self["work_income"] + self["investment_income"]
class Household:
    """
    A data class to represent a household
    """
    def __init__(self, members: list[Person]=[]):
        # create household from list of Persons
        for member in members:
            assert isinstance(member, Person)
        # self.members = members
        
        self.members = members
        self.features = {"members": self.members} # TODO: remove after integrating Nikhil's programs
        self.validate()
    @classmethod
    def from_dict(cls, hh_dict: dict):
        # create household from dictionary
        members = [Person.from_dict(member["features"]) for member in hh_dict["members"]]
        hh = cls(members)
        hh.validate()
        return hh
    def __str__(self):
        return str([member.features for member in self.members])
    def __getitem__(self, key):
        return self.features[key]
    def __setitem__(self, key, value):
        self.features[key] = value
    def validate(self):
        for member in self.members:
            member.validate()
        assert _one_self(self)

    ### CONVENIENCE METHODS FOR GRAPH LOGIC ###
    def user(self):
        return self.members[0]
    # def children(self):
    #     return [member for member in self.members if member["relation"] in ["child", "stepchild", "foster_child", "adopted_child"]]
    def spouse(self):
        spouses = [member for member in self.members if member["relation"] == "spouse"]
        if len(spouses) == 0:
            return None
        return spouses[0]
    def parents(self):
        user = self.members[0]
        spouse = self.spouse()
        parents = [user]
        if spouse:
            parents.append(spouse)
        return parents
    def marriage_work_income(self):
        user_income = self.members[0]["work_income"]
        spouse = self.spouse()
        if spouse and self.user()["filing_jointly"]:
            spouse_income = spouse["work_income"]
        else:
            spouse_income = 0
        return user_income + spouse_income
    def marriage_investment_income(self):
        user_income = self.members[0]["investment_income"]
        spouse = self.spouse()
        if spouse and self.user()["filing_jointly"]:
            spouse_income = spouse["investment_income"]
        else:
            spouse_income = 0
        return user_income + spouse_income
    def marriage_total_income(self):
        return self.marriage_work_income() + self.marriage_investment_income()
    def hh_work_income(self):
        return sum([member["work_income"] for member in self.members])
    def hh_investment_income(self):
        return sum([member["investment_income"] for member in self.members])
    def hh_total_income(self):  
        return self.hh_work_income() + self.hh_investment_income()
    def num_members(self):
        return len(self.members)
    
### CONSTANTS ###

GRADE_DICT = {'pk':'preschool', 'k':'kindergarten', 1:'1st grade', 2:'2nd grade', 3:'3rd grade', 4:'4th grade', 5:'5th grade', 6:'6th grade', 7:'7th grade', 8:'8th grade', 9:'9th grade', 10:'10th grade', 11:'11th grade', 12:'12th grade'}
        
# Static person features that 
# DefaultName is a 20 year old NEET who 
# - qualifies for basically nothing 
# - has SSN
# - does not pay rent

person_features = [
    #| Field Name | Schema | Random | Default | NL Function |
    # Schema is a tuple of callables that must return Truthy for the field to be valid
    # Demographic Info
    ("name", And(str, len), lambda: get_full_name(), "DefaultName", lambda n, x: f"Name: {n}"),
    ("age", And(int, lambda n: n >= 0), lambda: np.random.randint(0, 100), 20, lambda n, x: f"{n} is {x} years old."),
    ("disabled", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is disabled." if x else f"{n} is not disabled."),
    ("has_ssn", And(bool,), lambda: bool(np.random.choice([True, False])), True, lambda n, x: f"{n} has a social security number (SSN)." if x else f"{n} does not have a social security number (SSN)."),
    ("has_atin", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} has an adoption taxpayer ID number (ATIN)." if x else f"{n} does not have adoption taxpayer ID number (ATIN)."), 
    ("has_itin", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} has an individual taxpayer ID number (ITIN)." if x else f"{n} does not have an individual taxpayer ID number (ITIN)."), 
    ("can_care_for_self", And(bool,), lambda: bool(np.random.choice([True, False])), True, lambda n, x: f"{n} can care for themselves." if x else f"{n} cannot care for themselves."),
    ("place_of_residence", And(str, len), lambda: np.random.choice(["NYC", "Jersey"]), "NYC", lambda n, x: f"{n} lives in {x}."),

    # Training Info
    ("enrolled_in_educational_training", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is enrolled in educational training." if x else f"{n} is not enrolled in educational training."),
    ("enrolled_in_vocational_training", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is enrolled in vocational training." if x else f"{n} is not enrolled in vocational training."),
    
    # Financial Info
    ("work_income", And(int, lambda n: n >= 0), lambda: np.random.randint(0, 100000), 0, lambda n, x: f"{n} makes {x} per year working."), # annual
    ("investment_income", And(int, lambda n: n >= 0), lambda: np.random.randint(0, 100000), 0, lambda n, x: f"{n} makes {x} per year from investments."), # annual
    ("provides_over_half_of_own_financial_support", And(bool,), lambda: bool(np.random.choice([True, False])), True, lambda n, x: f"{n} provides over half of their own financial support." if x else f"{n} does not provide over half of their own financial support."),
    ("receives_hra", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} receives Health Reimbursement Arrangement (HRA)." if x else f"{n} does not receive Health Reimbursement Arrangement (HRA)."),
    ("receives_ssi", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} receives Supplemental Security Income (SSI)." if x else f"{n} does not receive Supplemental Security Income (SSI)."),
    ("receives_snap", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} receives Supplemental Nutrition Assistance Program (SNAP)." if x else f"{n} does not receive Supplemental Nutrition Assistance Program (SNAP)."),
    ("receives_ssdi", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} receives Social Security Disability Insurance (SSDI)." if x else f"{n} does not receive Social Security Disability Insurance (SSDI)."),
    ("receives_va_disability", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} receives Veterans Affairs (VA) disability pension or compensation." if x else f"{n} does not receive Veterans Affairs (VA) disability pension or compensation."),
    ("has_received_ssi_or_ssdi", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} has received Supplemental Security Income (SSI) or Social Security Disability Insurance (SSDI) in the past." if x else f"{n} has not received Supplemental Security Income (SSI) or Social Security Disability Insurance (SSDI) in the past."),
    ("receives_disability_medicaid", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} receives Medicaid due to disability." if x else f"{n} does not receive Medicaid due to disability."),
    
    # School Info
    ("student", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is a student." if x else f"{n} is not a student."),
    ("current_school_level", 
        And(lambda x: x in ("pk", "k", 1,2,3,4,5,6,7,8,9,10,11,12, None),),  # TODO: should be a lambda function (use in)
        lambda: np.random.choice(["pk", 1,2,3,4,5,6,7,8,9,10,11,12, None]), 
        None, 
        lambda n, x: f"{n} is in {GRADE_DICT[x]}." if x else f"{n} is not in school."
    ),
    
    # Work Info
    ("works_outside_home", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} works outside the home." if x else f"{n} does not work outside the home."),
    ("looking_for_work", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is looking for work." if x else f"{n} is not looking for work."),
    ("work_hours_per_week", And(int, lambda n: n >= 0), lambda: np.random.randint(0, 60), 0, lambda n, x: f"{n} works {x} hours per week."), # weekly
    ("days_looking_for_work", And(int, lambda n: n >= 0), lambda: np.random.randint(0, 365), 0, lambda n, x: f"{n} has been looking for work for {x} days." if x else f"{n} is not looking for work."), # daily
    
    # Family Info
    ("in_foster_care", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is in foster care." if x else f"{n} is not in foster care."),
    ("attending_service_for_domestic_violence", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is attending a service for domestic violence." if x else f"{n} is not attending a service for domestic violence."),
    ("has_paid_caregiver", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} has a paid caregiver." if x else f"{n} does not have a paid caregiver."),
    
    # Housing Info
    ("lives_in_temp_housing", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in temporary housing." if x else f"{n} does not live in temporary housing."),
    ("name_is_on_lease", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is on the household lease." if x else f"{n} is not on the household lease."),
    ("monthly_rent_spending", And(int, lambda n: n >= 0), lambda: np.random.randint(0, 10000), 0, lambda n, x: f"{n} spends {x} per month on rent."),
    ("lives_in_rent_stabilized_apartment", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in a rent stabilized apartment." if x else f"{n} does not live in a rent stabilized apartment."),
    ("lives_in_rent_controlled_apartment", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in a rent controlled apartment." if x else f"{n} does not live in a rent controlled apartment."),
    ("lives_in_mitchell-lama", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in a Mitchell-Lama development." if x else f"{n} does not live in a Mitchell-Lama development."),
    ("lives_in_limited_dividend_development", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in a limited dividend development." if x else f"{n} does not live in a limited dividend development."),
    ("lives_in_redevelopment_company_development", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in a redevelopment company development." if x else f"{n} does not live in a redevelopment company development."),
    ("lives_in_hdfc_development", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in a Housing Development Fund Corporation (HDFC) development." if x else f"{n} does not live in a Housing Developtment Fund Corporation (HDFC) development."),
    ("lives_in_section_213_coop", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in a Section 213 coop." if x else f"{n} does not live in a Section 213 coop."),
    ("lives_in_rent_regulated_hotel", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in a rent regulated hotel." if x else f"{n} does not live in a rent regulated hotel."),
    ("lives_in_rent_regulated_single", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} lives in a rent regulated single room occupancy (SRO)." if x else f"{n} does not live in a rent regulated single room occupancy (SRO)."),
    
    # Relation Info
    ("relation", 
        And(lambda x: x in ("self", "spouse", "child", "stepchild", "grandchild", "foster_child", "adopted_child", "sibling","niece_nephew", "other_family", "other_non_family"),), 
        lambda: np.random.choice(["spouse", "child", "stepchild", "grandchild", "foster_child", "adopted_child", "sibling_niece_nephew", "other_family", "other_non_family"]), "self",
        lambda n, x: f"You are {n}" if x == "self" else f"{n} is your {x}"
    ),
    ("duration_more_than_half_prev_year", And(bool,), lambda: bool(np.random.choice([True, False])), True, lambda n, x: f"{n} lived with you more than half of the previous year." if x else f"{n} did not live with you more than half of the previous year."),
    ("lived_together_last_6_months", And(bool,), lambda: bool(np.random.choice([True, False])), True, lambda n, x: f"{n} lived with you for the last 6 months." if x else f"{n} did not live with you for the last 6 months."),
    ("filing_jointly", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is filing taxes jointly with you." if x else f"{n} is not filing taxes jointly with you."),
    ("dependent", And(bool,), lambda: bool(np.random.choice([True, False])), False, lambda n, x: f"{n} is your dependent." if x else f"{n} is not your dependent."),

    # Miscellaneous
    ("receiving_treatment_for_substance_abuse", And(bool,), lambda: bool(bool(np.random.choice([True, False]))), False, lambda n, x: f"{n} is receiving treatment for substance abuse." if x else f"{n} is not receiving treatment for substance abuse."),
]
person_schema_df = pd.DataFrame(
    person_features, columns=["field", "schema", "random", "default", "nl_fn"]
).set_index("field")
# fmt: on
# def _check_single_feature(x: tuple, val: any) -> bool:
#     """
#     x: a row in person_features
#     val: the value in the person to check
#     """
#     cons = x[1]
#     try:
#         Schema(cons).validate(val)
#         return True
#     except SchemaError:
#         return False


# # person_schema = Schema({Optional(f[0]): f[1] for f in person_struct})

# person_schema = Schema(
#     And(
#         lambda p: isinstance(p, Person),
#         # lambda p: all(
#         #     [_check_single_feature(x, p.features[x[0]]) for x in person_features]
#         # ),
#         lambda p: _check_single_feature(
#             person_features[0], p.features[person_features[0][0]]
#         ),
#     )
# )


# household_schema = Schema(
#     And(
#         {
#             "members": [person_schema],
#         },
#         _one_self,
#     )
# )

if __name__ == "__main__":
    #
    user = default_unemployed()
    user.validate()
    user = random_person()
    # user.features["name"] = 1
    user.validate()
    hh = Household([user])
    hh.validate()

    for i in range(10):
        members = [random_self_person()]
        for n in range(3):  # num family members
            members.append(random_person())
        hh = Household(members)
        hh.validate()

    # check default person
    default_person = default_unemployed()
    household = {"members": [default_person]}
    print("Households are valid")

print
