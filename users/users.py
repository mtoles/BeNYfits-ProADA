from schema import Schema, And, Or, Use, Optional, SchemaError
from names import get_full_name
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Callable
from users.user_features import PersonAttributeMeta
from users.benefits_programs import BenefitsProgramMeta
np.random.seed(0)

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

    def validate(self):
        schemas = {
            attr.name: attr.schema for attr in PersonAttributeMeta.registry.values()
        }
        for attr, value in self.features.items():
            assert Schema(schemas[attr]).is_valid(
                value
            ), f"Invalid value `{value}` for attribute `{attr}` under schema `{schemas[attr]}`"

    @staticmethod
    def default_unemployed(random_name=True, is_self=False):
        attr_names = PersonAttributeMeta.registry.keys()
        defaults = [attr.default for attr in PersonAttributeMeta.registry.values()]
        person_dict = {attr: default for attr, default in zip(attr_names, defaults)}
        person = Person.from_dict(person_dict)
        if not is_self:
            person["relation"] = "other_family"
        return person

    @staticmethod
    def default_employed(random_name=True, is_self=False):
        attr_names = PersonAttributeMeta.registry.keys()
        defaults = [attr.default for attr in PersonAttributeMeta.registry.values()]
        person_dict = {attr: default for attr, default in zip(attr_names, defaults)}
        person = Person.from_dict(person_dict)
        person["works_outside_home"] = True
        person["work_income"] = 50000
        person["work_hours_per_week"] = 40
        if not is_self:
            person["relation"] = "other_family"
        return person

    @staticmethod
    def random_person(is_self=False):
        attr_names = PersonAttributeMeta.registry.keys()
        # person_dict = {attr: attr.random() for attr in attr_names}
        person_dict = {
            attr: PersonAttributeMeta.registry[attr].random() for attr in attr_names
        }
        person = Person.from_dict(person_dict)
        if is_self:
            person["relation"] = "self"
        return person

    @staticmethod
    def default_child(random_name=True):
        child = Person.default_unemployed(random_name=random_name)
        child["relation"] = "child"
        child["provides_over_half_of_own_financial_support"] = False
        child["can_care_for_self"] = False
        child["age"] = 4
        child["student"] = True
        child["current_school_level"] = "pk"
        child["dependent"] = True
        return child

    def nl_person_profile(self) -> str:
        name = self.features["name"]
        sentences = []
        for f, v in self.features.items():
            sentences.append(PersonAttributeMeta.registry[f].nl_fn(name, v))
            # sentences.append(fn(name, person[field]))
        return "\n".join(sentences).strip()


class Household:
    """
    A data class to represent a household
    """

    def __init__(self, members: list[Person] = [], co_owners: list[Person] = []):
        # create household from list of Persons
        for member in members + co_owners:
            assert isinstance(member, Person)
        # self.members = members

        self.members = members
        self.co_owners = co_owners

        self.features = {
            "members": self.members,
            "co_owners": self.co_owners,
        }  # TODO: remove after integrating Nikhil's programs
        self.validate()

    @classmethod
    def from_dict(cls, hh_dict: dict):
        # create household from dictionary
        members = [
            Person.from_dict(member["features"]) for member in hh_dict["members"]
        ]
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
        def _one_self(hh):
            # check the household has exactly one self and that it is member 0
            if hh["members"][0]["relation"] != "self":
                raise SchemaError("Household must have exactly one `self`")
            for member in hh["members"][1:]:
                if "relation" in member.features.keys():
                    if member["relation"] == "self":
                        raise SchemaError("Household cannot have more than one `self`")
            return True
        
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

    def nl_household_profile(self) -> str:
        user = self.members[0]
        user_name = user["name"]
        sentences = [
            f"You are {user_name}.",
            "You are seeking benefits on behalf of your household.",
        ]
        user_profile = user.nl_person_profile() + "\n=============="
        member_profiles = [member.nl_person_profile() for member in self.members[1:]]
        member_profiles = [x + "\n==============" for x in member_profiles]
        num_members = len(self.members)
        num_children = len([member for member in self.members if member["age"] < 18])
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
 

if __name__ == "__main__":
    #
    default_user = PersonAttributeMeta.default_unemployed()
    PersonAttributeMeta.validate(default_user)
    # should not pass
    default_user_2 = PersonAttributeMeta.default_unemployed()
    default_user_2["age"] = "fish"
    try:
        PersonAttributeMeta.validate(default_user_2)
    except AssertionError:
        e = True
    assert e

print
