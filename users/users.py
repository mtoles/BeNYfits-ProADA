from schema import Schema, And, Or, Use, Optional, SchemaError
from names import get_full_name
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Callable
from users.user_features import (
    PersonAttributeMeta,
    HousingEnum,
    RelationEnum,
    SexEnum,
    PlaceOfResidenceEnum,
    CitizenshipEnum,
    EducationLevelEnum,
    GradeLevelEnum,
)
from users import user_features


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
        return self["annual_work_income"] + self["annual_investment_income"]

    def validate(self):
        schemas = {
            attr.name: attr.schema for attr in PersonAttributeMeta.registry.values()
        }
        for attr, value in self.features.items():
            assert Schema(schemas[attr]).is_valid(
                value
            ), f"Invalid value `{value}` (of type {type(value)}) for attribute `{attr}` under schema `{schemas[attr]}`"

    # @staticmethod
    # def default_unemployed(random_name=True, is_self=False):
    #     attr_names = PersonAttributeMeta.registry.keys()
    #     defaults = [attr.default for attr in PersonAttributeMeta.registry.values()]
    #     person_dict = {attr: default for attr, default in zip(attr_names, defaults)}
    #     person = Person.from_dict(person_dict)
    #     if not is_self:
    #         person["relation"] = "other_family"
    #     return person

    # @staticmethod
    # def default_employed(random_name=True, is_self=False):
    #     attr_names = PersonAttributeMeta.registry.keys()
    #     defaults = [attr.default for attr in PersonAttributeMeta.registry.values()]
    #     person_dict = {attr: default for attr, default in zip(attr_names, defaults)}
    #     person = Person.from_dict(person_dict)
    #     person["works_outside_home"] = True
    #     person["work_income"] = 50000
    #     person["work_hours_per_week"] = 40
    #     if not is_self:
    #         person["relation"] = "other_family"
    #     return person

    @staticmethod
    def default_person(random_name=False, is_self=True):
        attr_names = PersonAttributeMeta.registry.keys()
        defaults = [attr.default for attr in PersonAttributeMeta.registry.values()]
        person_dict = {attr: default for attr, default in zip(attr_names, defaults)}
        person = Person.from_dict(person_dict)
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

    # @staticmethod
    # def default_child(random_name=True):
    #     child = Person.default_unemployed(random_name=random_name)
    #     child["relation"] = "child"
    #     child["provides_over_half_of_own_financial_support"] = False
    #     child["can_care_for_self"] = False
    #     child["age"] = 4
    #     child["student"] = True
    #     child["current_school_level"] = "pk"
    #     child["dependent"] = True
    #     return child

    # @staticmethod
    # def default_adult_dependent(random_name=True):
    #     child = Person.default_unemployed(random_name=random_name)
    #     child["relation"] = "dependent"
    #     child["provides_over_half_of_own_financial_support"] = False
    #     child["can_care_for_self"] = False
    #     child["age"] = 78
    #     child["dependent"] = True
    #     return child

    def nl_person_profile(self) -> str:
        name = self.features["name"]
        sentences = []

        for f, v in self.features.items():
            new_sentence = PersonAttributeMeta.registry[f].nl_fn(name, v)
            assert new_sentence is not None, f"Failed to generate sentence for {f}"
            sentences.append(new_sentence)
            # sentences.append(fn(name, person[field]))
        return "\n".join(sentences).strip()

    def nl_person_profile_always_include(self) -> str:
        name = self.features["name"]
        # sentences = [f"You are {name}"]
        sentences = []
        for f, v in self.features.items():
            if PersonAttributeMeta.registry[f].always_include: # fixed
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
            # Person.from_dict(member["features"]) for member in hh_dict["members"]
            Person.from_dict(member["features"])
            for member in hh_dict["features"]["members"]
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

    def conform(self):
        for i in range(len(self.members)):
            for attr in user_features.BasePersonAttr.registry.keys():
                cls = user_features.BasePersonAttr.registry[attr]
                self.members[i][attr] = cls.conform(cls, self, i, self.members[i][attr])
        self.validate()
        return self

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

    def marriage_annual_work_income(self):
        user_income = self.members[0]["annual_work_income"]
        spouse = self.spouse()
        if spouse and self.user()["filing_jointly"]:
            spouse_income = spouse["annual_work_income"]
        else:
            spouse_income = 0
        return user_income + spouse_income

    def marriage_annual_investment_income(self):
        user_income = self.members[0]["annual_investment_income"]
        spouse = self.spouse()
        if spouse and self.user()["filing_jointly"]:
            spouse_income = spouse["annual_investment_income"]
        else:
            spouse_income = 0
        return user_income + spouse_income

    def marriage_total_income(self):
        return (
            self.marriage_annual_work_income()
            + self.marriage_annual_investment_income()
        )

    def hh_annual_work_income(self):
        return sum([member["annual_work_income"] for member in self.members])

    def hh_annual_investment_income(self):
        return sum([member["annual_investment_income"] for member in self.members])

    def hh_annual_total_income(self):
        return self.hh_annual_work_income() + self.hh_annual_investment_income()

    def num_members(self):
        return len(self.members)

    def set_housing_type(self, htype: str):
        """
        Set the type of housing for the household property.
        Valid options might include:
        'one_family_home', 'two_family_home', 'three_family_home', 'condo', 'coop'
        Adjust or expand as needed.
        """
        valid_types = {
            "one_family_home",
            "two_family_home",
            "three_family_home",
            "condo",
            "coop",
        }
        if htype not in valid_types:
            raise ValueError(
                f"Invalid housing type: {htype}. Must be one of {valid_types}."
            )
        self.features["housing_type"] = htype

    # def get_housing_type(self) -> str:
    #     """
    #     Retrieve the household's housing type.
    #     """
    #     return self.features.get("housing_type", None)

    def property_owners(self):
        """
        Return a list of all members who are property owners.
        """
        return [m for m in self.members if m.get("is_property_owner", False)]

    def owners_total_income(self):
        """
        Return the sum of the (work + investment) income of all property owners.
        """
        owners = self.property_owners()
        return sum(
            o["annual_work_income"] + o["annual_investment_income"] for o in owners
        )

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


def nuclear_family():
    mom = Person.default_person(random_name=False)
    mom["sex"] = SexEnum.FEMALE.value
    mom["age"] = 40
    dad = Person.default_person(random_name=False)
    dad["sex"] = SexEnum.MALE.value
    dad["age"] = 40
    dad["relation"] = RelationEnum.SPOUSE.value
    child = Person.default_person(random_name=False)
    child["sex"] = SexEnum.FEMALE.value
    child["age"] = 10
    child["relation"] = RelationEnum.CHILD.value
    child["dependent"] = True
    return Household([mom, dad, child])


def show_abnormal(member, default_member):
    excluded_keys = ["relation", "age", "name"]
    result = []
    for key in member.features.keys():
        if key in excluded_keys:
            continue
        if member[key] != default_member[key]:
            result.append(f"{key}: {member[key]}")
    return "\n".join(result).strip()


def show_household(hh):
    result = []
    for member in hh["members"]:
        result.append(f"Relation: {member['relation']}")
        result.append(f"Age: {member['age']}")
        if member["relation"] == "self":
            result.append(
                show_abnormal(member, Person.default_person(random_name=False))
            )
        elif member["relation"] == "spouse":
            result.append(
                show_abnormal(member, Person.default_person(random_name=False))
            )
        elif member["relation"] == "child":
            result.append(
                show_abnormal(member, Person.default_person(random_name=False))
            )
        result.append("")  # for spacing between members
    return "\n".join(result).strip()


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
