"""
Procedurally generate households for testing the dataset
Focus on generating households with a diverse set of eligibility
"""

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


def random_members():
    members = []
    ### Generate the User ###
    user = default_unemployed(random_name=False)
    ## 50% chance of high income ##
    if np.random.rand() > 0.5:
        user["work_income"] = 500000
        user["investment_income"] = 500000
    ## 25% chance of middle income ##
    elif np.random.rand() > 0.25:
        user["work_income"] = 40000
    ## 50% chance of special bool ##
    if np.random.rand() > 0.5:
        bools = [
            "receives_hra",
            "receives_ssi",
            "lives_in_temp_housing",
            "name_is_on_lease",
            "lives_in_rent_stabilized_apartment",
            "has_ssn",
            "receives_snap",
        ]
        choice = np.random.choice(bools)
        user[choice] = not user[choice]
    members.append(user)
    ## 50% chance of paying rent ##
    if np.random.rand() > 0.5:
        user["monthly_rent_spending"] = 1000
    ## 50% chance of spouse ##
    if np.random.rand() > 0.5:
        spouse = default_unemployed(random_name=False)
        spouse["relation"] = "spouse"
        ## 50% chance of special case spouse ##
        if np.random.rand() > 0.5:
            bools = [
                "filing_jointly",
                "student",
                "has_paid_caregiver",
                "looking_for_work",
            ]
            choice = np.random.choice(bools)
            spouse[choice] = not spouse[choice]
            if choice == "filing_jointly":
                user["filing_jointly"] = True
            if choice == "has_paid_caregiver":
                spouse["can_care_for_self"] = False
        members.append(spouse)
    ## 50% chance of 3yo child ##
    if np.random.rand() > 0.5:
        child = default_child(random_name=False)
        child["age"] = 3
        members.append(child)
        ## 50% chance of special case child ##
        if np.random.rand() > 0.5:
            bools = [
                "has_paid_caregiver",
            ]
            choice = np.random.choice(bools)
            child[choice] = not child[choice]
    ## 50% chance of 11yo child ##
    if np.random.rand() > 0.5:
        child = default_child(random_name=False)
        child["age"] = 11
        child["current_school_level"] = 5
        members.append(child)
        if np.random.rand() > 0.5:
            bools = [
                "has_paid_caregiver",
            ]
            choice = np.random.choice(bools)
            child[choice] = not child[choice]
    ## 50% chance of 16yo child ##
    if np.random.rand() > 0.5:
        child = default_child(random_name=False)
        child["age"] = 17
        child["current_school_level"] = 12
        members.append(child)
        if np.random.rand() > 0.5:
            bools = [
                "has_paid_caregiver",
            ]
            choice = np.random.choice(bools)
            child[choice] = not child[choice]
    return members


def show_abnormal(member, default_member):
    excluded_keys = ["relation", "age", "name"]
    result = []
    for key in member:
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
            result.append(show_abnormal(member, default_unemployed(random_name=False)))
        elif member["relation"] == "spouse":
            result.append(show_abnormal(member, default_unemployed(random_name=False)))
        elif member["relation"] == "child":
            result.append(show_abnormal(member, default_child(random_name=False)))
        result.append("")  # for spacing between members
    return "\n".join(result).strip()


if __name__ == "__main__":
    duplicates = 0
    unique_hh_strs = set()
    unique_hhs = []
    while len(unique_hh_strs) < 100:
        hh = {"members": random_members()}
        household_schema.validate(hh)
        hh_str = str(hh)
        if hh_str in unique_hh_strs:
            print(f"Duplicate {duplicates}")
            duplicates += 1
            continue
        unique_hh_strs.add(hh_str)
        unique_hhs.append(hh)
    hh_diffs = []
    for hh in unique_hhs:
        hh_diffs.append(show_household(hh))
    df = pd.DataFrame(hh_diffs)
    df.to_csv("procedural_hh_dataset_0.0.2.csv")
    print
