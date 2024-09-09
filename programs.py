from schema import Schema, And, Or, Use, Optional, SchemaError
import networkx as nx
import numpy as np

from users import (
    person_schema,
    household_schema,
    get_random_person,
    get_random_self_person,
)


def missing_key_is_indeterminate(func):
    """
    Decoroator making missing keys return `None`, indicating that the requirement
    cannot be determined with certainty
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError:
            return None

    return wrapper


def check_child_and_dependent_care_tax_credit(hh: dict):
    def _check_member(member):
        G = nx.Graph()
        G.add_node("source")
        G.add_node("b")
        G.add_node("c")
        G.add_node("d")
        G.add_node("sink")
        G.add_edge("source", "a", con=lambda member: member["has_paid_caregiver"])
        G.add_edge("a", "b", con=lambda member: member["age"] < 13)
        G.add_edge("a", "c", con=lambda member: member["relation"] == "spouse")
        G.add_edge("c", "b", con=lambda member: not member["can_care_for_self"])
        G.add_edge("a", "d", con=lambda member: member["dependent"])
        G.add_edge("d", "c", con=lambda member: member["age"] >= 18)
        G.add_edge(
            "b", "sink", con=lambda member: member["duration_more_than_half_prev_year"]
        )
        for e in G.edges():
            try:
                # set the color of the edge
                if G[e[0]][e[1]]["con"](member):
                    G[e[0]][e[1]]["color"] = "pass"
                else:
                    G[e[0]][e[1]]["color"] = "fail"
            except KeyError:
                G[e[0]][e[1]]["color"] = "indeterminate"
        # copy nodes
        G_pass = nx.Graph()
        G_pass.add_nodes_from(G.nodes(data=True))
        G_pass.add_edges_from(
            [(u, v, d) for (u, v, d) in G.edges(data=True) if d["color"] == "pass"]
        )
        if nx.has_path(G_pass, "source", "sink"):
            return "pass"
        G_pass.add_edges_from(
            [
                (u, v, d)
                for (u, v, d) in G.edges(data=True)
                if d["color"] == "indeterminate"
            ]
        )
        if nx.has_path(G_pass, "source", "sink"):
            return "indeterminate"
        return "fail"

    for member in hh["members"]:
        member_colors = [_check_member(member) for member in hh["members"]]
        if "pass" in member_colors:
            return "pass"
        if "indeterminate" in member_colors:
            return "indeterminate"
    # TODO: add requirements 3-4

    return "fail"


if __name__ == "__main__":
    #### TEST CASES ####

    ## PASSING ##
    hh1 = {
        "members": [
            {
                "has_paid_caregiver": True,
                "age": 12,
                "relation": "child",
                "duration_more_than_half_prev_year": True,
            }
        ]
    }
    household_schema.validate(hh1)
    assert check_child_and_dependent_care_tax_credit(hh1) == "pass"

    hh2 = {
        "members": [
            {
                "has_paid_caregiver": True,
                "age": 99,
                "relation": "other_family",
                "can_care_for_self": False,
                "dependent": True,
                "duration_more_than_half_prev_year": True,
            }
        ]
    }
    household_schema.validate(hh2)
    assert check_child_and_dependent_care_tax_credit(hh2) == "pass"

    ## INDETERMINATE ##

    hh3 = {
        "members": [
            {
                # "has_paid_caregiver": True,
                "age": 12,
                "relation": "child",
                "duration_more_than_half_prev_year": True,
            }
        ]
    }
    household_schema.validate(hh3)
    assert check_child_and_dependent_care_tax_credit(hh3) == "indeterminate"

    hh4 = {
        "members": [
            {
                "has_paid_caregiver": True,
                # "age": 99,
                # "relation": "other_family",
                "can_care_for_self": False,
                "dependent": True,
                "duration_more_than_half_prev_year": True,
            }
        ]
    }
    household_schema.validate(hh4)
    assert check_child_and_dependent_care_tax_credit(hh4) == "indeterminate"

    ## FAILING ##
    hh5 = {
        "members": [
            {
                "has_paid_caregiver": True,
                "age": 12,
                "relation": "child",
                "duration_more_than_half_prev_year": False,
            }
        ]
    }
    household_schema.validate(hh5)
    assert check_child_and_dependent_care_tax_credit(hh5) == "fail"

    hh6 = {
        "members": [
            {
                "has_paid_caregiver": False,
                # "age": 99,
                # "relation": "other_family",
                "can_care_for_self": False,
                "dependent": True,
                "duration_more_than_half_prev_year": True,
            }
        ]
    }
    household_schema.validate(hh6)
    assert check_child_and_dependent_care_tax_credit(hh6) == "fail"

    hh7 = {
        "members": [
            {
                "has_paid_caregiver": False,
                # "age": 99,
                # "relation": "other_family",
                "can_care_for_self": True,
                # "dependent": True,
                # "duration_more_than_half_prev_year": True,
            }
        ]
    }
    household_schema.validate(hh7)
    assert check_child_and_dependent_care_tax_credit(hh7) == "fail"

    print("All tests passed")
