from schema import Schema, And, Or, Use, Optional, SchemaError
import networkx as nx
import numpy as np
from typing import List, Dict, Literal, Tuple
import string
import inspect
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from dataset_procedural import show_household
import re
import os

# from dataset import dataset

from users import (
    # person_schema,
    # household_schema,
    Person,
    Household,
    random_person,
    random_self_person,
)


def has_paid_caregiver(hh: dict, i: int) -> bool:
    return hh["members"][i]["has_paid_caregiver"]


class EligibilityGraph:
    @classmethod
    def __call__(cls, hh: dict) -> Literal["pass", "fail", "indeterminate"]:
        raise NotImplementedError

    @classmethod
    def make_graph(cls, hh: dict) -> nx.MultiGraph:
        raise NotImplementedError

    @classmethod
    def evaluate_graph(
        self, G: nx.MultiGraph, profile: dict
    ) -> Tuple[Literal["pass", "fail", "indeterminate"], nx.MultiGraph, pd.DataFrame]:
        # for e in list(G.edges()):
        for u, v, key, data in G.edges(data=True, keys=True):
            try:
                edge_color = data["con"](profile)
                # if "notSpouse" in e[0] or "notSpouse" in e[1]:
                #     print(f"Edge: {e}, color: {edge_color}")
                if edge_color in ["pass", "fail", "indeterminate"]:
                    data["color"] = edge_color
                # set the color of the edge
                if edge_color == True:
                    data["color"] = "pass"
                elif edge_color == False:
                    data["color"] = "fail"
                else:
                    data["color"] = "indeterminate"
            except KeyError:
                data["color"] = "indeterminate"
        # copy nodes
        G_pass = nx.MultiGraph()
        G_pass.add_nodes_from(G.nodes(data=True))
        G_pass.add_edges_from(
            [(u, v, d) for (u, v, d) in G.edges(data=True) if d["color"] == "pass"]
        )
        if nx.has_path(G_pass, "source", "sink"):
            return "pass", G
        G_pass.add_edges_from(
            [
                (u, v, d)
                for (u, v, d) in G.edges(data=True)
                if d["color"] == "indeterminate"
            ]
        )
        if nx.has_path(G_pass, "source", "sink"):
            return "indeterminate", G

        return "fail", G

    @classmethod
    def describe_graph(cls, G):
        # create a description of the graph
        # each line containing nodes, edges, edge description, edge color
        desc_rows = []
        for u, v, key, data in G.edges(data=True, keys=True):
            desc_rows.append(
                {
                    "n1": u,
                    "n2": v,
                    "color": data["color"],
                    "fn": cls._clean_fn_name(data["con"]),
                }
            )
        desc_df = pd.DataFrame(desc_rows)
        return desc_df

    @classmethod
    def _clean_fn_name(cls, fn):
        """
        con=lambda hh, i=i: hh["members"][i]["relation"] == "spouse",\n'
        to
        hh["members"][i]["relation"] == "spouse"
        """
        try:
            fn_str = inspect.getsource(fn)
            fn_str = re.sub(r"con=lambda hh, i=i: ", "", fn_str)
            fn_str = re.sub(r",\n", "", fn_str)
            return fn_str
        except:
            return "fn stringification error"

    @classmethod
    def draw_graph(cls, hh: dict):
        G = cls.make_graph(hh)
        ## COLORS ##
        graph_color, G = cls.evaluate_graph(G, hh)
        pos = nx.spring_layout(G)
        edge_colors = [G[u][v][key]["color"] for u, v, key in G.edges(keys=True)]
        color_map = {
            "pass": "green",
            "fail": "red",
            "indeterminate": "blue",
        }
        edge_colors = [color_map[color] for color in edge_colors]
        ## FUNCTION LABELS ##
        edge_fns = [G[u][v][key]["con"] for u, v, key in G.edges(keys=True)]

        edge_labels = [cls._clean_fn_name(fn) for fn in edge_fns]

        # edge labels
        # conditions = nx.get_edge_attributes(G, "con")
        # labels = [inspect.getsource(cond) for cond in conditions.values()]
        # labels = {k: v for k, v in zip(G.edges(), labels)}
        layout = nx.kamada_kawai_layout
        pos = layout(G)

        node_label_offset = {node: (x, y - 0.05) for node, (x, y) in pos.items()}

        # nx.draw(
        #     G,
        #     pos=layout(G),
        #     with_labels=False,
        #     edge_color=edge_colors,
        #     node_size=20,
        # )

        edge_dict = dict()  # set : list of all parallel edges
        es = list(G.edges())
        # populate edge_dict with unique edges
        for e in list(G.edges()):
            edge_dict[e] = []
        for i, e in enumerate(es):
            edge_dict[e].append(i)
        for e_set, e_list in edge_dict.items():
            for count, edge_idx in enumerate(e_list):
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edge_color=edge_colors[edge_idx],
                    edgelist=[es[edge_idx]],
                    connectionstyle=f"arc3, rad = {0.7/len(e_list)*((count+1)//2*(-1)**(count+1))}",
                )
        # nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
        nx.draw_networkx_nodes(G, pos, node_size=20)

        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_labels(
            G, node_label_offset, labels, bbox=dict(alpha=0), font_size=6
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={e: l for e, l in zip(G.edges(), edge_labels)},
            font_size=2,
            rotate=True,
            bbox=dict(alpha=0),
        )
        x_offset = -0.1  # Horizontal offset (negative moves left)
        y_offset = 0.1  # Vertical offset (positive moves up)
        # for node, (x, y) in pos.items():
        #     plt.text(
        #         # x + x_offset,
        #         # y + y_offset,
        #         s=node,
        #         bbox=dict(facecolor="white", alpha=0),
        #         horizontalalignment="right",
        #         verticalalignment="bottom",
        #     )
        if "graph.png" in os.listdir():
            os.remove("graph.png")
        plt.savefig(
            "graph.png",
            dpi=400,
        )
        pass

    @classmethod
    def __call__(cls, profile: dict) -> Literal["pass", "fail", "indeterminate"]:
        G = cls.make_graph(profile)
        graph_color, G, graph_description = cls.evaluate_graph(G, profile)
        return graph_color


class ChildAndDependentCareTaxCredit(EligibilityGraph):
    """
    1. Child and Dependent Care Tax Credit

    To be eligible for the Child and Dependent Care Tax Credit, you should be able to answer yes to the following questions:
    1. Did you pay someone to care for your dependent so that you and your spouse, if filing a joint return, could work or look for work? Qualifying dependents are a child under age 13 at the time of care or a spouse or adult dependent who cannot physically or mentally care for themselves.
    2. Did the dependent live with you for more than half of 2023?
    3. Did you and your spouse, if filing jointly, earn income? These can be from wages, salaries, tips, other taxable employee money, or earnings from self-employment.
    4. If you are married, do both you and your spouse work outside of the home? Or, does one of you work outside of the home while the other is a full-time student, has a disability, or is looking for work?
    """

    @classmethod
    def make_graph(
        cls,
        hh: dict,
    ) -> Literal["pass", "fail", "indeterminate"]:
        n = len(hh["members"])
        hh.validate()
        G = nx.MultiGraph()
        G.add_node("source")
        G.add_node("sink")

        G.add_node("m1")
        for i in range(1, n):
            # Requirement 1
            G.add_node(f"r1_caregiver{i}")
            G.add_node(f"r1_dependent{i}")
            G.add_node(f"r2_livedwith{i}")
            G.add_edge(
                "source",
                f"r1_caregiver{i}",
                con=lambda hh, i=i: hh["members"][i]["has_paid_caregiver"],
                # con=lambda hh, i=i: has_paid_caregiver(hh, i),
            )
            G.add_edge(
                f"r1_caregiver{i}",
                "m1",
                con=lambda hh, i=i: hh["members"][i]["age"] < 13,
            )
            G.add_edge(
                f"r1_caregiver{i}",
                f"r1_dependent{i}",
                con=lambda hh, i=i: hh["members"][i]["dependent"],
            )
            G.add_edge(
                f"r1_caregiver{i}",
                f"r1_dependent{i}",
                con=lambda hh, i=i: hh["members"][i]["relation"] == "spouse",
            )
            G.add_edge(
                f"r1_dependent{i}",
                f"r2_livedwith{i}",
                con=lambda hh, i=i: hh["members"][i]["can_care_for_self"],
            )
            # Requirement 2
            G.add_edge(
                f"r2_livedwith{i}",
                "m1",
                con=lambda hh, i=i: hh["members"][i][
                    "duration_more_than_half_prev_year"
                ],
            )

        # Requirement 3
        G.add_node("m3")
        G.add_edge("m1", "m3", con=lambda hh: hh["members"][0]["work_income"] > 0)
        for i in range(1, n):
            G.add_node(f"r3_joint{i}")
            G.add_node(f"r3_spouse{i}")
            G.add_edge(
                "m1", f"r3_joint{i}", con=lambda hh: hh["members"][0]["filing_jointly"]
            )
            G.add_edge(
                f"r3_joint{i}",
                f"r3_spouse{i}",
                con=lambda hh, i=i: hh["members"][i]["relation"] == "spouse",
            )
            G.add_edge(
                f"r3_spouse{i}",
                "m3",
                con=lambda hh, i=i: hh["members"][i]["work_income"] > 0,
            )

        # Requirement 4
        # User works outside the home, spouse works or fits other criteria
        for i in range(1, n):
            G.add_node(f"spouseworksoutside{i}")
            G.add_node(f"userworksoutside{i}")
            G.add_node(f"userother{i}")

        for i in range(1, n):
            G.add_edge(
                "m3",
                f"userworksoutside{i}",
                con=lambda hh: hh["members"][0]["works_outside_home"],
            )
            G.add_edge(
                "m3",
                f"userworksoutside{i}",
                con=lambda hh, i=i: hh["members"][i]["works_outside_home"],
            )
            G.add_edge(
                "m3",
                f"userworksoutside{i}",
                con=lambda hh, i=i: hh["members"][i]["student"],
            )
            G.add_edge(
                "m3",
                f"userworksoutside{i}",
                con=lambda hh, i=i: hh["members"][i]["disabled"],
            )
            G.add_edge(
                "m3",
                f"userworksoutside{i}",
                con=lambda hh, i=i: hh["members"][i]["looking_for_work"],
            )
            G.add_edge(
                f"userworksoutside{i}",
                "sink",
                con=lambda hh, i=i: hh["members"][i]["relation"] == "spouse",
            )
        # Spouse works outside the home, user works or fits other criteria
        for i in range(1, n):
            G.add_edge(
                "m3",
                f"spouseworksoutside{i}",
                con=lambda hh, i=i: hh["members"][i]["works_outside_home"],
            )
            G.add_edge(
                f"spouseworksoutside{i}",
                f"userother{i}",
                con=lambda hh, i=i: hh["members"][i]["relation"] == "spouse",
            )
            G.add_edge(
                f"userother{i}",
                "sink",
                con=lambda hh: hh["members"][0]["works_outside_home"],
            )
            G.add_edge(
                f"userother{i}", "sink", con=lambda hh: hh["members"][0]["student"]
            )
            G.add_edge(
                f"userother{i}", "sink", con=lambda hh: hh["members"][0]["disabled"]
            )
            G.add_edge(
                f"userother{i}",
                "sink",
                con=lambda hh: hh["members"][0]["looking_for_work"],
            )

        # Requirement 4 if no spouse
        def _get_node_name(i):
            if i == 0:
                return "m3"
            if i == n:
                return "sink"
            return f"r4_notSpouse{i}"

        def is_not_spouse(hh, i):
            return hh["members"][i]["relation"] != "spouse"

        for i in range(0, n):
            if _get_node_name(i) not in G.nodes() and i != n:
                G.add_node(_get_node_name(i))
            G.add_edge(
                _get_node_name(i),
                _get_node_name(i + 1),
                con=lambda hh, i=i: hh["members"][i]["relation"] != "spouse",
            )

        return G


class EarlyHeadStartPrograms(EligibilityGraph):
    """
    2. Early Head Start Programs

    The best way to find out if your family is eligible for Early Head Start is to contact a program directly. Your family qualifies for Early Head Start if your child is age 3 or younger and at least one of these categories applies to you:
    1. You live in temporary housing.
    2. You receive HRA Cash Assistance.
    3. You receive SSI (Supplemental Security Insurance).
    4. You are enrolling a child who is in foster care.
    5. If your household income is at or below these amounts:
    Family size and yearly income:
    1 - $14,580
    2 - $19,720
    3 - $24,860
    4 - $30,000
    5 - $35,140
    6 - $40,280
    7 - $45,420
    8 - $50,560
    For each additional person, add $5,140.

    """

    @classmethod
    def make_graph(
        cls,
        hh: dict,
    ) -> Literal["pass", "fail", "indeterminate"]:
        n = len(hh["members"])
        hh.validate()
        G = nx.MultiGraph()
        G.add_node("source")
        G.add_node("sink")

        G.add_node("m1")

        # Check all members
        # G.add_node("r1_temporary")
        # G.add_node("r2_hra")
        # G.add_node("r3_ssi")
        # G.add_node("m_income")

        G.add_edge(
            "source", "m1", con=lambda hh: any([c["age"] <= 3 for c in hh.members])
        )

        G.add_edge(
            "m1",
            f"sink",
            con=lambda hh: hh["members"][0]["lives_in_temp_housing"],
        )

        # G.add_edge(f"r1_temporary", "sink", con=lambda _: True)

        # Check if anyone in the whole household receives HRA
        G.add_edge(
            "m1",
            f"sink",
            con=lambda hh: hh["members"][0]["receives_hra"],
        )

        # G.add_edge(f"r2_hra", "sink", con=lambda _: True)

        # Check if anyone in the whole household receives SSI
        G.add_edge(
            "m1",
            f"sink",
            con=lambda hh: hh["members"][0]["receives_ssi"],
        )

        # G.add_edge(f"r3_ssi", "sink", con=lambda _: True)

        # for i in range(1, n):
        # G.add_node(f"r4_child{i}")
        # G.add_node(f"r4_foster{i}")
        # Check if anyone is in foster care
        # G.add_edge(
        #     "m1",
        #     f"r4_child{i}",
        #     con=lambda hh: hh["members"][i]["relation"] == "child",
        # )

        # G.add_edge(
        #     "m1",
        #     f"r4_foster{i}",
        #     con=lambda hh: hh["members"][i]["in_foster_care"],
        # )

        # G.add_edge(f"r4_foster{i}", "sink", con=lambda _: True)
        G.add_edge(
            "m1",
            "sink",
            con=lambda hh: any([c for c in hh.members if c["in_foster_care"]]),
        )

        def check_income(hh):
            hh_income = sum(
                hh["members"][i].get("work_income", 0)
                + hh["members"][i].get("investment_income", 0)
                for i in range(n)
            )
            family_size = len(hh["members"])
            return hh_income <= 14580 + (family_size - 1) * 5140

        G.add_edge("m1", "sink", con=check_income)
        # G.add_edge("m_income", "sink", con=lambda _: True)

        return G


class InfantToddlerPrograms(EligibilityGraph):
    """
    3. Infant/Toddler Programs

    You must have a child age 5 or younger and both parents have at least one of these approved reasons for care:
    1. You work 10+ hours per week.
    2. You are in an educational or vocational training program.
    3. You are starting to look for work or have been looking for work for up to 6 months, including looking for work while receiving unemployment.
    4. You live in temporary housing.
    5. You are attending services for domestic violence.
    6. You are receiving treatment for substance abuse.
    7. Your household income is at or below these amounts:
    Family size, monthly income, and yearly income:
    1 - $4,301 (monthly), $51,610 (yearly)
    2 - $5,624 (monthly), $67,490 (yearly)
    3 - $6,948 (monthly), $83,370 (yearly)
    4 - $8,271 (monthly), $99,250 (yearly)
    5 - $9,594 (monthly), $115,130 (yearly)
    6 - $10,918 (monthly), $131,010 (yearly)
    7 - $11,166 (monthly), $133,987 (yearly)
    8 - $11,414 (monthly), $136,965 (yearly)
    9 - $11,662 (monthly), $139,942 (yearly)
    10 - $11,910 (monthly), $142,920 (yearly)
    11 - $12,158 (monthly), $145,897 (yearly)
    12 - $12,406 (monthly), $148,875 (yearly)
    13 - $12,654 (monthly), $151,852 (yearly)
    14 - $12,903 (monthly), $154,830 (yearly)
    15 - $13,151 (monthly), $157,807 (yearly)
    """

    @classmethod
    def make_graph(
        cls,
        hh: dict,
    ) -> Literal["pass", "fail", "indeterminate"]:
        n = len(hh["members"])
        hh.validate()
        G = nx.MultiGraph()
        G.add_node("source")
        G.add_node("sink")

        G.add_node("m1")
        G.add_node("spouse")
        G.add_edge(
            "source", "m1", con=lambda hh: any([c["age"] <= 5 for c in hh.members])
        )

        G.add_edge(
            "m1",
            "spouse",
            con=lambda hh: hh["members"][0]["work_hours_per_week"] >= 10,
        )
        G.add_edge(
            "m1",
            "spouse",
            con=lambda hh: hh["members"][0]["enrolled_in_educational_training"],
        )
        G.add_edge(
            "m1",
            "spouse",
            con=lambda hh: hh["members"][0]["enrolled_in_vocational_training"],
        )

        G.add_node("self_looking_for_work")

        G.add_edge(  # self is looking for work
            "m1",
            "self_looking_for_work",
            con=lambda hh: hh["members"][0]["looking_for_work"],
        )
        G.add_edge(  # self has been looking <= 180 days
            "self_looking_for_work",
            "spouse",
            con=lambda hh: hh["members"][0]["days_looking_for_work"] <= 180,
        )
        G.add_edge(
            "m1",
            f"spouse",
            con=lambda hh: hh["members"][0]["lives_in_temp_housing"],
        )
        G.add_edge(
            "m1",
            f"spouse",
            con=lambda hh: hh["members"][0]["attending_service_for_domestic_violence"],
        )
        G.add_edge(
            "m1",
            f"spouse",
            con=lambda hh: hh["members"][0]["receiving_treatment_for_substance_abuse"],
        )
        ### spouse ###
        G.add_edge(
            "spouse",
            "sink",
            con=lambda hh: hh.spouse() is None
            or hh.spouse()["work_hours_per_week"] >= 10,
        )
        G.add_edge(
            "spouse",
            "sink",
            con=lambda hh: hh.spouse() is None
            or hh.spouse()["enrolled_in_educational_training"],
        )
        G.add_edge(
            "spouse",
            "sink",
            con=lambda hh: hh.spouse() is None
            or hh.spouse()["enrolled_in_vocational_training"],
        )
        G.add_edge(  # spouse is looking for work
            "spouse",
            "spouse_looking_for_work",
            con=lambda hh: hh.spouse() is None or hh.spouse()["looking_for_work"],
        )
        G.add_edge(  # spouse has been looking <= 180 days
            "spouse_looking_for_work",
            "sink",
            con=lambda hh: hh.spouse() is None
            or hh.spouse()["days_looking_for_work"] <= 180,
        )
        G.add_edge(
            "spouse",
            "sink",
            con=lambda hh: hh.spouse() is None or hh.spouse()["lives_in_temp_housing"],
        )
        G.add_edge(
            "spouse",
            "sink",
            con=lambda hh: hh.spouse() is None
            or hh.spouse()["attending_service_for_domestic_violence"],
        )
        G.add_edge(
            "spouse",
            "sink",
            con=lambda hh: hh.spouse() is None
            or hh.spouse()["receiving_treatment_for_substance_abuse"],
        )
        # G.add_edge( # spouse is looking for work or spouse does not exist
        #     "m1",
        #     "looking_for_work",
        #     con=lambda hh: hh.spouse() is None or hh.spouse()["looking_for_work"]
        # )
        # G.add_edge( # spouse has been looking <= 180 days
        #     "looking_for_work",
        #     "sink",
        #     con=lambda hh: hh.spouse() is None or hh.spouse()["days_looking_for_work"] <= 180
        # )
        # G.add_edge(
        #     "m1",
        #     "sink",
        #     con=lambda hh: hh["members"][0]["looking_for_work"]
        #     and (hh["members"][0]["days_looking_for_work"] <= 180),
        # )

        # G.add_edge(f"r3_looking_for_work", "sink", con=lambda _: True)

        # G.add_edge(f"r4_temporary", "sink", con=lambda _: True)

        # G.add_edge(
        #     f"r5_attending_service_for_domestic_violence", "sink", con=lambda _: True
        # )

        # G.add_edge(
        #     f"r6_receiving_treatment_for_substance_abuse", "sink", con=lambda _: True
        # )

        def check_income(hh):
            hh_income = sum(
                hh["members"][i].get("work_income", 0)
                + hh["members"][i].get("investment_income", 0)
                for i in range(n)
            )
            family_size = len(hh["members"])
            if family_size < 6:
                return hh_income <= 12 * (1323.4 * family_size + 2977.6)

            return hh_income <= 12 * (248.11 * family_size + 9429.34)

        G.add_edge("m1", "sink", con=check_income)
        # G.add_edge("m_income", "sink", con=lambda _: True)

        return G


class ChildTaxCredit(EligibilityGraph):
    """
    4. Child Tax Credit

    To be eligible for the credit in the 2023 tax year, you should meet these requirements:
    1. You earned up to $200,000, and up to $400,000 if you are married filing jointly.
    2. You're claiming a child on your tax return who is 16 or younger. The child must have a Social Security Number (SSN) or Adoption Tax Identification Number (ATIN). The filer may use an SSN or Individual Taxpayer Identification Number (ITIN). Qualifying children must be your child, stepchild, grandchild, eligible foster child, adopted child, sibling, niece, or nephew.
    3. Your child or dependent lived with you for over half of the year in the U.S. and you are claiming them as a dependent on your tax return. Your child cannot provide more than half of their own financial support.
    """

    @classmethod
    def make_graph(
        cls,
        hh: dict,
    ) -> Literal["pass", "fail", "indeterminate"]:
        n = len(hh["members"])
        hh.validate()
        G = nx.MultiGraph()
        G.add_node("source")
        G.add_node("sink")

        G.add_node("m1_income")
        G.add_node("filing_jointly")
        G.add_node("not_filing_jointly")

        G.add_edge(
            "source",
            "not_filing_jointly",
            con=lambda hh: not hh["members"][0]["filing_jointly"],
        )
        G.add_edge(
            "source",
            "filing_jointly",
            con=lambda hh: hh["members"][0]["filing_jointly"],
        )

        G.add_edge(
            "not_filing_jointly",
            "m1_income",
            con=lambda hh: hh["members"][0]["work_income"]
            + hh["members"][0]["investment_income"]
            <= 200000,
        )

        for i in range(1, n):
            G.add_node(f"spouse{i}")

            # Check if anyone is in foster care
            G.add_edge(
                "filing_jointly",
                f"spouse{i}",
                con=lambda hh: hh["members"][i]["relation"] == "spouse",
            )

            G.add_edge(
                f"spouse{i}",
                f"m1_income",
                con=lambda hh: hh["members"][0]["work_income"]
                + hh["members"][0]["investment_income"]
                + hh["members"][i]["work_income"]
                + hh["members"][i]["investment_income"]
                <= 400000,
            )

            G.add_edge(f"m1_income", "sink", con=lambda _: True)

        G.add_edge(f"r1_work_atleast_10_hours", "sink", con=lambda _: True)

        G.add_edge(
            "source",
            f"r2_training",
            con=lambda hh: hh["members"][0]["enrolled_in_educational_training"]
            or hh["members"][0]["enrolled_in_vocational_training"],
        )

        G.add_edge(f"r2_training", "sink", con=lambda _: True)

        G.add_edge(
            "source",
            f"r3_looking_for_work",
            con=lambda hh: hh["members"][0]["looking_for_work"]
            and (hh["members"][0]["days_looking_for_work"] <= 180),
        )

        G.add_edge(f"r3_looking_for_work", "sink", con=lambda _: True)

        G.add_edge(
            "source",
            f"r4_temporary",
            con=lambda hh: hh["members"][0]["lives_in_temp_housing"],
        )

        G.add_edge(f"r4_temporary", "sink", con=lambda _: True)

        G.add_edge(
            "source",
            f"r5_attending_service_for_domestic_violence",
            con=lambda hh: hh["members"][0]["attending_service_for_domestic_violence"],
        )

        G.add_edge(
            f"r5_attending_service_for_domestic_violence", "sink", con=lambda _: True
        )

        G.add_edge(
            "source",
            f"r6_receiving_treatment_for_substance_abuse",
            con=lambda hh: hh["members"][0]["receiving_treatment_for_substance_abuse"],
        )

        G.add_edge(
            f"r6_receiving_treatment_for_substance_abuse", "sink", con=lambda _: True
        )

        def check_income(hh):
            hh_income = sum(
                hh["members"][i].get("work_income", 0)
                + hh["members"][i].get("investment_income", 0)
                for i in range(n)
            )
            family_size = len(hh["members"])
            if family_size < 6:
                return hh_income <= 12 * (1323.4 * family_size + 2977.6)

            return hh_income <= 12 * (248.11 * family_size + 9429.34)

        G.add_edge("m1", "sink", con=check_income)
        # G.add_edge("m_income", "sink", con=lambda _: True)

        return G


class ComprehensiveAfterSchool(EligibilityGraph):
    """
    8. Comprehensive After School System of NYC

    All NYC students in kindergarten to 12th grade are eligible to enroll in COMPASS programs. Each program may have different age and eligibility requirements.
    """

    @classmethod
    def make_graph(cls, hh: dict) -> Literal["pass", "fail", "indeterminate"]:
        n = len(hh["members"])
        hh.validate()
        G = nx.MultiGraph()
        G.add_node("source")
        G.add_node("sink")
        for i in range(0, n):
            G.add_edge(
                "source",
                "sink",
                con=lambda hh, i=i: hh["members"][i]["current_school_level"]
                in ["pk", "k", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            )

        return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the eligibility programs")
    parser.add_argument(
        "--dataset_path",
        default="dataset/procedural_hh_dataset_0.1.8_annotated_50.jsonl",
        help="Path to the chat history or benefits description",
    )
    parser.add_argument(
        "--ds_shift",
        default=0,
        type=int,
        help="Shift the dataset by n rows",
    )
    args = parser.parse_args()

    df = pd.read_json(args.dataset_path, lines=True)
    # move the first n rows to the end
    df = pd.concat([df[args.ds_shift :], df[: args.ds_shift]], ignore_index=True)
    predictionss = []
    agreementss = []
    for i, row in df.iterrows():
        # hh = row["hh"]
        # members = Person.
        hh = Household.from_dict(row["hh"])
        labels = row["labels"]

        print(f"Index: {i}")
        predictions = []
        agreements = []
        for j, program_string in enumerate(row["programs"]):
            try:
                program = eval(program_string)
                label = row["labels"][j]
                hh.validate()
                G = program.make_graph(hh)
                prediction, G = program.evaluate_graph(G, hh)
                predictions.append(prediction)
            except NameError:
                prediction = None
            predictions.append(None)
            if prediction is not None:
                # agreement.append(prediction == label)
                a = prediction == label
                if a or not program_string in [
                    # "ChildAndDependentCareTaxCredit",
                    # "EarlyHeadStartPrograms",
                    "InfantToddlerPrograms",
                    # "ComprehensiveAfterSchool",
                ]:  # testing
                    pass
                else:
                    # print graph
                    program = eval(program_string)
                    print(f"Program: {program_string}")
                    print(f"Annotation: {label}")
                    print(f"Prediction: {prediction}")
                    print(show_household(hh))
                    G = program.make_graph(hh)
                    program.evaluate_graph(G, hh)

                    program.draw_graph(hh)
                    desc_df = program.describe_graph(G)
                    print
                    print("=================")
                agreements.append(a)

            else:
                agreements.append(None)
        predictionss.append(predictions)

        agreementss.append(agreements)
    df["predictions"] = predictionss

    # check agreement between predictions and labels
    df["agreements"] = agreementss
    print(df["agreements"])

    # for example in dataset:
    #     for i, program_string in enumerate(example["programs"]):
    #         program = eval(program_string)
    #         hh = example["hh"]
    #         label = example["labels"][i]
    # hh.validate()
    #         assert program.__call__(hh) == label

    # print("All tests passed")
    print
