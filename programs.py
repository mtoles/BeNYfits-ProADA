from schema import Schema, And, Or, Use, Optional, SchemaError
import networkx as nx
import numpy as np
from typing import List, Dict, Literal, Tuple
import string
import inspect
import matplotlib.pyplot as plt
from dataset import dataset

from users import (
    person_schema,
    household_schema,
    random_person,
    random_self_person,
)


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
    ) -> Tuple[Literal["pass", "fail", "indeterminate"], nx.MultiGraph]:
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
                else:
                    data["color"] = "fail"
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
    def draw_graph(cls, hh: dict):
        G = cls.make_graph(hh)
        # colors
        graph_color, G = cls.evaluate_graph(G, hh)
        pos = nx.spring_layout(G)
        edge_colors = [G[u][v][key]["color"] for u, v, key in G.edges(keys=True)]
        color_map = {
            "pass": "green",
            "fail": "red",
            "indeterminate": "blue",
        }
        edge_colors = [color_map[color] for color in edge_colors]
        # edge labels
        # conditions = nx.get_edge_attributes(G, "con")
        # labels = [inspect.getsource(cond) for cond in conditions.values()]
        # labels = {k: v for k, v in zip(G.edges(), labels)}
        layout = nx.kamada_kawai_layout
        pos = layout(G)

        node_label_offset = {node: (x, y - 0.05) for node, (x, y) in pos.items()}

        nx.draw(
            G,
            pos=layout(G),
            with_labels=False,
            edge_color=edge_colors,
            node_size=20,
        )
        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_labels(
            G, node_label_offset, labels, bbox=dict(alpha=0), font_size=6
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

        plt.savefig("graph.png")
        pass

    @classmethod
    def __call__(cls, profile: dict) -> Literal["pass", "fail", "indeterminate"]:
        G = cls.make_graph(profile)
        graph_color, G = cls.evaluate_graph(G, profile)
        return graph_color


class ChildAndDependentCareTaxCredit(EligibilityGraph):
    """
    To be eligible for the Child and Dependent Care Tax Credit, you should be able to answer yes to these questions:

    1. Did you pay someone to care for your dependent so that you (and your spouse, if filing a joint return) could work or look for work? Qualifying dependents are:
        - a child under age 13 at the time of care;
        - a spouse or adult dependent who cannot physically or mentally care for themselves.
    2. Did the dependent live with you for more than half of 2023?
    3. Did you (and your spouse if you file taxes jointly) earn income? These can be from wages, salaries, tips, other taxable employee money, or earnings from self-employment.
    4. If you are married, do both you and your spouse work outside of the home?
        - Or, do one of you work outside of the home while the other is a full-time student, has a disability, or is looking for work?
    """

    @classmethod
    def make_graph(
        cls,
        hh: dict,
    ) -> Literal["pass", "fail", "indeterminate"]:
        n = len(hh["members"])
        household_schema.validate(hh)
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
    The best way to find out if your family is eligible for Early Head Start is to contact a program directly. Your family may qualify for Early Head Start if at least one of these categories applies to you:
    * You live in temporary housing
    * You receive HRA Cash Assistance
    * You receive SSI (Supplemental Security Insurance)
    * You are enrolling a child who is in foster care
    * You may also qualify if your household income is at or below these amounts:

    +--------------+---------------+
    | Family Size  | Yearly Income |
    +--------------+---------------+
    |      1       |   $14,580     |
    |      2       |   $19,720     |
    |      3       |   $24,860     |
    |      4       |   $30,000     |
    |      5       |   $35,140     |
    |      6       |   $40,280     |
    |      7       |   $45,420     |
    |      8       |   $50,560     |
    +--------------+---------------+
    For each additional person, add: $5,140

    """

    @classmethod
    def make_graph(
        cls,
        hh: dict,
    ) -> Literal["pass", "fail", "indeterminate"]:
        n = len(hh["members"])
        household_schema.validate(hh)
        G = nx.MultiGraph()
        G.add_node("source")
        G.add_node("sink")

        G.add_node("m1")

        # Check all members
        for i in range(n):
            G.add_node(f"r1_temporary{i}")
            G.add_node(f"r2_hra{i}")
            G.add_node(f"r3_ssi{i}")
            G.add_node(f"r4_foster{i}")

            G.add_edge(
                "source",
                f"r1_temporary{i}",
                con=lambda hh, i=i: hh["members"][i]["lives_in_temp_housing"],
            )

            G.add_edge(f"r1_temporary{i}", "sink", con=lambda _: True)

            # Check if anyone in the whole household receives HRA
            G.add_edge(
                "source",
                f"r2_hra{i}",
                con=lambda hh, i=i: hh["members"][i]["receives_hra"],
            )

            G.add_edge(f"r2_hra{i}", "sink", con=lambda _: True)

            # Check if anyone in the whole household receives SSI
            G.add_edge(
                "source",
                f"r3_ssi{i}",
                con=lambda hh, i=i: hh["members"][i]["receives_ssi"],
            )

            G.add_edge(f"r3_ssi{i}", "sink", con=lambda _: True)

            # Check if anyone is in foster care
            G.add_edge(
                "source",
                f"r4_foster{i}",
                con=lambda hh, i=i: hh["members"][i]["in_foster_care"],
            )

            G.add_edge(f"r4_foster{i}", "sink", con=lambda _: True)

        def check_income(hh):
            hh_income = sum(
                hh["members"][i].get("work_income", 0)
                + hh["members"][i].get("investment_income", 0)
                for i in range(n)
            )
            family_size = len(hh["members"])
            return hh_income <= 14580 + (family_size - 1) * 5140

        G.add_edge("source", "m_income", con=check_income)
        G.add_edge("m_income", "sink", con=lambda _: True)

        return G


if __name__ == "__main__":
    for example in dataset:
        for i, program_string in enumerate(example["programs"]):
            program = eval(program_string)
            hh = example["hh"]
            label = example["labels"][i]
            household_schema.validate(hh)
            assert program.__call__(hh) == label

    print("All tests passed")
