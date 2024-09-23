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
    get_random_person,
    get_random_self_person,
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
        G.add_node("r1_temporary")
        G.add_node("r2_hra")
        G.add_node("r3_ssi")
        G.add_node("m_income")

        G.add_edge(
            "source",
            f"r1_temporary",
            con=lambda hh: hh["members"][0]["lives_in_temp_housing"],
        )

        G.add_edge(f"r1_temporary", "sink", con=lambda _: True)

        # Check if anyone in the whole household receives HRA
        G.add_edge(
            "source",
            f"r2_hra",
            con=lambda hh: hh["members"][0]["receives_hra"],
        )

        G.add_edge(f"r2_hra", "sink", con=lambda _: True)

        # Check if anyone in the whole household receives SSI
        G.add_edge(
            "source",
            f"r3_ssi",
            con=lambda hh: hh["members"][0]["receives_ssi"],
        )

        G.add_edge(f"r3_ssi", "sink", con=lambda _: True)

        for i in range(1, n):
            G.add_node(f"r4_child{i}")
            G.add_node(f"r4_foster{i}")
            # Check if anyone is in foster care
            G.add_edge(
                "source",
                f"r4_child{i}",
                con=lambda hh: hh["members"][i]["relation"] == "child",
            )

            G.add_edge(
                "source",
                f"r4_foster{i}",
                con=lambda hh: hh["members"][i]["in_foster_care"],
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

class InfantsAndToddlersPrograms(EligibilityGraph):
    """
    You must have at least one of these approved reasons for care:

    You work 10+ hours per week
    You are in an educational or vocational training program
    You are starting to look for work or have been looking for work for up to 6 months. This includes looking for work while receiving unemployment
    You live in temporary housing
    You are attending services for domestic violence
    You are receiving treatment for substance abuse
    You may qualify if your household income is at or below these amounts:

    +-------------+----------------+---------------+
    | Family Size | Monthly Income | Yearly Income |
    +-------------+----------------+---------------+
    | 1           | $4,301         | $51,610       |
    | 2           | $5,624         | $67,490       |
    | 3           | $6,948         | $83,370       |
    | 4           | $8,271         | $99,250       |
    | 5           | $9,594         | $115,130      |
    | 6           | $10,918        | $131,010      |
    | 7           | $11,166        | $133,987      |
    | 8           | $11,414        | $136,965      |
    | 9           | $11,662        | $139,942      |
    | 10          | $11,910        | $142,920      |
    | 11          | $12,158        | $145,897      |
    | 12          | $12,406        | $148,875      |
    | 13          | $12,654        | $151,852      |
    | 14          | $12,903        | $154,830      |
    | 15          | $13,151        | $157,807      |
    +-------------+----------------+---------------+
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

        G.add_node("r1_work_atleast_10_hours")
        G.add_node("r2_training")
        G.add_node("r3_looking_for_work")
        G.add_node("r4_temporary")
        G.add_node("r5_attending_services_for_domestic_violence")
        G.add_node("r6_receiving_treatment_for_substance_abuse")
        G.add_node("m_income")

        G.add_edge(
            "source",
            f"r1_work_atleast_10_hours",
            con=lambda hh: hh["members"][0]["work_hours_per_week"] >= 10,
        )

        G.add_edge(f"r1_work_atleast_10_hours", "sink", con=lambda _: True)

        G.add_edge(
            "source",
            f"r2_training",
            con=lambda hh: hh["members"][0]["enrolled_in_educational_training"] or hh["members"][0]["enrolled_in_vocational_training"],
        )

        G.add_edge(f"r2_training", "sink", con=lambda _: True)

        G.add_edge(
            "source",
            f"r3_looking_for_work",
            con=lambda hh: hh["members"][0]["looking_for_work"] and (hh["members"][0]["days_looking_for_work"] <= 180),
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
            f"r5_attending_services_for_domestic_violence",
            con=lambda hh: hh["members"][0]["attending_services_for_domestic_violence"],
        )

        G.add_edge(f"r5_attending_services_for_domestic_violence", "sink", con=lambda _: True)

        G.add_edge(
            "source",
            f"r6_receiving_treatment_for_substance_abuse",
            con=lambda hh: hh["members"][0]["receiving_treatment_for_substance_abuse"],
        )

        G.add_edge(f"r6_receiving_treatment_for_substance_abuse", "sink", con=lambda _: True)
        
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

        G.add_edge("source", "m_income", con=check_income)
        G.add_edge("m_income", "sink", con=lambda _: True)

        return G
    

class ChildTaxCredit(EligibilityGraph):
    """
    To be eligible for the credit in the 2023 tax year, you should meet these requirements:

    1. You earned up to $200,000 and up to $400,000 if you are married filing jointly.
    2. You're claiming a child on your tax return who is:
        * 16 or younger
        * The child must have a Social Security Number (SSN) or Adoption Tax Identification Number (ATIN). The filer may use an SSN or Individual Taxpayer Identification Number (ITIN).
    3. Qualifying children must be your child, stepchild, grandchild, eligible foster child, adopted child, sibling, niece, or nephew.
    4. Your child or dependent lived with you for over half of the year in the U.S. and you are claiming them as a dependent on your tax return.
    5. Your child cannot provide more than half of their own financial support.
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

        G.add_node("m1_income")
        G.add_node("filing_jointly")
        G.add_node("not_filing_jointly")
        
        G.add_edge("source",
                   "not_filing_jointly",
                   con=lambda hh: not hh["members"][0]["filing_jointly"])
        G.add_edge("source",
                   "filing_jointly",
                   con=lambda hh: hh["members"][0]["filing_jointly"])
        
        G.add_edge("not_filing_jointly",
                   "m1_income",
                   con=lambda hh: hh["members"][0]["work_income"] + hh["members"][0]["investment_income"] <= 200000)
        
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
                con=lambda hh: hh["members"][0]["work_income"] + hh["members"][0]["investment_income"] + 
                hh["members"][i]["work_income"] + hh["members"][i]["investment_income"] <= 400000,
            )


        for i in range(1, n):
            G.add_node(f"child{i}")
            G.add_node(f"valid_tax_info{i}")
            G.add_node(f"more_than_half{i}")
            G.add_node(f"over_half_financial_support{i}")
            
            G.add_edge(
                "m1_income",
                f"child{i}",
                con=lambda hh: hh["members"][i]["relation"] in ["child", "stepchild", "grandchild", "foster_child", "adopted_child"]
            )

            G.add_edge(
                f"child{i}",
                f"valid_tax_info{i}",
                con=lambda hh: (hh["members"][i]["has_atin"] or hh["members"][i]["has_ssn"]) and (hh["members"][0]["has_ssn"] or hh["members"][0]["has_itin "])
            )

            G.add(f"valid_tax_info{i}",
                  f"more_than_half{i}",
                  con=lambda hh: (hh["members"][i]["duration_more_than_half_prev_year"]))
            
            G.add(f"more_than_half{i}",
                  f"over_half_financial_support{i}",
                  con=lambda hh: (hh["members"][i]["provides_over_half_of_own_financial_support"]))
            
            G.add(f"over_half_financial_support{i}",
                  "sink",
                  con=lambda _: True
                  )

        return G

import networkx as nx
from typing import Literal

class DRIE(EligibilityGraph):
    """
    To be eligible for DRIE, you should meet these requirements:
    
    1. You are 18 years old or older.
    2. Your name is on the lease.
    3. Your combined household income is $50,000 or less in a year.
    4. You spend more than one-third of your monthly income on rent.
    5. You live in NYC in one of these types of housing:
        * rent stabilized apartment
        * rent controlled apartment
        * Mitchell-Lama development
        * Limited Dividend development
        * redevelopment company development
        * Housing Development Fund Company (HDFC) Cooperative development
        * Section 213 Cooperative unit
        * rent regulated hotel or single room occupancy unit.
    6. You have income from one of these benefits:
        * Supplemental Security Income (SSI)
        * Social Security Disability Insurance (SSDI)
        * Veterans Affairs (VA) disability pension or compensation
        * Disability-related Medicaid if you received either SSI or SSDI in the past.
    """
    
    @classmethod
    def make_graph(cls, hh: dict) -> Literal["pass", "fail", "indeterminate"]:
        n = len(hh["members"])
        household_schema.validate(hh)
        G = nx.MultiGraph()
        G.add_node("source")
        G.add_node("sink")

        # Adding nodes for each requirement
        G.add_node("age_check")
        G.add_node("lease_check")
        G.add_node("income_check")
        G.add_node("rent_check")
        G.add_node("housing_type_check")
        G.add_node("benefits_check")

        # Edges connecting the source to various checks
        G.add_edge("source", "age_check", con=lambda hh: hh["members"][0]["age"] >= 18)
        G.add_edge("age_check", "lease_check", con=lambda hh: hh["members"][0]["name_is_on_lease"])
        
        def check_income(hh):
            hh_income = sum(
                hh["members"][i].get("work_income", 0)
                + hh["members"][i].get("investment_income", 0)
                for i in range(n)
            )
            family_size = len(hh["members"])
            return hh_income <= 50000
        
        G.add_edge("lease_check", "income_check", con=check_income)
        G.add_edge("income_check", "rent_check", con=lambda hh: hh["members"][0]["monthly_rent_spending"] > (hh["members"][0]["work_income"] + hh["members"][0]["investment_income"]) / 3)

        # Check for eligible housing type
        G.add_edge(
            "rent_check", "housing_type_check", 
            con=lambda hh: hh["members"][0]["lives_in_rent_stabilized_apartment"] or 
                           hh["members"][0]["lives_in_rent_controlled_apartment"] or 
                           hh["members"][0]["lives_in_mitchell-lama"] or 
                           hh["members"][0]["lives_in_limited_dividend_development"] or 
                           hh["members"][0]["lives_in_redevelopment_company_development"] or 
                           hh["members"][0]["lives_in_hdfc_development"] or 
                           hh["members"][0]["lives_in_section_213_coop"] or 
                           hh["members"][0]["lives_in_rent_regulated_hotel"] or 
                           hh["members"][0]["lives_in_rent_regulated_single"]
        )

        # Check for benefit income
        G.add_edge(
            "housing_type_check", "benefits_check", 
            con=lambda hh: hh["members"][0]["receives_ssi"] or 
                           hh["members"][0]["receives_ssi"] or 
                           hh["members"][0]["receives_ssi"] or 
                           hh["members"][0]["receives_ssi"]
        )

        # Final pass or fail edge
        G.add_edge("benefits_check", "sink", con=lambda _: True)

        return G
    
class EITCCredit(EligibilityGraph):
    """
    To be eligible for the EITC on your 2023 tax return, you must meet these requirements:

    1. You have a valid Social Security Number (SSN).
    2. Your income, marital, and parental status must meet one of these conditions:
        - Married with qualifying children and earning up to $63,398.
        - Married with no qualifying children and earning up to $24,210.
        - Single with qualifying children and earning up to $56,838.
        - Single with no qualifying children and earning up to $17,640.
    3. If you have no children, you must be between ages 25 and 64.
    4. Married Filing Separately can claim the EITC if:
        - You had a qualifying child who lived with you for more than half the year.
        - You were legally separated or lived apart from your spouse for the last 6 months of 2023.
    5. You had investment income of less than $11,000 in 2023.
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

        G.add_node("m1_income")
        G.add_node("has_ssn")
        G.add_node("investment_income")
        G.add_node("age_eligibility")
        G.add_node("marital_status")
        G.add_node("has_qualifying_children")
        G.add_node("does_not_have_qualifying_children")

        # Validate SSN requirement
        G.add_edge("source", "has_ssn", con=lambda hh: hh["members"][0]["has_ssn"])

        # Check if there are qualifying children
        G.add_edge("has_ssn", "has_qualifying_children", con=lambda hh: any(
            member["relation"] in ["child", "stepchild", "grandchild", "foster_child", "adopted_child"] for member in hh["members"]
        ))

        # Check if there are no qualifying children
        G.add_edge("has_ssn", "does_not_have_qualifying_children", con=lambda hh: not any(
            member["relation"] in ["child", "stepchild", "grandchild", "foster_child", "adopted_child"] for member in hh["members"]
        ))

                # Income thresholds for households with qualifying children
        G.add_edge("has_qualifying_children", "m1_income", con=lambda hh: (
            any(member["relation"] == "spouse" for member in hh["members"]) and (
                (hh["members"][0]["filing_jointly"] and hh["members"][0]["work_income"] + hh["members"][0]["investment_income"] <= 63398) or
                (not hh["members"][0]["filing_jointly"] and hh["members"][0]["work_income"] + hh["members"][0]["investment_income"] <= 56838)
            ) or (
                not any(member["relation"] == "spouse" for member in hh["members"]) and
                hh["members"][0]["work_income"] + hh["members"][0]["investment_income"] <= 56838
            )
        ))

        # Income thresholds for households without qualifying children
        G.add_edge("does_not_have_qualifying_children", "m1_income", con=lambda hh: (
            any(member["relation"] == "spouse" for member in hh["members"]) and (
                (hh["members"][0]["filing_jointly"] and hh["members"][0]["work_income"] + hh["members"][0]["investment_income"] <= 24210) or
                (not hh["members"][0]["filing_jointly"] and hh["members"][0]["work_income"] + hh["members"][0]["investment_income"] <= 17640)
            ) or (
                not any(member["relation"] == "spouse" for member in hh["members"]) and
                hh["members"][0]["work_income"] + hh["members"][0]["investment_income"] <= 17640
            )
        ))


        # Validate investment income requirement
        G.add_edge("m1_income", "investment_income", con=lambda hh: hh["members"][0]["investment_income"] < 11000)

        # Validate age range for filers without children
        G.add_edge("investment_income", "age_eligibility", con=lambda hh: (
            hh["members"][0]["relation"] == "self" and
            25 <= hh["members"][0]["age"] <= 64 and
            not any(member["relation"] in ["child", "stepchild", "grandchild", "foster_child", "adopted_child"] for member in hh["members"])
        ))

        # Marital status condition for those filing separately
        G.add_edge("age_eligibility", "marital_status", con=lambda hh: (
            (not hh["members"][0]["filing_jointly"]) and 
            ((hh["members"][0]["relation"] == "self" and hh["members"][0]["dependent"]) or 
            hh["members"][0]["duration_more_than_half_prev_year"])
        ))

        # Connect to sink for valid EITC claim
        G.add_edge("marital_status", "sink", con=lambda _: True)

        return G

class HeadStartProgram(EligibilityGraph):
    """
    Your family may qualify for Head Start if one or more of these apply to you:

    - You live in temporary housing
    - You receive HRA Cash Assistance
    - You receive SNAP
    - You receive SSI (Supplemental Security Income)
    - You're enrolling a child who is in foster care
    
    Your family income falls below the amounts below:

    +----------------+---------------+
    | Household Size | Yearly Income |
    +-------------+------------------+
    |      2         |    20440.00   |
    |      3         |    25820.00   |
    |      4         |    31200.00   |
    |      5         |    36580.00   |
    |      6         |    41960.00   |
    |      7         |    47340.00   |
    |      8         |    52720.00   |
    +-------------+------------------+

    Each Additional Person

    $5,380
    """

    @classmethod
    def make_graph(cls, hh: dict) -> Literal["pass", "fail", "indeterminate"]:
        n = len(hh["members"])
        household_schema.validate(hh)
        G = nx.MultiGraph()
        
        G.add_node("source")
        G.add_node("sink")

        # Temporary housing condition
        G.add_node("temp_housing")
        G.add_edge("source", "temp_housing", con=lambda hh: hh["members"][0]["lives_in_temp_housing"])

        # Receiving HRA Cash Assistance
        G.add_node("hra_cash_assistance")
        G.add_edge("source", "hra_cash_assistance", con=lambda hh: hh["members"][0]["receives_hra"])

        # Receiving SNAP
        G.add_node("snap")
        G.add_edge("source", "snap", con=lambda hh: hh["members"][0]["receives_snap"])

        # Receiving SSI (Supplemental Security Income)
        G.add_node("ssi")
        G.add_edge("source", "ssi", con=lambda hh: hh["members"][0]["receives_ssi"])

        # Foster care condition
        G.add_node("foster_care")
        G.add_edge("source", "foster_care", con=lambda hh: any(member["relation"] == "foster_child" for member in hh["members"]))

        # Income threshold based on household size
        income_thresholds = {2: 20440, 3: 25820, 4: 31200, 5: 36580, 6: 41960, 7: 47340, 8: 52720}
        G.add_node("income_below_threshold")
        G.add_edge("source", "income_below_threshold", con=lambda hh: (
            hh["members"][0]["work_income"] + hh["members"][0]["investment_income"] <= 
            income_thresholds.get(n, 52720 + (n - 8) * 5380)
        ))

        # Combine all conditions for eligibility
        G.add_edge("temp_housing", "sink", con=lambda _: True)
        G.add_edge("hra_cash_assistance", "sink", con=lambda _: True)
        G.add_edge("snap", "sink", con=lambda _: True)
        G.add_edge("ssi", "sink", con=lambda _: True)
        G.add_edge("foster_care", "sink", con=lambda _: True)
        G.add_edge("income_below_threshold", "sink", con=lambda _: True)

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
