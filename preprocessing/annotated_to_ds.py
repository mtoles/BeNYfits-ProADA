"""
one-off script to convert annotated data to household json dicts because i failed to save the original data properly.
"""

import pandas as pd
from users import (
    default_unemployed,
    default_child,
    default_employed,
    nl_household_profile,
    Person,
    Household,
)
from dataset import top_8_programs
import argparse
from pathlib import PurePath

# ds_path = "dataset/procedural_hh_dataset_0.1.5_annotated.csv"
parser = argparse.ArgumentParser()
parser.add_argument("--ds_path", type=str)
args = parser.parse_args()
ds_path = args.ds_path

df = pd.read_csv(ds_path, header=0).iloc[:50]  # not annotated past 50

hhs = []
evs = []
for i, row in df.iterrows():
    desc = row[0]
    member_strs = desc.split("\n\n")

    members = []
    for member_str in member_strs:
        member_str = member_str.strip()
        feature_kvs = list(x.split(": ") for x in member_str.split("\n"))
        # cast keys to lowercase
        feature_kvs = [(k.lower(), v) for k, v in feature_kvs]
        # cast to int if possible
        for j, (k, v) in enumerate(feature_kvs):
            if v.isnumeric():
                feature_kvs[j] = (k, int(v))
            if v == "True" or v == "False":
                feature_kvs[j] = (k, bool(v))
        non_default_features = dict(feature_kvs)
        relation = non_default_features["relation"]
        if relation == "self":
            member = default_unemployed(random_name=True)
        elif relation == "spouse":
            member = default_unemployed(random_name=True)
        elif relation == "child":
            member = default_child(random_name=True)
        else:
            raise ValueError(f"Unknown relation: {relation}")
        for k, v in non_default_features.items():
            member[k] = v
        # member.update(non_default_features)

        # implied features
        if member["work_income"] > 0:
            member["work_hours_per_week"] = 40
            member["works_outside_home"] = True
        members.append(member)
    # hh = {"members": members}
    hh = Household(members=members)
    # household_schema.validate(hh)
    hh.validate()
    # ev = ["pass" if int(x) else "fail" for x in list(row[1:9])]  # eligibility vector
    # evs.append(ev)
    hhs.append(hh)
df["hh"] = hhs
# df["labels"] = evs
# df["programs"] = [top_8_programs] * len(df)
# for ev in enumerate(evs):
#     for i, x in enumerate(ev):
#         df.loc[ev[0], top_8_programs[i]] = x
df["note"] = [""] * len(df)
df["hh_nl_desc"] = df.apply(nl_household_profile, axis=1)
df = df.drop(columns=["0"])
# cast program values to int
for program in top_8_programs:
    df[program] = df[program].astype(int)
output_path = ".".join(ds_path.split(".")[:-1]) + "_50.jsonl"
# df[["programs", "labels", "hh", "note", "hh_nl_desc"]].to_json(
df.to_json(
    # "dataset/procedural_hh_dataset_0.1.5_annotated_50.jsonl",
    output_path,
    orient="records",
    lines=True,
)
print(f"ds saved to: {output_path}")
