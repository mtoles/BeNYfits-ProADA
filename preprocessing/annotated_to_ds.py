"""
one-off script to convert annotated data to household json dicts because i failed to save the original data properly.
"""

import pandas as pd
from users import (
    default_unemployed,
    default_child,
    default_employed,
    nl_household_profile,
    household_schema,
)
from dataset import top_8_programs

ds_path = "dataset/procedural_hh_dataset_0.1.3_annotated.csv"

df = pd.read_csv(ds_path, header=0).iloc[:50]  # not annotated past 50

hhs = []
evs = []
for i, row in df.iterrows():
    desc = row[0]
    member_strs = desc.split("\n\n")

    members = []
    for member_str in member_strs:
        member_str = member_str.lower().strip()
        feature_kvs = [x.split(": ") for x in member_str.split("\n")]
        # cast to int if possible
        for j, (k, v) in enumerate(feature_kvs):
            if v.isnumeric():
                feature_kvs[j] = (k, int(v))
        non_default_features = dict(feature_kvs)
        relation = non_default_features["relation"]
        if relation == "self":
            member = default_unemployed(random_name=False)
        elif relation == "spouse":
            member = default_unemployed(random_name=False)
        elif relation == "child":
            member = default_child(random_name=False)
        else:
            raise ValueError(f"Unknown relation: {relation}")
        member.update(non_default_features)
        members.append(member)
    hh = {"members": members}
    household_schema.validate(hh)
    ev = ["pass" if int(x) else "fail" for x in list(row[1:9])]  # eligibility vector
    evs.append(ev)
    hhs.append(hh)
df["hh"] = hhs
df["labels"] = evs
df["programs"] = [top_8_programs] * len(df)
df["note"] = [""] * len(df)
df["hh_nl_desc"] = df.apply(nl_household_profile, axis=1)
df[["programs", "labels", "hh", "note", "hh_nl_desc"]].to_json(
    "dataset/procedural_hh_dataset_0.1.3_annotated_50.jsonl", orient="records", lines=True
)
print(hh)
