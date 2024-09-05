"""
construct a preference dataset from a results file created by `ranking.py`
"""
import os
import pandas as pd
import argparse
from pathlib import PurePath
from utils import df_to_md

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_json_path",
    help="path to the results json file",
)
args = parser.parse_args()
results_json_path = PurePath(args.results_json_path)

df = pd.read_json(results_json_path)

annot_df = df[["title", "doc_full", "prompt", "ca_bm_pm_output", "ca_ex_pm_output"]]
# output_1: ca_bm_pm_output
# output_2: ca_ex_pm_output
# annot_df["output_1"] = annot_df["ca_bm_pm_output"]
# annot_df["output_2"] = annot_df["ca_ex_pm_output"]
# df = df.drop(columns=["ca_bm_pm_output", "ca_ex_pm_output"])
annot_df = annot_df.rename(
    columns={"ca_bm_pm_output": "output_1", "ca_ex_pm_output": "output_2"}
)

annot_df["annot_preference"] = "__"

annot_output_path = results_json_path._str.split("_")[0] + "_annot.md"
print("output path:", annot_output_path)
# crash if the annot file already exists to avoid overwriting annotations
assert not os.path.exists(annot_output_path)
df_to_md(annot_df, annot_output_path)
# annot_df.to_json(annot_output_path, orient="records", lines=True)


model_df = df[["pref_bm"]]
model_output_path = results_json_path._str.split("_")[0] + "_model.md"
print("output path:", model_output_path)
df_to_md(model_df, model_output_path)
