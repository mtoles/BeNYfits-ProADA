from analysis.dataset_constructor import DatasetConstructor

from users.benefits_programs import *
from users.benefits_programs import BenefitsProgramMeta
import json
from tqdm import tqdm

from utils import import_all_classes
import argparse



# Import all classes except "BaseBenefitsProgram", "BenefitsProgramMeta"
module_name = "users.benefits_programs"
classes = import_all_classes(module_name)

benefits_classes = {}

for class_name, cls in classes.items():
    if class_name not in ["BaseBenefitsProgram", "BenefitsProgramMeta"]:
        benefits_classes[class_name] = cls


parser = argparse.ArgumentParser(description="Construct an edge case dataset")
parser.add_argument("-l", "--limit", type=int, default=100, help="The number of households to generate")
parser.add_argument("-t", "--trials", type=int, default=10000, help="The number of times to attempt to generate a valid household")
parser.add_argument("-o", "--output", type=str, default="edge_case_dataset.jsonl", help="Output path to save the dataset to")
parser.add_argument("-s", "--sampler", type=str, default="diversity", help="The sampler to use")
args = parser.parse_args()

assert args.sampler in ["diversity", "demographic"]
if args.sampler == "demographic":
    pass
else:
    ds_df = DatasetConstructor.fuzz(limit=args.limit, trials=args.trials)

    # households = [hh for hh in ]
    households = ds_df["hh"].tolist()
    households_members = [eval(str(hh)) for hh in households]

    with open(args.output, "w") as fout:

        # for hh, members in tqdm(zip(households, households_members)):
        for i in tqdm(range(len(households))):
            hh = households[i]
            members = households_members[i]
            household_dict = {"hh": {"features": {"members": []}}}

            for member in members:
                household_dict["hh"]["features"]["members"].append({"features": member})

            household_dict["hh_nl_desc"] = hh.nl_household_profile()
            # household_dict["hh_nl_desc_always_include"] = hh.nl_household_profile_always_include()
            # household_dict["hh_nl_desc_always_include"] = hh.members[
            #     0
            # ].nl_person_profile_always_include()
            household_dict["hh_nl_always_include"] = "\n".join([x.nl_person_profile_always_include() for x in hh.members])
            household_dict["note"] = ""
            household_dict["edge_case_programs"] = ds_df.iloc[i]["programs"]

            for program in benefits_classes.values():
                # for program in BenefitsProgramMeta.registry.values(): # doesn't work because of complicated tracing interaction
                household_dict[program.__name__] = program.__call__(hh)

            fout.write(json.dumps(household_dict) + "\n")

