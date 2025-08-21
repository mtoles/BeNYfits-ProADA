from analysis.dataset_constructor import DatasetConstructor

from users.benefits_programs import *
import json
from tqdm import tqdm
from users.users import Household

from utils import import_all_classes
import argparse
import numpy as np
import pandas as pd


np.random.seed(0)


def generate_representative_dataset(limit, benefits_classes):
    households = []
    # n_programs = [np.random.randint(1, len(benefits_classes)) for _ in range(limit)]
    # weights = np.array([min(1 / (2**n), 1 / (2**8)) for n in range(len(benefits_classes))])
    # n_programs = [np.random.choice(
    #     list(range(1, len(benefits_classes) + 1)), p=weights / sum(weights)
    # ) for _ in range(limit)]
    # program_lists = []
    eligibilities = []

    N_REPEATS = 3  # add each program to the ds N_REPEATS times
    N_PROGRAMS = len(benefits_classes)
    N_HOUSEHOLDS = 25
    program_lists = [[] for _ in range(N_HOUSEHOLDS)]
    for _ in range(N_REPEATS):
        for i in range(N_PROGRAMS):
            target = np.random.randint(0, N_HOUSEHOLDS)
            p = list(benefits_classes.keys())[i]
            program_lists[target].append(p)
    for pl in program_lists:
        assert len(pl) != 0

    for i in range(limit):
        hh = Household.demographic_hh()
        hh = hh.conform()
        program_list = program_lists[i]
        eligibility = {p: benefits_classes[p].__call__(hh) for p in program_list}

        households.append(hh)
        eligibilities.append(eligibility)

    ds_df = pd.DataFrame(
        {"hh": households, "programs": program_lists, "eligibility": eligibilities}
    )
    return ds_df


# Import all classes except "BaseBenefitsProgram", "BenefitsProgramMeta"
module_name = "users.benefits_programs"
classes = import_all_classes(module_name)

benefits_classes = {}

for class_name, cls in classes.items():
    if class_name not in ["BaseBenefitsProgram", "BenefitsProgramMeta"]:
        benefits_classes[class_name] = cls


parser = argparse.ArgumentParser(description="Construct an edge case dataset")
parser.add_argument(
    "-l", "--limit", type=int, default=100, help="The number of households to generate"
)
parser.add_argument(
    "-t",
    "--trials",
    type=int,
    default=10000,
    help="The number of times to attempt to generate a valid household",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="dataset.jsonl",
    help="Output path to save the dataset to",
)
parser.add_argument(
    "-s", "--sampler", type=str, default="diversity", help="The sampler to use"
)
args = parser.parse_args()

assert args.sampler in ["diversity", "demographic"]
if args.sampler == "demographic":
    ds_df = generate_demographic_dataset(
        limit=args.limit, benefits_classes=benefits_classes
    )
    output_path = "demographic_" + args.output
else:
    ds_df = DatasetConstructor.fuzz(limit=args.limit, trials=args.trials)
    output_path = "edge_case_" + args.output

# households = [hh for hh in ]
households = ds_df["hh"].tolist()
households_members = [eval(str(hh)) for hh in households]

with open(output_path, "w") as fout:

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
        household_dict["hh_nl_always_include"] = "\n".join(
            [x.nl_person_profile_always_include() for x in hh.members]
        )
        household_dict["note"] = ""
        household_dict["edge_case_programs"] = ds_df.iloc[i]["programs"]

        for program in benefits_classes.values():
            # for program in BenefitsProgramMeta.registry.values(): # doesn't work because of complicated tracing interaction
            household_dict[program.__name__] = program.__call__(hh)

        fout.write(json.dumps(household_dict) + "\n")
