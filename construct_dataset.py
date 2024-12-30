from analysis.dataset_constructor import DatasetConstructor

from users.benefits_programs import *
from users.benefits_programs import BenefitsProgramMeta
import json
from tqdm import tqdm


households = [hh for hh in DatasetConstructor.fuzz(2, 100)]
households_members = [eval(str(hh)) for hh in households]

with open("edge_case_dataset.jsonl", "w") as fout:

    for hh, members in tqdm(zip(households, households_members)):
        household_dict = {"hh": {"features": {"members": []}}}

        for member in members:
            household_dict["hh"]["features"]["members"].append({"features": member})

        household_dict["hh_nl_desc"] = hh.nl_household_profile()
        household_dict["note"] = ""

        for program in [
            ChildAndDependentCareTaxCredit,
            EarlyHeadStartPrograms,
            InfantToddlerPrograms,
            ComprehensiveAfterSchool,
            InfantToddlerPrograms,
            ChildTaxCredit,
            DisabilityRentIncreaseExemption,
            EarnedIncomeTaxCredit,
            HeadStart,
        ]:
            # for program in BenefitsProgramMeta.registry.values(): # doesn't work because of complicated tracing interaction
            household_dict[program.__name__] = program.__call__(hh)

        fout.write(json.dumps(household_dict) + "\n")

