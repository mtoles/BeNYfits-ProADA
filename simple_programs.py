import argparse
import pandas as pd
from users import Household, Person
from dataset import top_8_programs


def ChildAndDependentCareTaxCredit(hh: Household) -> bool:
    """ "
    To be eligible for the Child and Dependent Care Tax Credit, you should be able to answer yes to the following questions:
    1. Did you pay someone to care for your dependent so that you and your spouse, if filing a joint return, could work or look for work? Qualifying dependents are a child under age 13 at the time of care or a spouse or dependent (of any age) who cannot physically or mentally care for themselves.
    2. Did the dependent live with you for more than half of 2023?
    3. Did you and your spouse, if filing jointly, earn income? These can be from wages, salaries, tips, other taxable employee money, or earnings from self-employment.
    4. If you are married, do both you and your spouse work outside of the home? Or, does one of you work outside of the home while the other is a full-time student, has a disability, or is looking for work?
    """

    def _r1_and_r2(hh: Household) -> bool:
        self_works = hh.user()["works_outside_home"] or hh.user()["looking_for_work"]
        spouse_works = (
            hh.spouse() is None
            or hh.spouse()["works_outside_home"]
            or hh.spouse()["looking_for_work"]
        )
        filing_jointly = hh.user()["filing_jointly"]
        members_with_paid_caregiver = [m for m in hh.members if m["has_paid_caregiver"]]
        qualifying_children = [m for m in members_with_paid_caregiver if m["age"] < 13]
        qualify_adults = [
            m
            for m in members_with_paid_caregiver
            if not m["can_care_for_self"] and m["dependent"]
        ]
        qualifying_family = qualifying_children + qualify_adults
        # drop family members who did not live with the household for more than half of the year
        qualifying_family_lived_with_hh = [
            m for m in qualifying_family if m["duration_more_than_half_prev_year"]
        ]
        if filing_jointly:
            return self_works and spouse_works and qualifying_family_lived_with_hh
        else:
            return bool(self_works and qualifying_family_lived_with_hh)

    def _r3(hh: Household) -> bool:
        return bool(hh.marriage_income() > 0)

    def _r4(hh: Household) -> bool:
        user_not_at_home = (
            hh.user()["works_outside_home"]
            or hh.user()["looking_for_work"]
            or hh.user()["student"]
            or hh.user()["disabled"]
        )
        spouse_not_at_home = (
            hh.spouse() is None
            or hh.spouse()["works_outside_home"]
            or hh.spouse()["looking_for_work"]
            or hh.spouse()["student"]
            or hh.spouse()["disabled"]
        )
        return bool(user_not_at_home and spouse_not_at_home)

    return _r1_and_r2(hh) and _r3(hh) and _r4(hh)


def ComprehensiveAfterSchool(hh: Household) -> bool:
    """
    All NYC students in kindergarten to 12th grade are eligible to enroll in COMPASS programs. Each program may have different age and eligibility requirements.
    """
    children = hh.children()
    for child in children:
        if child["current_school_level"] in list(range(1, 13)) + ["k"]:
            return True
    return False


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
    # map programs to columns
    df_labels = pd.DataFrame(columns=top_8_programs)
    for i in range(len(top_8_programs)):
        df_labels[top_8_programs[i]] = df["labels"].apply(lambda x: x[i])
    # convert pass/fail to True/False
    df_labels = df_labels.applymap(lambda x: x == "pass")
    df_preds = pd.DataFrame(columns=top_8_programs)
    df_agreement = pd.DataFrame(columns=top_8_programs)

    for program in [
        ChildAndDependentCareTaxCredit,
        ComprehensiveAfterSchool,
    ]:
        ### TEST ELIGIBILITY ###
        for i, row in df.iterrows():
            print(f"index: {i}")
            hh = Household.from_dict(row["hh"])
            label = df_labels.loc[i, program.__name__]
            pred = program(hh)
            df_preds.loc[i, program.__name__] = pred

        df_agreement[program.__name__] = (
            df_preds[program.__name__] == df_labels[program.__name__]
        )

        acc = df_agreement[program.__name__].mean()
        print(f"Accuracy for {program.__name__}: {acc:.2f}")
        pass

    pass
