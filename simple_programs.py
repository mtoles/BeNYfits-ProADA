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
        return bool(hh.marriage_work_income() > 0)

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
    for m in hh.members:
        if m["current_school_level"] in list(range(1, 13)) + ["k"]:
            return True
    return False


def EarlyHeadStartPrograms(hh: Household) -> bool:
    """
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

    def _has_toddler(hh: Household) -> bool:
        members = hh.members
        for m in members:
            if m["age"] <= 3:
                return True
        return False

    temp_housing = hh.user()["lives_in_temp_housing"]
    hra = hh.user()["receives_hra"]
    ssi = hh.user()["receives_ssi"]
    foster_care = bool([m for m in hh.members if m["in_foster_care"]])
    hh_income = hh.hh_total_income()
    hh_size = hh.num_members()

    def _income_eligible(hh_income: float, hh_size: int) -> bool:
        return hh_income <= 9440 + 5140 * hh_size

    return _has_toddler(hh) and (
        temp_housing
        or hra
        or ssi
        or foster_care
        or _income_eligible(hh_income, hh_size)
    )


def InfantToddlerPrograms(hh: Household) -> bool:
    """
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

    def _has_infant_toddler(hh: Household) -> bool:
        for m in hh.members:
            if m["age"] <= 5:
                return True
        return False

    def _qualifies(member: Person) -> bool:
        # return member is None or member["work_hours_per_week"] >= 10 or member["student"] or member["enrolled_in_educational_training"] or member["enrolled_in_vocational_training"] or member["looking_for_work"] or member["lives_in_temp_housing"] or member["attending_services_for_domestic_violence"] or member["receiving_treatment_for_substance_abuse"]
        if member is None:
            return True
        reasons = [
            member["work_hours_per_week"] >= 10,
            member["student"],
            member["enrolled_in_educational_training"],
            member["enrolled_in_vocational_training"],
            member["looking_for_work"],
            member["lives_in_temp_housing"],
            member["attending_service_for_domestic_violence"],
            member["receiving_treatment_for_substance_abuse"],
        ]
        return any(reasons)

    def _income_eligible(hh_income: float, hh_size: int) -> bool:
        limit = {
            1: 51610,
            2: 67490,
            3: 83370,
            4: 99250,
            5: 115130,
            6: 131010,
            7: 133987,
            8: 136965,
            9: 139942,
            10: 142920,
            11: 145897,
            12: 148875,
            13: 151852,
            14: 154830,
            15: 157807,
        }
        if hh_size > 15:
            return hh_income <= 157807 + 3022.5 * (hh_size - 15)
        return hh_income <= limit[hh_size]

    has_toddler = _has_infant_toddler(hh)
    user_qualifies = _qualifies(hh.user())
    spouse_qualifies = _qualifies(hh.spouse())
    hh_income_qualifies = _income_eligible(hh.hh_total_income(), hh.num_members())
    return has_toddler and (
        (user_qualifies and spouse_qualifies) or hh_income_qualifies
    )


def ChildTaxCredit(hh: Household) -> bool:
    """To be eligible for the credit in the 2023 tax year, you should meet these requirements:
    1. You earned up to $200,000, and up to $400,000 if you are married filing jointly.
    2. You're claiming a child on your tax return who is 16 or younger. The child must have a Social Security Number (SSN) or Adoption Tax Identification Number (ATIN). The filer may use an SSN or Individual Taxpayer Identification Number (ITIN). Qualifying children must be your child, stepchild, grandchild, eligible foster child, adopted child, sibling, niece, or nephew.
    3. Your child or dependent lived with you for over half of the year in the U.S. and you are claiming them as a dependent on your tax return. Your child cannot provide more than half of their own financial support.
    """

    def _r1(hh: Household) -> bool:
        if hh.user()["filing_jointly"]:
            return hh.marriage_total_income() <= 400000
        return hh.marriage_total_income() <= 200000

    def _r2(hh: Household) -> list[Person]:
        eligible_child = False

        def _qualifies(m: Person) -> bool:
            return (
                m["age"] <= 16
                and (m["has_ssn"] or m["has_atin"])
                and m["dependent"]
                and m["relation"]
                in [
                    "child",
                    "stepchild",
                    "grandchild",
                    "foster_child",
                    "adopted_child",
                    "sibling",
                    "niece_nephew",
                ]
            )

        eligible_children = [m for m in hh.members if _qualifies(m)]

        user_has_ssn = hh.user()["has_ssn"] or hh.user()["has_itin"]
        if user_has_ssn:
            return eligible_children
        else:
            return []

    def _r3(eligible_children: list[Person]) -> list[Person]:
        def _qualifies(m: Person) -> bool:
            return (
                m["duration_more_than_half_prev_year"]
                and m["dependent"]
                and not m["provides_over_half_of_own_financial_support"]
            )

        r3_children = [m for m in eligible_children if _qualifies(m)]
        return r3_children

    r1 = _r1(hh)
    r2_children = _r2(hh)
    r3_children = _r3(r2_children)
    return r1 and bool(r2_children) and bool(r3_children)


def DisabilityRentIncreaseExemption(hh: Household) -> bool:
    """
    To be eligible for DRIE, you should be able to answer "yes" to all of these questions:
    1. Are you 18 years old or older?
    2. Is your name on the lease?
    3. Is your combined household income $50,000 or less in a year?
    4. Do you spend more than one-third of your monthly income on rent?
    5. Do you live in NYC in one of these types of housing: a rent stabilized apartment, a rent controlled apartment, a Mitchell-Lama development, a Limited Dividend development, a redevelopment company development, a Housing Development Fund Company (HDFC) Cooperative development, a Section 213 Cooperative unit, or a rent regulated hotel or single room occupancy unit?
    6. Do you have income from the following benefits: Supplemental Security Income (SSI), Federal Social Security Disability Insurance (SSDI), U.S. Department of Veterans Affairs (VA) disability pension or compensation, or disability-related Medicaid if you received either SSI or SSDI in the past?
    """

    def _r1(hh: Household) -> bool:
        return hh.user()["age"] >= 18

    def _r2(hh: Household) -> bool:
        return hh.user()["name_is_on_lease"]

    def _r3(hh: Household) -> bool:
        return hh.hh_total_income() <= 50000

    def _r4(hh: Household) -> bool:
        income = hh.user().total_income() / 12
        rent = hh.user()["monthly_rent_spending"]
        return rent > income / 3

    def _r5(hh: Household) -> bool:
        return hh.user()["place_of_residence"] == "NYC" and any(
            [
                hh.user()["lives_in_rent_stabilized_apartment"],
                hh.user()["lives_in_rent_controlled_apartment"],
                hh.user()["lives_in_mitchell-lama"],
                hh.user()["lives_in_limited_dividend_development"],
                hh.user()["lives_in_redevelopment_company_development"],
                hh.user()["lives_in_hdfc_development"],
                hh.user()["lives_in_section_213_coop"],
                hh.user()["lives_in_rent_regulated_hotel"],
                hh.user()["lives_in_rent_regulated_single"],
            ]
        )

    def _r6(hh: Household) -> bool:
        return any(
            [
                hh.user()["receives_ssi"],
                hh.user()["receives_ssdi"],
                hh.user()["receives_va_disability"],
                (
                    hh.user()["receives_disability_medicaid"]
                    and hh.user()["has_received_ssi_or_ssdi"]
                ),
            ]
        )

    return _r1(hh) and _r2(hh) and _r3(hh) and _r4(hh) and _r5(hh) and _r6(hh)


def EarnedIncomeTaxCredit(hh: Household):
    """
    To claim the EITC credit on your 2023 tax return, these must apply to you:
    1. You have a valid Social Security Number.
    2. Your income, marital, and parental status in 2023 were one of these: Married with qualifying children and earning up to $63,398, Married with no qualifying children and earning up to $24,210, Single with qualifying children and earning up to $56,838, Single with no qualifying children and earning up to $17,640.
    3. Qualifying children include biological children, stepchildren, foster children, and grandchildren.
    4. If you have no children, the EITC is only available to filers between ages 25 and 64.
    5. Married Filing Separate: A spouse who is not filing a joint return may claim the EITC if you had a qualifying child who lived with you for more than half of the year.
    6. You had investment income of less than $11,000 in 2023.
    """

    def _r1(hh: Household) -> bool:
        return hh.user()["has_ssn"]

    def _r2_r3(hh: Household) -> bool:
        qualifying_children = bool(
            [
                m
                for m in hh.members
                if m["relation"] in ["child", "stepchild", "foster_child", "grandchild"]
            ]
        )
        if hh.user()["filing_jointly"] and qualifying_children:
            return hh.marriage_total_income() <= 63398
        elif hh.user()["filing_jointly"] and not qualifying_children:
            return hh.marriage_total_income() <= 24210
        elif not hh.user()["filing_jointly"] and qualifying_children:
            return hh.marriage_total_income() <= 56838
        else:
            return hh.marriage_total_income() <= 17640

    def _r4(hh: Household) -> bool:
        qualifying_children = bool(
            [
                m
                for m in hh.members
                if m["relation"] in ["child", "stepchild", "foster_child", "grandchild"]
            ]
        )
        if not qualifying_children:
            return 25 <= hh.user()["age"] <= 64
        else:
            return True

    def _r5(hh: Household) -> bool:
        # Check if the child lived with the user long enough
        if hh.user()["filing_jointly"]:
            return True
        else:
            return bool(
                [m for m in hh.members if m["duration_more_than_half_prev_year"]]
            )

    def _r6(hh: Household) -> bool:
        return hh.marriage_investment_income() < 11000

    return _r1(hh) and _r2_r3(hh) and _r4(hh) and _r5(hh) and _r6(hh)


def HeadStart(hh: Household) -> bool:
    """
    Your family qualifies for Head Start if you have a child aged 3-4 and one or more of these apply to you:
    1. You live in temporary housing.
    2. You receive HRA Cash Assistance.
    3. You receive SNAP.
    4. You receive SSI (Supplemental Security Income).
    5. You're enrolling a child who is in foster care.
    6. Your family income falls below the amounts below:
    Household size and yearly income:
    2 - $20,440
    3 - $25,820
    4 - $31,200
    5 - $36,580
    6 - $41,960
    7 - $47,340
    8 - $52,720
    For each additional person, add $5,380.
    The best way to find out if your family is eligible for Head Start is to contact a program directly.
    """

    def _r0(hh: Household) -> bool:
        for m in hh.members:
            if 3 <= m["age"] <= 4:
                return True
        return False

    def _r1(hh: Household) -> bool:
        return hh.user()["lives_in_temp_housing"]

    def _r2(hh: Household) -> bool:
        return hh.user()["receives_hra"]

    def _r3(hh: Household) -> bool:
        return hh.user()["receives_snap"]

    def _r4(hh: Household) -> bool:
        return hh.user()["receives_ssi"]

    def _r5(hh: Household) -> bool:
        return bool([m for m in hh.members if m["in_foster_care"]])

    def _r6(hh: Household) -> bool:
        hh_income = hh.hh_total_income()
        hh_size = hh.num_members()
        return hh_income <= 20440 + 5380 * (hh_size - 2)

    return _r0(hh) and (_r1(hh) or _r2(hh) or _r3(hh) or _r4(hh) or _r5(hh) or _r6(hh))


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
        EarlyHeadStartPrograms,
        InfantToddlerPrograms,
        ChildTaxCredit,
        DisabilityRentIncreaseExemption,
        EarnedIncomeTaxCredit,
        HeadStart,
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
