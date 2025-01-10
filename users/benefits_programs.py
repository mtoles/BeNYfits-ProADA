import random
from names import get_full_name
from users.users import Household, Person
from users import user_features
from users.user_features import (
    HousingEnum,
    RelationEnum,
    SexEnum,
    PlaceOfResidenceEnum,
    CitizenshipEnum,
    EducationLevelEnum,
    GradeLevelEnum,
)
import numpy as np


# from users.users import Household, Person # don't import this to avoid circular logic
class BenefitsProgramMeta(type):
    registry = {}

    def __new__(cls, name, bases, attrs):
        attrs["name"] = name
        new_program = super().__new__(cls, name, bases, attrs)
        if name != "BaseBenefitsProgram":
            cls.registry[name] = new_program
        return new_program


class BaseBenefitsProgram(metaclass=BenefitsProgramMeta):
    pass


def get_random_household_input():
    """
    Fetches a random household to be used for all programs.
    """
    members = []

    has_spouse = random.choice([True, False])
    num_children = random.randint(0, 2)
    num_adults = random.randint(0, 1)

    ### user
    user = Person.random_person(is_self=True)
    members = [user]
    if has_spouse:
        spouse = Person.random_person(is_self=False)
        spouse["relation"] = "spouse"
        members.append(spouse)
    for _ in range(num_children):
        child = Person.random_person(is_self=False)
        child["age"] = random.randint(0, 18)
        child["relation"] = np.random.choice(
            [
                RelationEnum.CHILD.value,
                RelationEnum.ADOPTED_CHILD.value,
                RelationEnum.STEPCHILD.value,
                RelationEnum.GRANDCHILD.value,
                RelationEnum.FOSTER_CHILD.value,
            ]
        )
        members.append(child)
    for _ in range(num_adults):
        adult = Person.random_person(is_self=False)
        adult["age"] = random.randint(18, 100)
        adult["relation"] = np.random.choice(
            [
                RelationEnum.SIBLING.value,
                RelationEnum.OTHER_FAMILY.value,
                RelationEnum.OTHER_NON_FAMILY.value,
            ]
        )
        members.append(adult)

    hh = Household(members)
    for i in range(len(hh.members)):
        for attr in user_features.BasePersonAttr.registry.keys():
            cls = user_features.BasePersonAttr.registry[attr]
            hh.members[i][attr] = cls.conform(cls, hh, i, hh.members[i][attr])
    hh.validate()
    return hh


class ChildAndDependentCareTaxCredit(BaseBenefitsProgram):
    """ "
    To be eligible for the Child and Dependent Care Tax Credit, you should be able to answer yes to the following questions:
    1. Did you pay someone to care for your dependent so that you and your spouse, if filing a joint return, could work or look for work? Qualifying dependents are a child under age 13 at the time of care or a spouse or dependent (of any age) who cannot physically or mentally care for themselves.
    2. Did the dependent live with you for more than half of 2023?
    3. Did you and your spouse, if filing jointly, earn income? These can be from wages, salaries, tips, other taxable employee money, or earnings from self-employment.
    4. If you are married, do both you and your spouse work outside of the home? Or, does one of you work outside of the home while the other is a full-time student, has a disability, or is looking for work?
    """

    @staticmethod
    def __call__(hh) -> bool:

        def _r1_and_r2(hh) -> bool:

            self_works = False

            if hh.user()["works_outside_home"]:
                self_works = True

            elif hh.user()["days_looking_for_work"]> 0: 
                self_works = True

            spouse_works = False

            if hh.spouse() is None:
                spouse_works = True

            elif hh.spouse()["works_outside_home"]:
                spouse_works = True

            elif hh.spouse()["days_looking_for_work"] > 0:
                spouse_works = True

            filing_jointly = False

            if hh.user()["filing_jointly"]:
                filing_jointly = True

            members_with_paid_caregiver = []

            for m in hh.members:
                if m["has_paid_caregiver"]:
                    members_with_paid_caregiver.append(m)

            qualifying_children = []

            for m in members_with_paid_caregiver:
                if m["age"] < 13:
                    qualifying_children.append(m)

            qualify_adults = []

            for m in members_with_paid_caregiver:
                if not m["can_care_for_self"] and m["dependent"]:
                    qualify_adults.append(m)

            qualifying_family = qualifying_children + qualify_adults
            # drop family members who did not live with the household for more than half of the year

            qualifying_family_lived_with_hh = []

            for m in qualifying_family:
                if m["duration_more_than_half_prev_year"]:
                    qualifying_family_lived_with_hh.append(m)

            if filing_jointly:
                if self_works:
                    return True
                if spouse_works:
                    return True
                if qualifying_family_lived_with_hh:
                    return True

                return False
            else:
                if self_works and qualifying_family_lived_with_hh:
                    return True
                return False

        def _r3(hh) -> bool:
            if hh.marriage_annual_work_income() > 0:
                return True
            return False

        def _r4(hh) -> bool:
            user_not_at_home = False

            if hh.user()["works_outside_home"]:
                user_not_at_home = True
            elif hh.user()["days_looking_for_work"] > 0:
                user_not_at_home = True
            elif hh.user()["current_school_level"] != GradeLevelEnum.NONE.value:
                user_not_at_home = True
            elif hh.user()["disabled"]:
                user_not_at_home = True

            spouse_not_at_home = False

            if hh.spouse() is None:
                spouse_not_at_home = True
            elif hh.spouse()["works_outside_home"]:
                spouse_not_at_home = True
            elif hh.spouse()["current_school_level"] != GradeLevelEnum.NONE.value:
                spouse_not_at_home = True
            elif hh.spouse()["disabled"]:
                spouse_not_at_home = True

            if user_not_at_home and spouse_not_at_home:
                return True

            return False

        if not _r1_and_r2(hh):
            return False
        elif not _r3(hh):
            return False
        elif not _r4(hh):
            return False

        return True


# def ComprehensiveAfterSchool(hh) -> bool:
class ComprehensiveAfterSchool(BaseBenefitsProgram):
    @staticmethod
    def __call__(hh) -> bool:
        """
        All NYC students in kindergarten to 12th grade are eligible to enroll in COMPASS programs. Each program may have different age and eligibility requirements.
        """
        for m in hh.members:
            if m["current_school_level"] in list(range(1, 13)) + ["k"]:
                return True
        return False
    @staticmethod
    def unit_tests(hh):
        hh1=Household([Person.default_unemployed(is_self=True)])
        hh1.members[0]["current_school_level"] = 1 # in first grade
        hh1=Household([Person.default_unemployed(is_self=True)])
        hh1.members[0]["current_school_level"] = 1 # in first grade
        return [
            {
                hh1: True,
            },

        ]


# def EarlyHeadStartPrograms(hh) -> bool:
class EarlyHeadStartPrograms(BaseBenefitsProgram):
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

    @staticmethod
    def __call__(hh) -> bool:

        def _has_toddler(hh) -> bool:
            members = hh.members
            for m in members:
                if m["age"] <= 3:
                    return True
            return False

        # temp_housing = hh.user()["lives_in_temp_housing"]
        temp_housing = hh.user()["housing_type"] == HousingEnum.TEMPORARY_HOUSING.value
        hra = hh.user()["receives_hra"]
        ssi = hh.user()["receives_ssi"]
        foster_care = bool([m for m in hh.members if m["in_foster_care"]])
        hh_income = hh.hh_total_income()
        hh_size = hh.num_members()

        def _income_eligible(hh_income: float, hh_size: int) -> bool:
            if hh_income <= 9440 + 5140 * hh_size:
                return True
            return False

        secondary_conditions = False

        if temp_housing:
            secondary_conditions = True
        elif hra:
            secondary_conditions = True
        elif ssi:
            secondary_conditions = True
        elif _income_eligible(hh_income, hh_size):
            secondary_conditions = True

        if not _has_toddler(hh):
            return False
        elif not secondary_conditions:
            return False

        return True


# def InfantToddlerPrograms(hh) -> bool:
class InfantToddlerPrograms(BaseBenefitsProgram):
    @staticmethod
    def __call__(hh):
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

        def _has_infant_toddler(hh) -> bool:
            for m in hh.members:
                if m["age"] <= 5:
                    return True
            return False

        def _qualifies(member) -> bool:
            # return member is None or member["work_hours_per_week"] >= 10 or member["student"] or member["enrolled_in_educational_training"] or member["enrolled_in_vocational_training"] or member["looking_for_work"] or member["lives_in_temp_housing"] or member["attending_services_for_domestic_violence"] or member["receiving_treatment_for_substance_abuse"]
            if member is None:
                return True

            if member["work_hours_per_week"] >= 10:
                return True
            elif member["current_school_level"] != GradeLevelEnum.NONE.value:
                return True
            elif member["enrolled_in_educational_training"]:
                return True
            elif member["enrolled_in_vocational_training"]:
                return True
            elif member["days_looking_for_work"] > 0:
                return True
            elif member["housing_type"] == HousingEnum.TEMPORARY_HOUSING.value:
                return True
            elif member["attending_service_for_domestic_violence"]:
                return True
            elif member["receiving_treatment_for_substance_abuse"]:
                return True

            return False

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
                if hh_income <= 157807 + 3022.5 * (hh_size - 15):
                    return True
                return False
            if hh_income <= limit[hh_size]:
                return True
            return False

        has_toddler = _has_infant_toddler(hh)
        user_qualifies = _qualifies(hh.user())
        spouse_qualifies = _qualifies(hh.spouse())
        hh_income_qualifies = _income_eligible(hh.hh_total_income(), hh.num_members())

        if not has_toddler:
            return False
        if hh_income_qualifies:
            return True
        if not user_qualifies:
            return False
        if not spouse_qualifies:
            return False

        return True


# def ChildTaxCredit(hh) -> bool:
class ChildTaxCredit(BaseBenefitsProgram):
    """To be eligible for the credit in the 2023 tax year, you should meet these requirements:
    def __call__(hh):
    1. You earned up to $200,000, and up to $400,000 if you are married filing jointly.
    2. You're claiming a child on your tax return who is 16 or younger. The child must have a Social Security Number (SSN) or Adoption Tax Identification Number (ATIN). The filer may use an SSN or Individual Taxpayer Identification Number (ITIN). Qualifying children must be your child, stepchild, grandchild, eligible foster child, adopted child, sibling, niece, or nephew.
    3. Your child or dependent lived with you for over half of the year in the U.S. and you are claiming them as a dependent on your tax return. Your child cannot provide more than half of their own financial support.
    """

    @staticmethod
    def __call__(hh):
        def _r1(hh) -> bool:
            if hh.user()["filing_jointly"]:
                if hh.marriage_total_income() <= 400000:
                    return True
                return False
            if hh.marriage_total_income() <= 200000:
                return True
            return False

        def _r2(hh) -> list:
            eligible_child = False

            def _qualifies(m) -> bool:

                if not m["age"] <= 16:
                    return False
                elif not (m["has_ssn"] or m["has_atin"]):
                    return False
                elif not m["dependent"]:
                    return False
                elif not m["relation"]:
                    return False
                return True

            eligible_children = []

            for m in hh.members:
                if _qualifies(m):
                    eligible_children.append(m)

            user_has_ssn = False

            if hh.user()["has_ssn"]:
                user_has_ssn = True
            elif hh.user()["has_itin"]:
                user_has_ssn = True

            if user_has_ssn:
                return eligible_children
            else:
                return []

        def _r3(eligible_children: list) -> list:
            def _qualifies(m) -> bool:

                if not m["duration_more_than_half_prev_year"]:
                    return False
                elif not m["dependent"]:
                    return False
                elif m["provides_over_half_of_own_financial_support"]:
                    return False

                return True

            r3_children = []

            for m in eligible_children:
                if _qualifies(m):
                    r3_children.append(m)
            return r3_children

        r1 = _r1(hh)
        r2_children = _r2(hh)
        r3_children = _r3(r2_children)

        if not r1:
            return False
        elif not bool(r2_children):
            return False
        elif not bool(r3_children):
            return False

        return True


# def DisabilityRentIncreaseExemption(hh) -> bool:
class DisabilityRentIncreaseExemption(BaseBenefitsProgram):
    """
    To be eligible for DRIE, you should be able to answer "yes" to all of these questions:
    1. Are you 18 years old or older?
    2. Is your name on the lease?
    3. Is your combined household income $50,000 or less in a year?
    4. Do you spend more than one-third of your monthly income on rent?
    5. Do you live in NYC in one of these types of housing: a rent stabilized apartment, a rent controlled apartment, a Mitchell-Lama development, a Limited Dividend development, a redevelopment company development, a Housing Development Fund Company (HDFC) Cooperative development, a Section 213 Cooperative unit, or a rent regulated hotel or single room occupancy unit?
    6. Do you have income from the following benefits: Supplemental Security Income (SSI), Federal Social Security Disability Insurance (SSDI), U.S. Department of Veterans Affairs (VA) disability pension or compensation, or disability-related Medicaid if you received either SSI or SSDI in the past?
    """

    @staticmethod
    def __call__(hh):
        def _r1(hh) -> bool:
            if hh.user()["age"] >= 18:
                return True
            return False

        def _r2(hh) -> bool:
            if hh.user()["name_is_on_lease"]:
                return True
            return False

        def _r3(hh) -> bool:
            if hh.hh_total_income() <= 50000:
                return True
            return False

        def _r4(hh) -> bool:
            income = hh.user().total_income() / 12
            rent = hh.user()["monthly_rent_spending"]
            if rent > income / 3:
                return True
            return False

        def _r5(hh) -> bool:
            if not hh.user()["place_of_residence"] == "NYC":
                return False
            elif (
                hh.user()["housing_type"] == HousingEnum.RENT_STABILIZED_APARTMENT.value
            ):
                return True
            elif (
                hh.user()["housing_type"] == HousingEnum.RENT_CONTROLLED_APARTMENT.value
            ):
                return True
            elif (
                hh.user()["housing_type"] == HousingEnum.MITCHELL_LAMA_DEVELOPMENT.value
            ):
                return True
            elif (
                hh.user()["housing_type"]
                == HousingEnum.LIMITED_DIVIDEND_DEVELOPMENT.value
            ):
                return True
            elif (
                hh.user()["housing_type"]
                == HousingEnum.REDEVELOPMENT_COMPANY_DEVELOPMENT.value
            ):
                return True
            elif hh.user()["housing_type"] == HousingEnum.HDFC_DEVELOPMENT.value:
                return True
            elif hh.user()["housing_type"] == HousingEnum.SECTION_213_COOP.value:
                return True
            elif hh.user()["housing_type"] == HousingEnum.RENT_REGULATED_HOTEL.value:
                return True
            elif (
                hh.user()["housing_type"]
                == HousingEnum.RENT_REGULATED_SINGLE_ROOM_OCCUPANCY.value
            ):
                return True

            return False

        def _r6(hh) -> bool:
            if hh.user()["receives_ssi"]:
                return True
            elif hh.user()["receives_ssdi"]:
                return True
            elif hh.user()["receives_va_disability"]:
                return True
            elif hh.user()["receives_disability_medicaid"]:
                if hh.user()["has_received_ssi_or_ssdi"]:
                    return True

            return False

        if not _r1(hh):
            return False
        elif not _r2(hh):
            return False
        elif not _r3(hh):
            return False
        elif not _r4(hh):
            return False
        elif not _r5(hh):
            return False
        elif not _r6(hh):
            return False

        return True


# def EarnedIncomeTaxCredit(hh):
class EarnedIncomeTaxCredit(BaseBenefitsProgram):
    """
    To claim the EITC credit on your 2023 tax return, these must apply to you:
    1. You have a valid Social Security Number.
    2. Your income, marital, and parental status in 2023 were one of these: Married with qualifying children and earning up to $63,398, Married with no qualifying children and earning up to $24,210, Single with qualifying children and earning up to $56,838, Single with no qualifying children and earning up to $17,640.
    3. Qualifying children include biological children, stepchildren, foster children, and grandchildren.
    4. If you have no children, the EITC is only available to filers between ages 25 and 64.
    5. Married Filing Separate: A spouse who is not filing a joint return may claim the EITC if you had a qualifying child who lived with you for more than half of the year.
    6. You had investment income of less than $11,000 in 2023.
    """

    @staticmethod
    def __call__(hh):

        def _r1(hh) -> bool:
            if hh.user()["has_ssn"]:
                return True
            return False

        def _r2_r3(hh) -> bool:
            qualifying_children = []

            for m in hh.members:
                if m["relation"] in [
                    # "child",
                    # "stepchild",
                    # "foster_child",
                    # "grandchild",
                    RelationEnum.CHILD,
                    RelationEnum.STEPCHILD,
                    RelationEnum.FOSTER_CHILD,
                    RelationEnum.GRANDCHILD,
                    RelationEnum.ADOPTED_CHILD,
                ]:
                    qualifying_children.append(m)

            qualifying_children = bool(qualifying_children)

            if hh.user()["filing_jointly"]:
                if qualifying_children:
                    if hh.marriage_total_income() <= 63398:
                        return True
                    return False
            elif hh.user()["filing_jointly"]:
                if not qualifying_children:
                    if hh.marriage_total_income() <= 24210:
                        return True
                    return False
            elif not hh.user()["filing_jointly"]:
                if qualifying_children:
                    if hh.marriage_total_income() <= 56838:
                        return True
                    return False
            else:
                if hh.marriage_total_income() <= 17640:
                    return True
                return False

        def _r4(hh) -> bool:
            qualifying_children = []

            for m in hh.members:
                if m["relation"] in [
                    # "child",
                    # "stepchild",
                    # "foster_child",
                    # "grandchild",
                    RelationEnum.CHILD,
                    RelationEnum.STEPCHILD,
                    RelationEnum.FOSTER_CHILD,
                    RelationEnum.GRANDCHILD,
                    RelationEnum.ADOPTED_CHILD,
                ]:
                    qualifying_children.append(m)

            qualifying_children = bool(qualifying_children)

            if not qualifying_children:
                if 25 <= hh.user()["age"] <= 64:
                    return True
                return False
            else:
                return True

        def _r5(hh) -> bool:
            # Check if the child lived with the user long enough
            if hh.user()["filing_jointly"]:
                return True
            else:
                duration_more_than_half_prev_year_members = []

                for m in hh.members:
                    if m["duration_more_than_half_prev_year"]:
                        duration_more_than_half_prev_year_members.append(m)
                if bool(duration_more_than_half_prev_year_members):
                    return True
                return False

        def _r6(hh) -> bool:
            if hh.marriage_annual_investment_income() < 11000:
                return True
            return False

        if not _r1(hh):
            return False
        elif not _r2_r3(hh):
            return False
        elif not _r4(hh):
            return False
        elif not _r5(hh):
            return False
        elif not _r6(hh):
            return False

        return True


# def EarlyHeadStartPrograms(hh) -> bool:
class EarlyHeadStartPrograms(BaseBenefitsProgram):
    @staticmethod
    def __call__(hh) -> bool:
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

        def _has_toddler(hh) -> bool:
            members = hh.members
            for m in members:
                if m["age"] <= 3:
                    return True
            return False

        # temp_housing = hh.user()["lives_in_temp_housing"]
        temp_housing = hh.user()["housing_type"] == HousingEnum.TEMPORARY_HOUSING.value
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


# def HeadStart(hh) -> bool:
class HeadStart(BaseBenefitsProgram):
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

    @staticmethod
    def __call__(hh):
        def _r0(hh) -> bool:
            for m in hh.members:
                if 3 <= m["age"] <= 4:
                    return True
            return False

        def _r1(hh) -> bool:
            if hh.user()["lives_in_temp_housing"]:
                return True
            return False

        def _r2(hh) -> bool:
            if hh.user()["receives_hra"]:
                return True
            return False

        def _r3(hh) -> bool:
            if hh.user()["receives_snap"]:
                return True
            return False

        def _r4(hh) -> bool:
            if hh.user()["receives_ssi"]:
                return True
            return False

        def _r5(hh) -> bool:
            if bool([m for m in hh.members if m["in_foster_care"]]):
                return True
            return False

        def _r6(hh) -> bool:
            hh_income = hh.hh_total_income()
            hh_size = hh.num_members()
            if hh_income <= 20440 + 5380 * (hh_size - 2):
                return True
            return False

        if not _r0(hh):
            return False
        elif _r1(hh):
            return True
        elif _r2(hh):
            return True
        elif _r3(hh):
            return True
        elif _r4(hh):
            return True
        elif _r5(hh):
            return True
        elif _r6(hh):
            return True

        return False


class ComprehensiveAfterSchool(BaseBenefitsProgram):
    @staticmethod
    def __call__(hh) -> bool:
        """
        All NYC students in kindergarten to 12th grade are eligible to enroll in COMPASS programs. Each program may have different age and eligibility requirements.
        """
        for m in hh.members:
            if m["current_school_level"] in list(range(1, 13)) + ["k"]:
                return True
        return False


# TODO - RATTAN
# Cash Assistance --- Very vague --- No clear requirements

# Health Insurance Assistance - Vague and Not clear


class SchoolTaxReliefProgram(BaseBenefitsProgram):
    """
    Eligibility for the School Tax Relief (STAR) Program.

    To be eligible for STAR, you should be a homeowner of one of these types of housing:
    - a house
    - a condo
    - a cooperative apartment
    - a manufactured home
    - a farmhouse
    - a mixed-use property, including apartment buildings (only the owner-occupied portion is eligible)

    There are two types of STAR benefits:

    **1. Basic STAR**
       - **Age:** No age restriction
       - **Primary residence:** An owner must live on the property as their primary residence.
       - **Income:**
         - The total income of only the owners and their spouses who live at the property must be:
           - $500,000 or less for the credit
           - $250,000 or less for the exemption (you cannot apply for the exemption anymore but you can restore it if you got it in 2015-16 but lost the benefit later.)

    **2. Enhanced STAR**
       - **Age:**
         - All owners must be 65 or older as of December 31 of the year of the exemption.
         - However, only one owner needs to be 65 or older if the property is jointly owned by only a married couple or only siblings.
       - **Primary residence:**
         - At least one owner who's 65 or older must live on the property as their primary residence.
       - **Income:**
         - Total income of all owners and resident spouses or registered domestic partners must be $98,700 or less.

    *Income eligibility for the 2024 STAR credit is based on your federal or state income tax return from the 2022 tax year.*
    """

    @staticmethod
    def __call__(hh):
        def _is_eligible_homeowner(hh):
            # Check if the user owns an eligible type of housing
            eligible_housing_types = [
                # "house",
                # "condo",
                # "cooperative_apartment",
                # "manufactured_home",
                # "farmhouse",
                # "mixed_use_property",
                HousingEnum.HOUSE_2B.value,
                HousingEnum.CONDO.value,
                HousingEnum.COOPERATIVE_APARTMENT.value,
                HousingEnum.MANUFACTURED_HOME.value,
                HousingEnum.FARMHOUSE.value,
                HousingEnum.MIXED_USE_PROPERTY.value,
            ]
            return hh.user().get("housing_type") in eligible_housing_types

        def _basic_star_primary_residence(hh):
            # An owner must live on the property as their primary residence
            return hh.user().get("primary_residence", False)

        def _basic_star_income(hh):
            # Total income of owners and their spouses who live at the property must be <= $500,000
            owners = [hh.user()]
            spouse = hh.spouse()
            if spouse and spouse.get("primary_residence", False):
                owners.append(spouse)
            total_income = sum(owner.total_income() for owner in owners)
            return total_income <= 500000

        def _enhanced_star_age(hh):
            # All owners must be 65 or older, unless jointly owned by only a married couple or only siblings
            owners = [hh.user()]
            co_owners = hh.features.get("co_owners", [])
            owners.extend(co_owners)

            if all(owner["age"] >= 65 for owner in owners):
                return True
            elif len(owners) == 2:
                if hh.user().get("filing_jointly") and any(
                    owner["age"] >= 65 for owner in owners
                ):
                    # Jointly owned by a married couple
                    return True
                elif all(
                    owner["relation"] == RelationEnum.SIBLING.value for owner in owners
                ) and any(owner["age"] >= 65 for owner in owners):
                    # Jointly owned by siblings
                    return True
            return False

        def _enhanced_star_primary_residence(hh):
            # At least one owner who's 65 or older must live on the property as their primary residence
            owners = [hh.user()]
            co_owners = hh.features.get("co_owners", [])
            owners.extend(co_owners)
            for owner in owners:
                if owner["age"] >= 65 and owner.get("primary_residence", False):
                    return True
            return False

        def _enhanced_star_income(hh):
            # Total income of all owners and resident spouses or registered domestic partners must be <= $98,700
            owners = [hh.user()]
            co_owners = hh.features.get("co_owners", [])
            owners.extend(co_owners)
            resident_spouses = []
            for owner in owners:
                if owner.get("primary_residence", False):
                    # Include resident spouses or registered domestic partners
                    spouse = (
                        hh.spouse()
                        if owner["relation"] == RelationEnum.SELF.value
                        else None
                    )
                    if spouse and spouse.get("primary_residence", False):
                        resident_spouses.append(spouse)
            total_income = sum(
                owner.total_income() for owner in owners + resident_spouses
            )
            return total_income <= 98700

        # Check eligibility for Basic STAR
        basic_star_eligible = (
            _is_eligible_homeowner(hh)
            and _basic_star_primary_residence(hh)
            and _basic_star_income(hh)
        )

        # Check eligibility for Enhanced STAR
        enhanced_star_eligible = (
            _is_eligible_homeowner(hh)
            and _enhanced_star_age(hh)
            and _enhanced_star_primary_residence(hh)
            and _enhanced_star_income(hh)
        )

        return basic_star_eligible or enhanced_star_eligible


class Section8HousingChoiceVoucherProgram(BaseBenefitsProgram):
    """
    Eligibility for the Section 8/HCV programs is primarily based on how much your family earns and family size.

    Household Size | Annual Income
    -------------- | -------------
    1              | $54,350
    2              | $62,150
    3              | $69,900
    4              | $77,650
    5              | $83,850
    6              | $90,050
    7              | $96,300
    8              | $102,500
    """

    @staticmethod
    def __call__(hh):
        def _income_eligible(hh) -> bool:
            income_limits = {
                1: 54350,
                2: 62150,
                3: 69900,
                4: 77650,
                5: 83850,
                6: 90050,
                7: 96300,
                8: 102500,
            }
            hh_size = hh.num_members()
            if hh_size <= 8:
                income_limit = income_limits[hh_size]
            else:
                # For household sizes over 8, increase limit by an approximate amount per additional member
                additional_members = hh_size - 8
                income_limit = income_limits[8] + (additional_members * 6200)

            return hh.hh_total_income() <= income_limit

        return _income_eligible(hh)

        # ChatGPT Link - https://chatgpt.com/c/6748084a-bf98-8002-83e0-cc8e4d80c137
        hh_size = hh.num_members()
        total_income = hh.hh_total_income()

        # Determine income limit for the household size
        if hh_size in income_limits:
            income_limit = income_limits[hh_size]
        else:
            # For households larger than 8, extrapolate the income limit
            extra_members = hh_size - 8
            additional_limit = extra_members * (income_limits[8] - income_limits[7])
            income_limit = income_limits[8] + additional_limit

        # Check if the household's total income is within the limit
        return total_income <= income_limit


class SeniorCitizenHomeownersExemption(BaseBenefitsProgram):
    """
    Senior Citizen Homeowners’ Exemption (SCHE)

    Eligibility Requirements (simplified):
      1. The property must be a one-, two-, or three-family home, condo, or coop apartment.
      2. All owners of the property must be 65 or older.
         Exception: If the property is owned by spouses or siblings only, then at least one must be 65+.
      3. All owners must live on the property as their primary residence.
      4. The combined income for all owners must be less than or equal to $58,399.
      5. The owners must have owned the property for at least 12 consecutive months before filing
         (unless they already received SCHE on a previously owned property).
    """

    @staticmethod
    def __call__(hh) -> bool:
        # 1. Check housing type using the new Household getter
        valid_housing_types = {
            "one_family_home",
            "two_family_home",
            "three_family_home",
            "condo",
            "coop",
        }
        housing_type = hh.get_housing_type()  # e.g., "one_family_home", "condo", etc.
        if housing_type not in valid_housing_types:
            return False

        # 2. Identify the property owners using the new Household helper
        owners = (
            hh.property_owners()
        )  # Returns all members where "is_property_owner" = True
        if not owners:
            # If there are no identified owners, the exemption cannot apply
            return False

        # Helper function to determine if two people are spouses or siblings
        def are_spouses_or_siblings(p1, p2):
            rel1, rel2 = p1.get("relation", ""), p2.get("relation", "")
            # Basic logic: If either reports "spouse" or "sibling", treat them accordingly
            return (RelationEnum.SPOUSE.value in [rel1, rel2]) or (
                RelationEnum.SIBLING.value in [rel1, rel2]
            )

        # 2A. Check the age requirement
        if len(owners) == 1:
            # If there is only one owner, that owner must be at least 65
            if owners[0].get("age", 0) < 65:
                return False

        elif len(owners) == 2:
            # If there are exactly two owners, check if they are spouses or siblings
            if are_spouses_or_siblings(owners[0], owners[1]):
                # Only one must be 65+
                if not any(o.get("age", 0) >= 65 for o in owners):
                    return False
            else:
                # Otherwise, both must be 65+
                if not all(o.get("age", 0) >= 65 for o in owners):
                    return False

        else:
            # If more than two owners, all must be 65+
            if not all(o.get("age", 0) >= 65 for o in owners):
                return False

        # 3. Primary residence requirement (all owners must live in the property they own)
        for o in owners:
            if not o["primary_residence"]:
                return False

        # 4. Check combined income using new Household helper
        combined_income = (
            hh.owners_total_income()
        )  # Sums work + investment incomes of owners
        if combined_income > 58399:
            return False

        # 5. Ownership duration (>=12 months) or previously had SCHE
        for o in owners:
            print(o["months_owned_property"])
            if not o["had_previous_sche"] and o["months_owned_property"] < 12:
                return False

        return True


class FamilyPlanningBenefitProgram(BaseBenefitsProgram):
    """To be eligible, you need to meet these criteria:

    You're a New York State resident
    You're a U.S. Citizen, National, or lawfully present.
    Your income equal to or less than these program income requirements:
    Family size
    Income in a month
    1   $2,799
    2   $3,799
    3   $4,799
    4   $5,598
    5   $6,798
    6   $7,798
    7   $8,798
    8   $9,798
    For each additional person, add:
    $1000"""

    @staticmethod
    def __call__(self, hh):
        num_members = len(hh.members)
        income = hh.hh_total_annual_income()
        if num_members == 1:
            if income > 2799:
                return False
        elif num_members == 2:
            if income > 3799:
                return False
        elif num_members == 3:
            if income > 4799:
                return False
        elif num_members == 4:
            if income > 5598:
                return False
        elif num_members == 5:
            if income > 6798:
                return False
        elif num_members == 6:
            if income > 7798:
                return False
        elif num_members == 7:
            if income > 8798:
                return False
        elif num_members == 8:
            if income > 9798:
                return False
        else:
            if income > 9789 + (num_members - 8) * 1000:
                return False

        def user_is_eligible(user):
            if user["place_of_residence"] != PlaceOfResidenceEnum.NYC.value:
                return False
            if user["citizenship"] not in [
                CitizenshipEnum.CITIZEN_OR_NATIONAL,
                CitizenshipEnum.LAWFULLY_RESIDENT,
            ]:
                return False

        for m in hh.members:
            if user_is_eligible(m):
                return True
        return False


class IDNYC(BaseBenefitsProgram):
    """Anyone who lives in NYC and is age 10 and older is eligible to apply for an IDNYC card."""

    @staticmethod
    def __call__(self, hh):
        for m in hh.members:
            if m["place_of_residence"] != PlaceOfResidenceEnum.NYC.value:
                return False
            if m["age"] < 10:
                return False
        return True


class OfficeOfChildSupportServices(BaseBenefitsProgram):
    """To be eligible for child support services, you should be able to answer yes to these questions:

    Are you a singleparent or legal guardian of a child under the age of 21?
    Are you primarily responsible for the child's day-to-day life?"""

    @staticmethod
    def __call__(self, hh):
        has_spouse = False
        for m in hh.members:
            if m["relation"] == RelationEnum.SPOUSE.value:
                has_spouse = True
        if has_spouse:
            return True
        child_under_21 = False
        for m in hh.members:
            if m["age"] < 21:
                if m["relation"] in [
                    RelationEnum.CHILD.value,
                    RelationEnum.STEPCHILD.value,
                    RelationEnum.FOSTER_CHILD.value,
                    RelationEnum.ADOPTED_CHILD.value,
                ]:
                    child_under_21 = True
        if not child_under_21:
            return False
        if has_spouse:
            if hh.spouse()["responsible_for_day_to_day"]:
                return True
        if hh.user()["responsible_for_day_to_day"]:
            return True
        return False


class HIVAIDSServicesAdministration(BaseBenefitsProgram):
    """To be eligible for HASA, you must have been diagnosed with HIV or with AIDS, as defined by the Centers for Disease Control and Prevention (CDC). You do not need to have symptoms to be eligible."""

    @staticmethod
    def __call__(self, hh):
        for m in hh.members:
            if m["hiv_aids"]:
                return True
        return False


class AdultProtectiveServices(BaseBenefitsProgram):
    """To be eligible for APS, you should be able to answer yes to the following questions:

    Are you 18 or older?
    Do you have a physical or mental disability?
    Are you unable to manage your own resources, carry out daily activities, or protect yourself from dangerous situations without help from others?
    Do you have no one else who is willing and able to help you responsibly?
    """

    @staticmethod
    def __call__(hh):
        def is_eligible(m):
            if m["age"] < 18:
                return False
            if not m["disability"]:
                return False
            if m["can_manage_self"]:
                return False
            if m["has_family_to_help"]:
                return False
            return True

        for m in hh.members:
            if is_eligible(m):
                return True
        return False


class AccessARideParatransitService(BaseBenefitsProgram):
    """You are eligible for Access-A-Ride if:
    you have a disability that prevents you from using accessible buses or subways for some or all of your trips OR:
    if you're recovering from surgery, have a long-term condition, or are seeking Paratransit Service during your visit to NYC.
    """

    @staticmethod
    def __call__(self, hh):
        def is_eligible(m):
            if m["can_access_subway_or_bus"]:
                return True
            if m["recovering_from_surgery"]:
                return True
            return False

        for m in hh.members:
            if is_eligible(m):
                return True
        return False


class BeaconPrograms(BaseBenefitsProgram):
    """Youth ages 5-21 who are enrolled in school can participate."""

    @staticmethod
    def __call__(hh):
        def is_eligible(m):
            if m["age"] < 5 or m["age"] > 21:
                return False
            else:
                return True

        for m in hh.members:
            if is_eligible(m):
                return True
        return False


class NYCFreeTaxPrep(BaseBenefitsProgram):
    """For tax year 2023, youre eligible for NYC Free Tax Prep if you earned:

    $85,000 or less in 2023 and have dependents.
    $59,000 or less in 2023 and dont have dependents."""

    @staticmethod
    def __call__(hh):
        income = hh.total_annual_income()
        has_dependents = False
        for m in hh.members:
            if m["dependent"]:
                has_dependents = True
        if has_dependents:
            if income <= 85000:
                return True
        else:
            if income <= 59000:
                return True
        return False


class MiddleSchool(BaseBenefitsProgram):
    """To be eligible to apply for a New York City public middle school, your child should be able to answer yes to these questions:

    Are you a New York City resident?
    Are you currently a 5th-grade student?"""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["current_school_level"] == 5:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    return True
        return False


class KindergartenAndElementarySchool(BaseBenefitsProgram):
    """All NYC children age 4-5 are eligible for kindergarten and are guaranteed placement."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] >= 4 and m["age"] <= 5:
                return True
        return False


class CUNYFatherhoodAcademy(BaseBenefitsProgram):
    """To be eligible for CUNY Fatherhood Academy, you should be able to answer yes to these questions.

    For the high school equivalency AND college prep track:
    Do you live in New York City?
    Are you a father?

    For the high school equivalency track only:
    Are you between 18 and 30 years old?

    For the college prep track only:
    Are you between 18 and 30 years old?
    Do you have a high school diploma, HSE diploma, or GED?
    Do you have less than 12 college credits?
    Are you currently not enrolled in college?"""

    @staticmethod
    def __call__(hh):
        def both_tracks(m):
            if not m["sex"] == SexEnum.MALE.value:
                return False
            if not m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                return False
            return True

        def hs_track(m):
            if m["age"] < 18 or m["age"] > 30:
                return False
            return True

        def college_track(m):
            if m["age"] < 18 or m["age"] > 30:
                return False
            if m["education_level"] not in [
                EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value,
                EducationLevelEnum.HSE_DIPLOMA.value,
                EducationLevelEnum.GED.value,
            ]:
                return False
            if m["college_credits"] >= 12:
                return False
            if m["college"]:
                return False
            return True

        has_children = False
        for m in hh.members:
            if m["relation"] in [
                RelationEnum.CHILD.value,
                RelationEnum.ADOPTED_CHILD.value,
                RelationEnum.STEPCHILD.value,
                RelationEnum.FOSTER_CHILD.value,
            ]:
                has_children = True
        if not has_children:
            return False

        def eligible(m):
            if not both_tracks(m):
                return False
            if hs_track(m):
                return True
            if college_track(m):
                return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False


class NewbornHomeVisitingProgram(BaseBenefitsProgram):
    """You are eligible if you are a parent with a baby less than 3 months old and who:

    lives in a NYCHA development
    currently gets help from the Administration for Children's Services (ACS).
    lives in a Department of Homeless Services (DHS) shelter."""

    @staticmethod
    def __call__(hh):
        has_3_month_baby = False
        for m in hh.members:
            if m["relation"] in [
                RelationEnum.CHILD.value,
                RelationEnum.ADOPTED_CHILD.value,
                RelationEnum.STEPCHILD.value,
                RelationEnum.FOSTER_CHILD.value,
            ]:
                if m["age"] <= 0.25:
                    has_3_month_baby = True
        if not has_3_month_baby:
            return False

        if hh.user()["housing_type"] == HousingEnum.NYCHA_DEVELOPMENT.value:
            return True
        if hh.user()["housing_type"] == HousingEnum.DHS_SHELTER.value:
            return True
        if hh.user()["acs"] == True:
            return True

        # housing type same for spouse
        if hh.spouse() and hh.spouse()["acs"] == True:
            return True

        return False


class ChildrenandYouthwithSpecialHealthCareNeeds(BaseBenefitsProgram):
    """Eligible children must:

    Be age 21 or younger

    Live in New York City

    Have been diagnosed with or may have a serious or chronic health condition, physical disability, or developmental or emotional/behavioral condition

    Need extra health care and assistance"""

    @staticmethod
    def __call__(hh):
        def is_eligible(m):
            if not m["age"] > 21:
                return False
            if not m["place_of_residence"] != PlaceOfResidenceEnum.NYC.value:
                return False
            if not m["can_care_for_self"]:
                return False

            if m["chronic_health_condition"]:
                return True
            if m["disabled"]:
                return True
            if m["emotional_behavioral_condition"]:
                return True
            return False

        for m in hh.members:
            if is_eligible(m):
                return True
        return False


class OutpatientTreatmentServices(BaseBenefitsProgram):
    """Clinic treatment is available for:

    Children from birth to age 18 (not all clinics accept children under 5 years old)
    Children with emotional, behavioral, or mental health challenges
    Day treatment programs are available for:

    Children between the ages of 3 and 18 (there are two programs in NYC for children ages 3 - 5)
    Children with mental health disorders that interfere with school or social functions in a major way
    Children who have serious difficulties in a regular classroom"""

    @staticmethod
    def __call__(hh):
        has_children = False
        for m in hh.members:
            if m["age"] <= 18:
                has_children = True
        if has_children:
            return True
        return False


class ChildHealthPlusAndChildrensMedicaid(BaseBenefitsProgram):
    """Your child is eligible for CHP or Children's Medicaid if they are:

    Under 19 years old
    A resident of New York State
    Not covered by other health insurance (for Child Health Plus)"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["age"] >= 19:
                return False
            if m["place_of_residence"] != PlaceOfResidenceEnum.NYC.value:
                return False
            if m["health_insurance"]:
                return False
            return True

        for m in hh.members:
            if eligible(m):
                return True
        return False


class PrimaryAndPreventiveHealthCare(BaseBenefitsProgram):
    """All New York City youth are eligible for health care regardless of immigration status or whether they can pay"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["age"] >= 19:
                return False
            if m["place_of_residence"] != PlaceOfResidenceEnum.NYC.value:
                return False
            return True

        for m in hh.members:
            if eligible(m):
                return True
        return False


class FamilyResourceCenters(BaseBenefitsProgram):
    """Parents or caregivers of a child 0 to 24 years old who is at risk of or currently experiences emotional, behavioral or mental health challenges."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["relation"] in [
                RelationEnum.CHILD.value,
                RelationEnum.ADOPTED_CHILD.value,
                RelationEnum.STEPCHILD.value,
                RelationEnum.FOSTER_CHILD.value,
            ]:
                if m["age"] <= 24:
                    if m["emotional_behavioral_condition"]:
                        return True

        for m in hh.members:
            if eligible(m):
                return True
        return False


class FamilyAssessmentProgam(BaseBenefitsProgram):
    """Children up to 18 years old and their families who are struggling to relate to one another can get help.
    Any family can get help from FAP; you do NOT need to have an open ACS case.
    """

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["age"] < 18:
                if m["struggles_to_relate"]:
                    return True

        for m in hh.members:
            if eligible(m):
                return True
        return False


class CornerstonePrograms(BaseBenefitsProgram):
    """Cornerstone Programs are available to NYCHA residents who are Kindergarten-age or older."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["age"] >= 5:
                return True

        for m in hh.members:
            if eligible(m):
                return True
        return False


class TheEarlyInterventionProgram(BaseBenefitsProgram):
    """To be eligible for EIP your child must:

    Live in New York City.
    Be three years old or younger."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["age"] <= 3:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    return True

        for m in hh.members:
            if eligible(m):
                return True
        return False


class NYCHAPublicHousing(BaseBenefitsProgram):
    """To be eligible for public housing, you should be able to answer yes to these questions.

    Is at least one person in your household a US citizen? If not, does at least one person have legal immigration status (e.g. permanent resident, refugee, or asylum status)?
    Are you 18 or older, or are you an emancipated minor?
    If you are applying with someone (a spouse or domestic partner), are they 18 years or older, or are they an emancipated minor?
    Do you meet NYCHA's definition of family, which includes:
    two or more people related by blood, marriage, domestic partnership, adoption, guardianship, or court-awarded custody?
    a single person?
    Is your family's income at or below the income limits?
    Family size

    Income in a year
    1 $87,100
    2 $99,550
    3 $111,950
    4 $124,400
    5 $134,350
    6 $144,300
    7 $154,250
    8 $164,200"""

    @staticmethod
    def __call__(hh):
        def r1(hh):
            for m in hh.members:
                if m["citizenship"] in [
                    CitizenshipEnum.CITIZEN_OR_NATIONAL.value,
                    CitizenshipEnum.LAWFUL_RESIDENT.value,
                ]:
                    return True

        def r2(hh):
            if hh.user()["age"] >= 18:
                return True
            if hh.user()["emancipated_minor"]:
                return True
            return False

        def r3(hh):
            if hh.spouse():
                if hh.spouse()["age"] >= 18:
                    return True
                if hh.spouse()["emancipated_minor"]:
                    return True
                return False
            return True

        def r4(hh):
            if len(hh.members) == 1:
                return True
            for m in hh.members:
                if m["relation"] in [
                    RelationEnum.SPOUSE.value,
                    RelationEnum.CHILD.value,
                    RelationEnum.GRANDCHILD.value,
                    RelationEnum.ADOPTED_CHILD.value,
                    RelationEnum.SIBLING.value,
                    RelationEnum.NIECE_NEPHEW.value,
                    RelationEnum.OTHER_FAMILY.value,
                ]:
                    return True
            return False

        def r5(hh):
            thresholds = {
                1: 87100,
                2: 99550,
                3: 119950,
                4: 124400,
                5: 134350,
                6: 144300,
                7: 154250,
                8: 164200,
            }
            income = hh.total_annual_income()
            family_size = len(hh.members)
            if family_size > 8:
                return income <= thresholds[8] + 950 * (family_size - 8)
            return income <= thresholds[family_size]

        if not r1(hh):
            return False
        if not r2(hh):
            return False
        if not r3(hh):
            return False
        if not r4(hh):
            return False
        if not r5(hh):
            return False
        return True


class SchoolBasedServicesAgeAndEarlyChildhoodFamilyAndCommunityEngagementFACECenters(
    BaseBenefitsProgram
):
    """All NYC families who care for a child with disabilities are eligible to receive services for free. Trainings are open to everyone."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["age"] < 18:
                if m["disabled"]:
                    return True

        for m in hh.members:
            if eligible(m):
                return True
        return False


class ThreeK(BaseBenefitsProgram):
    """All New York City families with children under age 4 can enroll for the 2024-2025 school year. This includes children with disabilities or who are learning English.

    Children do not need to be toilet trained to attend 3-K."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["age"] < 4:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    return True

        for m in hh.members:
            if eligible(m):
                return True
        return False


class MedicaidForPregnantWomen(BaseBenefitsProgram):
    """To be eligible, you should be able to answer yes to these questions:

    Are you a New York State resident?
    Is your income equal to or below these guidelines?
    Are you pregnant?
    Family size (include unborn children)

    Yearly income
    1 $33,584
    2 $45,581
    3 $57,579
    4 $69,576
    5 $81,573
    6 $93,571
    7 $105,568
    8 $117,566
    For each additional person, add: $11,997"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["pregnant"]:
                    return True
            return False

        def income(hh):
            thresholds = {
                1: 33584,
                2: 45581,
                3: 57579,
                4: 69576,
                5: 81573,
                6: 93571,
                7: 105568,
                8: 117566,
            }
            family_size = len(hh.members)
            if family_size > 8:
                return income <= thresholds[8] + 11997 * (family_size - 8)
            return income <= thresholds[family_size]

        if income(hh):
            for m in hh.members:
                if eligible(m):
                    return True
        return False


class AcceleratedStudyInAssociatePrograms(BaseBenefitsProgram):
    """To be eligible for CUNY ASAP, you should have all of the following:

    Have been accepted to a CUNY college
    Be a New York City resident or eligible for in-state tuition
    Be proficient in Math and/or English (reading and writing)
    Have no more than 15 college credits and a minimum GPA of 2.0."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["in_state_tuition"]:
                    if m["proficient_in_math"]:
                        if m["proficient_in_english_reading_and_writing"]:
                            if m["college_credits"] <= 15:
                                if m["gpa"] >= 2.0:
                                    return True

        for m in hh.members:
            if eligible(m):
                return True
        return False


class CUNYStart(BaseBenefitsProgram):
    """You might be eligible for CUNY Start/Math Start if you can answer yes to these questions:

    Have you earned a high school or high school equivalency diploma?
    Have you completed all CUNY admissions requirements and accepted your offer to CUNY?
    Specific colleges may have other eligibility criteria. Contact the CUNY Start program at your campus for more information.
    """

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["high_school_equivalent"] in [
                EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value,
                EducationLevelEnum.HSE_DIPLOMA.value,
                EducationLevelEnum.GED.value,
            ]:
                if m["accepted_to_cuny"]:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False


class AdvanceAndEarn(BaseBenefitsProgram):
    """To be eligible for Advance & Earn, you should be able to answer yes to these questions:Are you a resident of NYC?
    Are you 16-24 years old?
    Are you not attending school? (i.e. you have not graduated high school, you do not have a high school equivalency diploma)
    Are you not working?
    Are you legally authorized to work in New York City?"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["age"] >= 16 and m["age"] <= 24:
                    if not m["college"]:
                        if m["annual_work_income"] <= 0:
                            if m["citizenship"] in [
                                CitizenshipEnum.CITIZEN_OR_NATIONAL.value,
                                CitizenshipEnum.LAWFUL_RESIDENT.value,
                            ]:
                                return True
            return False

        for m in hh.members:
            if eligible(m):
                return True


class TrainAndEarn(BaseBenefitsProgram):
    """To be eligible for Train & Earn, you should be able to answer yes to these questions:

    Are you a resident of NYC?
    Are you 16-24 years old?
    Are you not working?
    Are you registered for selective service, if you're an eligible male?
    Are you not attending school? (i.e. you have not graduated high school, or you do not have a high school equivalency diploma)
    OR you have a high school diploma or equivalency but are an English language learner
    Do you meet one of these requirements:
    You or someone in your household gets cash assistance or SNAP (food stamps)
    You are a homeless or runaway youth
    You are a foster care youth or have aged out of the foster care system
    You are involved in the justice system
    You have a disability
    You are pregnant or a parent
    You are low-income
    Is your households income at or below the amount shown in this chart?

    .Household size
    Monthly income
    Yearly income

    1
    $1,215
    $14,580

    2
    $1,643
    $19,720

    3
    $2,072
    $24,860

    4
    $2,500
    $30,000

    5
    $2,928
    $35,140

    6
    $3,357
    $40,280

    7
    $3,785
    $45,420

    8
    $4,213
    $50,560

    For each additional person
    add $428
    add $5,140
    You must be able to work lawfully in the United States to participate in paid work experiences.
    """

    @staticmethod
    def __call__(hh):
        def ss(m):
            if m["selective_service"]:
                return True
            if m["sex"] == SexEnum.MALE.value:
                return True
            return False

        def student(m):
            if m["current_school_level"] != GradeLevelEnum.NONE.value:
                return True
            if m["english_language_learner"]:
                return True
            return False

        def r6(m):
            if m["receives_snap"]:
                return True
            if m["housing_type"] == HousingEnum.HOMELESS.value:
                return True
            if m["is_runaway"]:
                return True
            if m["involved_in_justice_system"]:
                return True
            if m["disabled"]:
                return True
            if m["pregnant"]:
                return True
            return False

        def low_income(hh):
            thresholds = {
                1: 14580,
                2: 19720,
                3: 24860,
                4: 30000,
                5: 35140,
                6: 40280,
                7: 45420,
                8: 50560,
            }
            income = hh.total_annual_income()
            family_size = len(hh.members)
            if family_size > 8:
                return income <= thresholds[8] + 5140 * (family_size - 8)
            return income <= thresholds[family_size]

        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["age"] >= 16 and m["age"] <= 24:
                    if m["annual_work_income"] <= 0:
                        if ss(m):
                            if not student(m):
                                return True
            return False

        for m in hh.members:
            if eligible(m):
                if low_income(hh):
                    return True
                if r6(m):
                    return True
        return False


class NYCYouthHealth(BaseBenefitsProgram):
    """All NYC youth, regardless of ability to pay, immigration status, or sexual orientation, disability, sexual orientation, or gender identity are eligible for care at YouthHealth clinics."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["age"] < 18:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False


class NYCLaddersForLeaders(BaseBenefitsProgram):
    """You are eligible if you can answer yes to these questions.

    Do you live in New York City?
    Are you currently enrolled in high school or college?
    Are you between 16 - 22 years old?
    Do you have previous work experience (even as a volunteer)?
    Do you have a minimum grade-point average (GPA) of 3.00?"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["current_school_level"] in [9, 10, 11, 12, "college"]:
                    if m["age"] >= 16 and m["age"] <= 22:
                        if m["work_experience"]:
                            if m["gpa"] >= 3.0:
                                return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False


class NYCYouthLeadershipCouncils(BaseBenefitsProgram):
    """Youth are eligible to apply for YLCs if they are:

    14 - 21 years old
    Enrolled in high school or an equivalency program"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["age"] >= 14 and m["age"] <= 21:
                if m["current_school_level"] in [9, 10, 11, 12]:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True


class JobsPlus(BaseBenefitsProgram):
    """NYCHA residents who are old enough to work and live in a Jobs Plus neighborhood can enroll in Jobs-Plus:"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["lives_in_jobs_plus_neighborhood"]:
                if m["age"] >= 16:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True


class HighSchool(BaseBenefitsProgram):
    """To be eligible to apply for a New York City public high school, you should be able to answer yes to these questions:

    Are you an NYC resident?
    Are you currently an 8th-grade student or a first-time 9th-grade student?
    During the application period, all of the following students are welcome to apply:

    Current public district and charter school students,
    Private or parochial school students,
    Homeschooled students,
    Students with disabilities and students with accessibility needs,
    Students from immigrant families or who are learning English,
    Students in temporary housing,
    LGBTQ and gender nonconforming students, and
    Students with children"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["current_school_level"] in [8, 9]:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True


class CareerAndTechnicalEducation(BaseBenefitsProgram):
    """To be eligible to apply for a New York City public high school, you must be able to answer yes to these questions:

    Are you a New York City resident?
    Are you currently an 8th-grade student or a first-time 9th-grade student?"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["current_school_level"] in [8, 9]:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True


class VeteransAffairsSupportedHousing(BaseBenefitsProgram):
    """To be eligible for HUD-VASH, you must:

    be eligible for VA health care.
    be homeless."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["homeless"]:
                if m["va_heathcare"]:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False


class CoolingAssistanceBenefit(BaseBenefitsProgram):
    """You are eligible if all these apply to you:

    Your household has at least one of these:
        at least one person with a documented medical condition that is exacerbated by extreme heat. This must be verified in writing by a physician, physician assistant, or nurse practitioner
        Someone age 60 or older
        Children under age 6
    Someone in your household is a U.S. Citizen or legal resident.
    You do not have a working air conditioner or the one you have is at least five years old.
    You did not get a HEAP-funded air conditioner in the last five years.
    One of these:
        You get SNAP benefits, Temporary Assistance (TA), or Code A Supplemental Security Income (SSI Living Alone).
        You live in like NYCHA or Section 8 housing with heat included in your rent.
        Your household's gross monthly income is at or below the guidelines in this table:
        Household Size

        Maximum Gross Monthly Income for 2024
        1 $3,035
        2 $3,970
        3 $4,904
        4 $5,838
        5 $6,772
        6 $7,706
        7 $7,881
        8 $8,056
        9 $8,231
        10 $8,407
        11 $8,582
        12 $8,890
        13 $9,532
        For each additional person Add $642"""

    @staticmethod
    def __call__(hh):
        def r1(hh):
            for m in hh.members:
                if m["heat_exacerbated_condition"]:
                    return True
                if m["age"] >= 60:
                    return True
                if m["age"] < 6:
                    return True
            return False

        def r2(hh):
            for m in hh.members:
                if m["citizenship"] == CitizenshipEnum.CITIZEN_OR_NATIONAL.value:
                    return True
            return False

        def r3(hh):
            for m in hh.members:
                if m["ac"]:
                    return False
            return True

        def r4(hh):
            for m in hh.members:
                if m["got_heap_ac"] >= 5:
                    return True
                if m["got_heap_ac"] == 0:
                    return True
            return False

        def r5(hh):
            for m in hh.members:
                if m["receives_snap"]:
                    return True
                if m["receives_temporary_assistance"]:
                    return True
                if m["receives_ssi"]:
                    return True
                if m["heat_included_in_rent"]:
                    if m["place_of_residence"] in [
                        PlaceOfResidenceEnum.NYCHA_DEVELOPMENT.value,
                        PlaceOfResidenceEnum.SECTION_8.value,
                    ]:
                        return True
                def income(hh):
                    thresholds = {
                        1: 3035,
                        2: 3970,
                        3: 4904,
                        4: 5838,
                        5: 6772,
                        6: 7706,
                        7: 7811,
                        8: 8056,
                        9: 8231,
                        10: 8407,
                        11: 8452,
                        12: 8890,
                        13: 9532,
                    }
                    hh_income = hh.hh_total_annual_income()
                    hh_size = hh.num_members()
                    if hh_size > 13:
                        return hh_income <= 9532 + 642 * (hh_size - 13)
                    return hh_income <= thresholds[hh_size]
                if income(hh):
                    return True
            return False

        if r1(hh):
            if r2(hh):
                if r3(hh):
                    if r4(hh):
                        if r5(hh):
                            return True
        return False
    
class NYCCare(BaseBenefitsProgram):
    """To be eligible for NYC Care, you must:

    Live in New York City.
    Not qualify for any health insurance plan available in New York State."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NEW_YORK.value:
                if not m["qualify_for_health_insurance"]:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False

class ActionNYC(BaseBenefitsProgram):
    """ActionNYC is for all New Yorkers, regardless of immigration status. Your documented status does not affect your eligbility to use ActionNYC."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["place_of_residence"] == PlaceOfResidenceEnum.NEW_YORK.value:
                return True
        return False
    
class FairFaresNYC(BaseBenefitsProgram):
    """You may be eligible if you answer yes to these questions:

    Do you live in New York City?
    Are you 18-64 years old?
    Do you have a household income at or below these amounts?
    Household size

    Yearly income
    1 $18,072
    2 $24,528
    3 $30,984
    4 $37,440
    5 $43,896
    6 $50,352
    7 $56,808
    8 $63,264
    9 $69,720
    10 $76,176
    11 $82,632
    12 $89,088
    13 $95,544
    14 $102,000"""

    @staticmethod
    def __call__(hh):
        def threshold(hh):
            income = hh.hh_total_annual_income()
            size = hh.num_members()
            thresholds = {
                1: 18072,
                2: 24528,
                3: 30984,
                4: 37440,
                5: 43896,
                6: 50352,
                7: 56808,
                8: 63264,
                9: 69720,
                10: 76176,
                11: 82632,
                12: 89088,
                13: 95544,
                14: 102000,
            }
            if size > 14:
                return income <= 102000 + 6456 * (size - 14)
            return income <= thresholds[size]
        if not threshold(hh):
            return False
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NEW_YORK.value:
                if m["age"] >= 18 and m["age"] <= 64:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False

class WeSpeakNYC(BaseBenefitsProgram):
    """Anyone is eligible to sign up for an online class. Classes and materials are created for intermediate English language learners ages 16 and above."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] >= 16:
                if m["english_language_learner"]:
                    return True
        return False
    
class Homebase(BaseBenefitsProgram):
    """You are eligible for Homebase services if you:
    Live in one of the five boroughs of NYC
    May be soon at risk of entering the NYC shelter system"""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["at_risk_of_homelessness"]:
                    return True
        return False
    
class SafeAndSickLeave(BaseBenefitsProgram):
    """NYC Paid Safe and Sick Leave covers most employees at any size business or nonprofit in NYC:
        Full and part-time employees, transitional jobs program employees, and employees who live outside of NYC.

    You are not covered if you:

        Are a student in a federal work study program
        Are compensated by qualified scholarship programs
        Are employed by a government agency
        Are a physical therapist, occupational therapist, speech language pathologist, or audiologist licensed by the New York State Department of Education and meet a minimum pay threshold
        Are an independent contractor who does not meet the definition of an employee under New York State Labor Law (go to labor.ny.gov and search Independent Contractors)
        Participate in a Work Experience Program (WEP)
        An employee subject to a collective bargaining agreement that waives the law and has a comparable benefit.
    NY State Emergency COVID-19 and Paid Sick and Family Leave covers people under mandatory quarantine or isolation orders or whose minor dependent is."""

    @staticmethod
    def __call__(hh):
        def ineligible(m):
            if m["federal_work_study"]:
                return True
            if m["scholarship"]:
                return True
            if m["government_job"]:
                return True
            if m["is_therapist"]:
                return True
            if m["contractor"]:
                return True
            if m["wep"]:
                return True
            if m["collective_bargaining"]:
                return True
            return False
        for m in hh.members:
            if not ineligible(m):
                if m["work_income"] > 0:
                    return True
        return False

class STEMMattersNYC(BaseBenefitsProgram):
    """Available to students entering grades 1 through 12 in NYC public and charter schools in September 2024."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["current_school_level"] in [list(range(1, 13))]:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    return True
        return False
    
class COVIDNineteenFuneralAssistance(BaseBenefitsProgram):
    """You are be eligible if:

    You are a U.S. citizen, noncitizen national, or qualified noncitizen.
    You incurred COVID-19-related funeral expenses on or after January 20, 2020.
    The death was attributed to COVID-19, as noted on the death certificate."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["covid_funeral_expenses"]:
                if m["citizenship"] in [
                    CitizenshipEnum.CITIZEN_OR_NATIONAL.value,
                    CitizenshipEnum.LAWFUL_RESIDENT.value,
                ]:
                    return True    
        return False
    
class Lifeline(BaseBenefitsProgram):
    """You are eligible for Lifeline if you can answer yes to any of the questions below.

    Does someone in your household participate in one of these programs?
    Supplemental Nutrition Assistance Program (SNAP)
    Supplemental Security Income (SSI)
    Medicaid
    Federal Public Housing Assistance (FPHA)
    Veterans Pension and Survivors Benefit
    Is your income equal to or less than the income requirements?
    Household size Income in a year
    1 $20,331
    2 $27,594
    3 $34,857
    4 $42,120
    5 $49,383
    6 $56,646
    7 $63,909
    8 $71,172
    For each additional person, add: $7,263"""
    @staticmethod
    def __call__(hh):
        thresholds = {
            1: 20331,
            2: 27594,
            3: 34857,
            4: 42120,
            5: 49383,
            6: 56646,
            7: 63909,
            8: 71172,
        }
        income = hh.total_annual_income()
        family_size = len(hh.members)
        if family_size > 8:
            if income <= thresholds[8] + 7263 * (family_size - 8):
                return True
        if income <= thresholds[family_size]:
            return True
        for m in hh.members:
            if m["receives_snap"]:
                return True
            if m["receives_ssi"]:
                return True
            if m["receives_medicaid"]:
                return True
            if m["receives_fpha"]:
                return True
            if m["receives_vpsb"]:
                return True
        return False

class ChildCareVouchers(BaseBenefitsProgram):
    @staticmethod
    def __call__(hh):
        """You qualify for a voucher if your household gets Cash Assistance or is experiencing homelessness. You may also qualify if your household earns less than the incomes shown below:

        Family size
        Monthly income
        Yearly income

        2
        $6,156
        $73.869.56

        3
        $7,604
        $91,250.63

        4
        $9,053
        $108,631.70

        5
        $10,501
        $126,012.77

        6
        $11,949
        $143,393.84

        7
        $12,221
        $146,652.80

        8
        $12,493
        $149,911.75

        9
        $12,764
        $153,170.70

        10
        $13,036
        $156,429.65

        11
        $13,307
        $159,688.60

        12
        $13,579
        $162,947.55

        13
        $13,851
        $166,206.50

        14
        $14,122
        $169,465.45

        15
        $14,394
        $172,724.40

        16
        $14,665
        $175,983.35

        17
        $14,937
        $179,242.31

        18
        $15,208
        $182,501.26

        You also need to have one of these "reasons for care":

        You work 10+ hours per week
        You are in an educational or vocational training program
        You have been looking for work
        You live in temporary housing (priority access)
        You are attending services for domestic violence
        You are receiving treatment for substance abuse"""
    
    @staticmethod
    def __call__(hh):
        def income_req(hh):
            thresholds = {
                2: 6156,
                3: 7604,
                4: 9053,
                5: 10051,
                6: 11949,
                7: 12221,
                8: 12493,
                9: 12764,
                10: 13036,
                11: 13307,
                12: 13579,
                13: 13851,
                14: 14122,
                15: 14394,
                16: 14665,
                17: 14937,
                18: 15208,
            }
            income = hh.total_annual_income() / 12
            family_size = len(hh.members)
            if income > thresholds[family_size]:
                return False
            if family_size > 18:
                if income > thresholds[18] + 271 * (family_size - 18):
                    return False
            return True
        def has_kids(hh):
            for m in hh.members:
                if m["age"] < 14:
                    return True
                
        def other_reason(hh):
            for m in hh.members:
                if m["work_hours"] >= 10:
                    return True
                if m["enrolled_in_educational_training"]:
                    return True
                if m["enrolled_in_vocational_training"]:
                    return True
                if m["days_looking_for_work"]> 0:
                    return True
                if m["housing_type"] == HousingEnum.TEMPORARY_HOUSING.value:
                    return True
                if m["attending_service_for_domestic_violence"]:
                    return True
                if m["receiving_treatment_for_substance_abuse"]:
                    return True
            return False
                

        if len(hh.members) < 2:
            return False
        if has_kids(hh):
            if income_req(hh):
                return True
            if other_reason(hh):
                return True
        return False

class NYCFinancialEmpowermentCenters(BaseBenefitsProgram):
    """You are eligible for free financial counseling if you:

    Live or work in NYC; and
    Are at least 18 years old
    Income and immigration status do not matter."""
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] >= 18:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    return True
        return False

class FamilyHomelessnessAndEvictionPreventionSupplement(BaseBenefitsProgram):
    """Your family must have one of the following:
        A child under 18 years of age
        A child under 19 years of age who is enrolled full-time in high school or a vocational or technical training program
        A pregnant person
    One of these must also apply to your family:
        Have an active Cash Assistance case
        If you are in shelter, qualify for Cash Assistance when you leave shelter
    Your family must also meet one of the following:
        You are in HRA shelter.
        You are in DHS shelter and eligible for HRA shelter.
        You are in DHS shelter and were evicted* in NYC sometime in the year before you entered shelter.
        You are currently being evicted or were evicted* in NYC within the last 12 months.

        *Evicted can mean:
        An eviction proceeding against you or the person on the lease for your home.
        A foreclosure action for your building or home.
        A City agency said that you must leave your building or home because of health or safety reasons.
        A landlord has issued you a rent demand letter and threatened an eviction for non-payment of rent.
    """
    @staticmethod
    def __call__(hh):
        def r1(hh):
            for m in hh.members:
                if m["age"] < 18:
                    return True
                if m["age"] == 18 and m["enrolled_in_educational_training"]:
                    return True
                if m["age"] == 19 and m["enrolled_in_vocational_training"]:
                    return True
                if m["pregnant"]:
                    return True
            return False

        def r2(hh):
            for m in hh.members:
                if m["receives_cash_assistance"]:
                    return True
        def r3(hh):
            for m in hh.members:
                if m["housing_type"] == HousingEnum.HRA_SHELTER.value:
                    return True
                if m["housing_type"] == HousingEnum.DHS_SHELTER.value:
                    if m["eligible_for_HRA_shelter"]:
                        return True
                if m["evicted_months_ago"] > 0:
                    if m["evicted_months_ago"] < 12:
                        return True
                if m["currently_being_evicted"]:
                    return True
            return False

        if r1(hh):
            if r2(hh):
                if r3(hh):
                    return True
        return False
    
class NYSPaidFamilyLeave(BaseBenefitsProgram):
    """You can take Paid Family Leave if you:

    Are a resident of New York State.
    Work for a private employer in New York State or for a public employer who has opted in.
    Meet the time-worked requirements before taking Paid Family Leave:
    Full-time employees who regularly work 20 or more hours/week can take PFL after working 26 consecutive weeks.
    Part-time employees who regularly work less than 20 hours/week can take PFL after working 175 days. These days don't need to be consecutive."""
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["employer_opt_in"]:
                    if m["work_hours_per_week"] >= 20:
                        if m["consecutive_work_weeks"] >= 26:
                            return True
                        if m["nonconsecutive_work_weeks"] >= 175:
                            return True
        return False

class FamilyTypeHomesForAdults(BaseBenefitsProgram):
    """To be considered for FTHA housing, individuals must be:

    18 years or older, and
    Unable to live independently without semi-protective care. This may be due to a:
        Developmental disability
        Mental illness
        Physical disability
        Mental impairment
        Other requirements:

    If an individual has been diagnosed with a developmental disability or mental illness, they must be attending a day treatment program.
    If an individual has a history of substance abuse, they must be at least five years clean or sober.

    Individuals cannot be considered for FTHA housing if they are:
        Wheelchair bound
        Bed-ridden
        Have a history of:
        Substance abuse (and are not at least five years clean or sober)
        Non-compliance with medication and treatment
        Arson
        Verbal and/or abusive behavior
        Imprisonment"""
    @staticmethod
    def __call__(hh):
        def r1(m):
            if m["age"] >= 18:
                if m["can_care_for_self"]:
                    return True
            return False
        def r2(m):
            if m["developmental_condition"]:
                if not m["developmental_mental_day_treatment"]:
                    return False
            if m["mental_health_condition"]:
                if not m["developmental_mental_day_treatment"]:
                    return False
            return True
        
        def r3(m):
            if m["wheelchair"]:
                return False
            if m["bedridden"]:
                return False
            if 0 <= m["years_sober"] < 5:
                return False
            if m["medication_treatment_non_compliance"]:
                return False
            if m["arson"]:
                return False
            if m["verbal_abuse"]:
                return False
            if m["imprisonment"]:
                return False
            return True


        for m in hh.members:
            if r1(m):
                if r2(m):
                    if r3(m):
                        return True
        return False

class HomeFirstDownPaymentAssistance(BaseBenefitsProgram):
    """You will need to meet these income requirements to qualify for a HomeFirst loan:

    Family Size

    Maximum Household Income (80% AMI)
    1 $87,100
    2 $99,550
    3 $111,950
    4 $124,400
    5 $134,350
    6 $144,300
    7 $154,250
    8 $164,200

    You will also need to:

    Be a first-time homebuyer.
    This means you cannot have owned a home three years before buying a home with a HomeFirst loan.
    This requirement is waived for U. S. military veterans with a DD-214 that verifies honorable service."""

    @staticmethod
    def __call__(hh):
        def income(hh):
            thresholds = {
                1: 87000,
                2: 99550,
                3: 111950,
                4: 124400,
                5: 134350,
                6: 144300,
                7: 154250,
                8: 164200
            }
            income = hh.total_annual_income()
            family_size = len(hh.members)
            if family_size > 8:
                return income <= thresholds[8] + 9950 * (family_size - 8)
            return income <= thresholds[family_size]

        def r1(m):
            if m["first_time_homebuyer"]:
                return True
            if m["honorable_service"]:
                return True
            return False
        if income(hh):
            for m in hh.members:
                if r1(m):
                    return True
        return False
    
class NYCMitchellLama(BaseBenefitsProgram):
    """
    Eligibility for Mitchell-Lama rental units is determined by:

    The number of people who will be living in the unit
    Your total household income which cannot be more than the limits shown below. Different developments have different limits:
    Household Size

    Federally Assisted Rental

    Federally Assisted Cooperative

    Non-Federally Assisted
    1 | $86,960 | $135,875 | $135,875 | 
    2 | $99,440 | $155,375 | $155,375 | 
    3 | $111,840 | $174,750 | $174,750 | 
    4 | $124,240 | $194,125 | $194,125 | 
    5 | $134,160 | $209,625 | $209,625 | 
    6 | $144,080 | $225,125 | $225,125 | 
    7 | $154,080 | $240,750 | $240,750 | 
    8 | $164,000 | $256,250 | $256,250"""

    @staticmethod
    def __call__(hh):
        family_size = len(hh.members)
        income = hh.total_annual_income()
        thresholds = {
            1: 86960,
            2: 99440,
            3: 111840,
            4: 124240,
            5: 134160,
            6: 144080,
            7: 154080,
            8: 164000
        }
        if family_size > 8:
            return income <= thresholds[8] + 9920 * (family_size - 8)
        return income <= thresholds[family_size]

class NYCTenantResourcePortal(BaseBenefitsProgram):
    """
    All renters are eligible to receive help regardless of immigration status.
    """
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["monthly_rent_spending"] > 0:
                return True
        return False

class Text2Work(BaseBenefitsProgram):
    """TXT-2-WORK is available for:
    NYC residents
    HRA clients who receive assistance including temporary cash, SNAP, or housing assistance"""
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                return True
            if m["receives_hra"]:
                return True
            if m["receives_snap"]:
                return True
            if m["receives_fpha"]:
                return True
        return False
    
class SilverCorps(BaseBenefitsProgram):
    """You're eligible for Silver Corps if you:

    are at least 55 years of age
    are an NYC resident
    are currently unemployed
    have income at or below these levels
    Household size
    Your income
    1 $60,240
    2 $81,760
    3 $103,280
    4 $124,800
    5 $146,320
    6 $167,840
    7 $189,360
    8 $210,880
    For each additional person:
    Add $6,425
    """
    @staticmethod
    def __call__(hh):
        def income(hh):
            family_size = len(hh.members)
            income = hh.total_annual_income()
            thresholds = {
                1: 60240,
                2: 81760,
                3: 103280,
                4: 124800,
                5: 146320,
                6: 167840,
                7: 189360,
                8: 210880
            }
            if family_size > 8:
                return income <= thresholds[8] + 6425 * (family_size - 8)
            return income <= thresholds[family_size]
        if income(hh):
            for m in hh.members:
                if m["age"] >= 55:
                    if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                        if m["work_hours_per_week"] <= 0:
                            return True
        return False
    
class BigAppleConnect(BaseBenefitsProgram):
    """You can enroll in Big Apple Connect if you live in a NYCHA development."""
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["housing_type"] == HousingEnum.NYCHA_DEVELOPMENT.value:
                return True
        return False
    

class SeniorCitizenRentIncreaseExemption(BaseBenefitsProgram):
    """To be eligible for SCHE, you must meet these requirements:

    Own a one-, two-, or three-family home, condo, or coop apartment.
    All owners of the property are 65 or older. However, if you own the property with a spouse or sibling, only one of you need to be 65 or older.
    All owners must live on the property as the primary residence.
    The combined income for all owners must be less than or equal to $58,399.
    You must own the property for at least 12 consecutive months before the date of filing for the exemption. This is not a requirement if you got the exemption on a property that you owned before."""
    @staticmethod
    def __call__(hh):
        if hh.user()["housing_type"] not in [
            HousingEnum.HOUSE_2B, 
            HousingEnum.CONDO,
            HousingEnum.COOPERATIVE_APARTMENT
        ]:
            return False
        owner_indices = []
        spouse_or_sibling_owner = False
        for m in hh.members:
            if m["is_property_owner"]:
                owner_indices.append(hh.members.index(m))
                if m["relation"] == RelationEnum.SPOUSE.value or m["relation"] == RelationEnum.SIBLING.value:
                    spouse_or_sibling_owner = True
        num_over_65 = 0
        primary_residents = 0
        for i in owner_indices:
            if hh.members[i]["age"] >= 65:
                num_over_65 += 1
            if hh.members[i]["primary_residence"]:
                primary_residents += 1
        if spouse_or_sibling_owner:
            if num_over_65 == 0:
                # R1
                return False
        else:
            if num_over_65 != len(owner_indices):
                return False
        if primary_residents != len(owner_indices):
            return False
        if hh.total_annual_income() > 58399:
            return False
        if hh.user()["months_owned_property"] < 12:
            return False
        return True


class PreKForAll(BaseBenefitsProgram):
    """All NYC children age 3 or 4 are eligible. This includes children with disabilities or who are learning English.
    Children do not need to be toilet trained to attend pre-K."""
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] == 3 or m["age"] == 4:
                return True
        return False

class DisabledHomeownersExemption(BaseBenefitsProgram):
    """To be eligible for DHE, you must meet these requirements:

    Own a one-, two-, or three-family home, condo, or coop apartment.
    All property owners are people with disabilities. However, if you own the property with a spouse or sibling, only one of you need to have a disability to qualify.
    You must live on the property as your primary residence.
    The combined income for all owners must be less than or equal to $58,399.
    Your property cannot be within a housing development controlled by a Limited Profit Housing Company, Mitchell-Lama, Limited Dividend Housing Company, or redevelopment company. Contact your property manager if you're not sure."""
    @staticmethod
    def __call__(hh):
        if hh.user()["housing_type"] not in [
            HousingEnum.HOUSE_2B, 
            HousingEnum.CONDO,
            HousingEnum.COOPERATIVE_APARTMENT
        ]:
            return False
        owner_indices = []
        spouse_or_sibling_owner = False
        for m in hh.members:
            if m["is_property_owner"]:
                owner_indices.append(hh.members.index(m))
                if m["relation"] == RelationEnum.SPOUSE.value or m["relation"] == RelationEnum.SIBLING.value:
                    spouse_or_sibling_owner = True
        num_disabled = 0
        primary_residents = 0
        for i in owner_indices:
            if hh.members[i]["disabled"]:
                num_disabled += 1
            if hh.members[i]["primary_residence"]:
                primary_residents += 1
        if spouse_or_sibling_owner:
            if num_disabled == 0:
                # R1
                return False
        else:
            if num_disabled != len(owner_indices):
                return False
        if primary_residents != len(owner_indices):
            return False
        if hh.total_annual_income() > 58399:
            return False
        return True

class VeteransPropertyTaxExemption(BaseBenefitsProgram):
    """To be eligible for Veterans' Exemption, you should be able to answer yes to all of these questions.

    Are you the current deeded owner, spouse, or surviving un-remarried widow/widower of the property owner?
    Is the property your primary residence?
    Are you a veteran who served in the US armed forces during one of the following conflicts?
    Persian Gulf Conflict (includes the Afghanistan and Iraq Conflicts) - beginning August 2, 1990 to present
    Vietnam War - November 1, 1955 to May 7, 1975
    Korean War - June 27, 1950 - January 31, 1955
    World War II - December 7, 1941 to December 31, 1946
    World War I - April 6, 1917 to November 11, 1918"""
    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["is_property_owner"]:
                if m["primary_residence"]:
                    if m["conflict_veteran"]:
                        return True
            return False
        for m in hh.members:
            if eligible(m):
                return True
        return False

class HomeEnergyAssistanceProgram(BaseBenefitsProgram):
    """Regular HEAP eligibility and benefits are based on:

    the presence of a household member who is under age 6, age 60 or older or permanently disabled
    the primary heating source
    income and household size
    Household Size

    Maximum Monthly Gross Income for 2023-2024
    1 $3,035
    2 $3,970
    3 $4,904
    4 $5,838
    5 $6,772
    6 $7,706
    7 $7,881
    8 $8,056
    9 $8,231
    10 $8,407
    11 $8,582
    12 $8,890
    13 $9,532
    For each additional person, add:
    Add $642

    Your household is eligible for Emergency HEAP benefits if you can answer yes to these questions:

    Do any of these apply to you?
    The electricity that runs your heating system or thermostat is shut off or scheduled to be shut off.
    Your heat has been turned off or is in danger of being turned off.
    You are out of fuel or you have less than a quarter tank of fuel.
    Is your heating or electric bill in your name?
    Are your household's available resources:
    less than $3,750 if it includes someone 60 or older, or under age 6?
    less than $2,500 if it doesn't include someone 60 or older, or under age 6?
    Do you meet one of these income guidelines?
    You receive SNAP benefits, Temporary Assistance, or Code A Supplemental Security Income.
    Your family is at or under the following gross monthly income guidelines for your household size in the table above."""
    @staticmethod
    def __call__(hh):
        def r1(hh):
            user = hh.user()
            if user["electricity_shut_off"]:
                return True
            if user["heat_shut_off"]:
                return True
            if user["out_of_fuel"]:
                return True
            return False
        def r2(hh):
            for m in hh.members:
                if m["heating_electrical_bill_in_name"]:
                    return True
            return False
        def r3(hh):
            age_req = False
            for m in hh.members:
                if m["age"] < 6 or m["age"] >= 60:
                    age_req = True
            if age_req:
                if hh.user()["available_financial_resources"] < 3750:
                    return True
            else:
                if hh.user()["available_financial_resources"] < 2500:
                    return True
            return False
        def r4(hh):
            for m in hh.members:
                if m["receives_snap"]:
                    return True
                if m["receives_temp_assistance"]:
                    return True
                if m["receives_ssi"]:
                    return True
            thresholds = {
                1: 3035,
                2: 3970,
                3: 4904,
                4: 5838,
                5: 6772,
                6: 7706,
                7: 7708,
                8: 8056,
                9: 8231,
                10: 8407,
                11: 8582,
                12: 8890,
                13: 9532,
            }
            income = hh.hh_total_annual_income()
            family_size = len(hh.members)
            if family_size > 13:
                return income <= thresholds[13] + 642 * (family_size - 13)
            return income <= thresholds[family_size]
        if r1(hh):
            if r2(hh):
                if r3(hh):
                    if r4(hh):
                        return True
        return False
            

class NYSUnemploymentInsurance(BaseBenefitsProgram):
    """You are eligible for Unemployment Insurance (UI) if you:

    lost your job through no fault of your own (for example, you got laid off).
    worked within the last 18 months, and able to work immediately.
    are authorized to work in the US andwere authorized to work when you lost your job."""
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["lost_job"]:
                if m["months_since_worked"] <= 18:
                    if m["authorized_to_work"]:
                        return True
        return False


class SummerMeals(BaseBenefitsProgram):
    """Summer Meals is available to anyone age 18 or younger."""
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] < 18:
                return True
        return False

class NYCHAResidentEconomicEmpowermentAndSustainability(BaseBenefitsProgram):
    """Anyone who lives in NYCHA housing is eligible for REES."""
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["housing_type"] == HousingEnum.NYCHA_DEVELOPMENT.value:
                return True
        return False

class OlderAdultEmploymentProgram(BaseBenefitsProgram):
    """To be eligible for the Older Adult Employment Program, you must:

    Be 55 years or older
    Live in the five boroughs
    Be unemployed

    Your households income must be at or below the amount shown in this chart.
    Household size
    Your income (125% or less than the federal poverty level)
    1 $18,825
    2 $25,550
    3 $32,275
    4 $39,000
    5 $45,725
    6 $52,450
    7 $59,175
    8 $65,900
    For each additional person add $6,725"""
    @staticmethod
    def __call__(hh):
        def income(hh):
            income = hh.hh_total_annual_income()
            family_size = len(hh.members)
            thresholds = {
                1: 18825,
                2: 25550,
                3: 32275,
                4: 39000,
                5: 45725,
                6: 52450,
                7: 59175,
                8: 65900,
            }
            if family_size > 8:
                return income <= thresholds[8] + 6725 * (family_size - 8)
            return income <= thresholds[family_size]
        if income(hh):
            for m in hh.members:
                if m["age"] < 55:
                    if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                        if m["work_hours_per_week"] <= 0:
                            return True
        return False

class Workforce1CareerCenters(BaseBenefitsProgram):
    """You can get services from a Workforce1 Career Center if you:

    live in New York City
    are 18 or older
    are legally authorized to work in the U.S."""
    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["age"] >= 18:
                    if m["authorized_to_work_in_us"]:
                        return True
        return False

class CommoditySupplementalFoodProgram(BaseBenefitsProgram):
    """To be eligible, you should be able to answer yes to these questions:

    Are you 60 years old or older?
    Are you a New York State resident?
    You do not need to be a US citizen to receive CSFP benefits.
    Is your household's total pre-tax income equal to or less than CSFP income requirements below?
    Household size
    Income in a year
    1 $19,578
    2 $26,572
    3 $33,566
    4 $40,560
    5 $47,554
    6 $54,548
    7 $61,542
    8 $68,536
    For each additional person, add: $6,994"""
    @staticmethod
    def __call__(hh):
        def income(hh):
            income = hh.hh_total_annual_income()
            family_size = len(hh.members)
            thresholds = {
                1: 19578,
                2: 26572,
                3: 33566,
                4: 40560,
                5: 47554,
                6: 54548,
                7: 61542,
                8: 68536,
            }
            if family_size > 8:
                return income <= thresholds[8] + 6994 * (family_size - 8)
            return income <= thresholds[family_size]
        if not income:
            return False
        for m in hh.members:
            if m["age"] >= 60:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    return True
        return False

class LearnAndEarn(BaseBenefitsProgram):
    """You are eligible if you can answer yes to these questions:

Are you between 16 - 21 years old?
Are you an NYC high school junior or senior?
Do you have a social security number?
Can you legally work in the US?
Are you registered for selective service, if you're an eligible male?
Do you meet one of these requirements?
You or someone in your household gets cash assistance or SNAP (food stamps)
You are a homeless or runaway youth
You are a foster care youth or have aged out of the foster care system
You are involved in the justice system
You are pregnant or a parent
You have a disability.
Is your households income at or below the amount shown in this chart?
Household size
Monthly income | Yearly income | 
1 | $1,215 | $14,580 | 
2 | $1,643 | $19,720 | 
3 | $2,072 | $24,860 | 
4 | $2,500 | $30,000 | 
5 | $2,928 | $35,140 | 
6 | $3,357 | $40,280 | 
7 | $3,785 | $45,420 | 
8 | $4,213 | $50,560
For each additional person
add $428
add $5,140"""
    @staticmethod
    def __call__(hh):
        def income(hh):
            income = hh.hh_total_annual_income()
            family_size = len(hh.members)
            thresholds = {
                1: 14580,
                2: 19720,
                3: 24860,
                4: 30000,
                5: 35140,
                6: 40280,
                7: 45420,
                8: 50560,
            }
            if family_size > 8:
                return income <= thresholds[8] + 5140 * (family_size - 8)
            return income <= thresholds[family_size]
        if not income:
            return False
        def other(m):
            if m["receives_cash_assistance"]:
                return True
            if m["receives_snap"]:
                return True
            if m["housing_type"] == HousingEnum.HOMELESS.value:
                return True
            if m["is_runaway"]:
                return True
            if m["in_foster_care"]:
                return True
            if m["foster_age_out"]:
                return True
            if m["months_pregnant"] > 0:
                return True
            if m["is_parent"]:
                return True
            if m["disabled"]:
                return True
            
        for m in hh.members:
            if m["age"] >= 16 and m["age"] <= 21:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    if m["current_school_level"] in [9, 10, 11, 12]:
                        if m["ssn"]:
                            if m["authorized_to_work_in_us"]:
                                if m["registered_for_selective_service"] or m["is_eligible_for_selective_service"]:
                                    if other(m):
                                        return True
        return False


class NYCNurseFamilyPartnership(BaseBenefitsProgram):
    """You're eligible for NYC NFP if you can answer 'yes' to the below questions:

    Are you 28 weeks pregnant or less with your first baby?
    Do you live in New York City?
    Are you eligible for Medicaid?
    This program is available to all eligible parents, regardless of age, immigration status, or gender identity."""
    @staticmethod
    def __call__(hh):
        pass

class SummerYouthEmploymentProgram(BaseBenefitsProgram):
    """Description for Summer Youth Employment Program"""
    @staticmethod
    def __call__(hh):
        pass
