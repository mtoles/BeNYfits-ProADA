import random
from names import get_full_name
from users.users import Household, Person, nuclear_family
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
import unittest
from copy import deepcopy
import sys


def trace_returns(frame, event, arg):
    if event == "return":  # and frame.f_code.co_name == "foo":
        print(f"Returning from {frame.f_code.co_name} at line {frame.f_lineno}")
    return trace_returns


# from users.users import Household, Person # don't import this to avoid circular logic
class BenefitsProgramMeta(type):
    registry = {}

    def __new__(cls, name, bases, attrs):
        attrs["name"] = name
        new_program = super().__new__(cls, name, bases, attrs)
        if name != "BaseBenefitsProgram":
            cls.registry[name] = new_program
        return new_program

    @classmethod
    def run_tests(cls):
        passed = 0
        total = len(cls.registry)
        no_tests = []
        for program in cls.registry.values():
            if hasattr(program, "test_cases"):
                # print(f"Running tests for {program.__name__}")
                program.test_cases()
                # print(f"Finished running tests for {program.__name__}")
                passed += 1
            else:
                no_tests.append(program.__name__)
        print(f"Passed {passed} out of {total} tests")
        print(f"No tests for {', '.join(no_tests)}")


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
    # for i in range(len(hh.members)):
    #     for attr in user_features.BasePersonAttr.registry.keys():
    #         cls = user_features.BasePersonAttr.registry[attr]
    #         hh.members[i][attr] = cls.conform(cls, hh, i, hh.members[i][attr])
    hh = hh.conform()
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

            elif hh.user()["days_looking_for_work"] > 0:
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
                if m["lived_together_last_6_months"]:
                    qualifying_family_lived_with_hh.append(m)

            if filing_jointly:
                if not self_works:
                    return False
                if not spouse_works:
                    return False
                if not qualifying_family_lived_with_hh:
                    return False

                return True
            else:
                if not self_works:
                    return False
                if not qualifying_family_lived_with_hh:
                    return False
                return True

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

    @classmethod
    def test_cases(cls):
        ### Passing
        adult = Person.default_person(is_self=True)
        adult["works_outside_home"] = True
        child = Person.default_person(is_self=False)
        child["age"] = 12
        child["lived_together_last_6_months"]
        child["has_paid_caregiver"] = True
        adult["annual_work_income"] = 10000
        hh_pass_1 = Household([adult, child])

        hh_pass_2 = deepcopy(hh_pass_1)
        spouse = Person.default_person(is_self=False)
        spouse["relation"] = "spouse"
        spouse["works_outside_home"] = True
        spouse["filing_jointly"] = True
        hh_pass_2.user()["filing_jointly"] = True

        ### Failing
        hh_fail_1 = deepcopy(hh_pass_1)
        hh_fail_1.user()["works_outside_home"] = False

        hh_fail_2 = deepcopy(hh_pass_2)
        hh_fail_2.members[1]["age"] = 15

        for i, hh in enumerate([hh_pass_1, hh_pass_2]):
            result = cls.__call__(hh)
            assert result, f"test {i} failed"
        for i, hh in enumerate(
            [
                hh_fail_1,
                hh_fail_2,
            ]
        ):
            result = cls.__call__(hh)
            assert not result, f"test {i} failed"
        # return {"pass": [hh_pass_1, hh_pass_2], "fail": [hh_fail_1, hh_fail_2]}


# def ComprehensiveAfterSchool(hh) -> bool:
class ComprehensiveAfterSchoolSystemOfNYC(BaseBenefitsProgram):
    @staticmethod
    def __call__(hh) -> bool:
        """
        All NYC students in kindergarten to 12th grade are eligible to enroll in COMPASS programs. Each program may have different age and eligibility requirements.
        """
        for m in hh.members:
            if m["current_school_level"] in list(range(1, 13)) + ["k"]:
                return True
        return False

    @classmethod
    def test_cases(cls):
        ### Passing
        adult = Person.default_person(is_self=True)
        adult["current_school_level"] = 1
        hh_pass_1 = Household([adult])

        hh_fail_1 = deepcopy(hh_pass_1)
        hh_fail_1.members[0]["current_school_level"] = "college"

        for i, hh in enumerate(
            [
                hh_pass_1,
            ]
        ):
            result = cls.__call__(hh)
            assert result, f"test {i} failed"
        for i, hh in enumerate(
            [
                hh_fail_1,
            ]
        ):
            result = cls.__call__(hh)
            assert not result, f"test {i} failed"


# def EarlyHeadStartPrograms(hh) -> bool:
class EarlyHeadStart(BaseBenefitsProgram):
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
        hh_income = hh.hh_annual_total_income()
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
        elif foster_care:
            secondary_conditions = True

        if not _has_toddler(hh):
            return False
        elif not secondary_conditions:
            return False

        return True

    @classmethod
    def test_cases(cls):
        # Early Head Start Tests
        # Passing Test Case
        person_pass_ehs = Person.default_person(is_self=True)
        child1 = Person.default_person(is_self=False)
        child1["age"] = 2
        person_pass_ehs["housing_type"] = HousingEnum.TEMPORARY_HOUSING.value
        person_pass_ehs["annual_work_income"] = 14000
        hh_pass_ehs = Household([person_pass_ehs, child1])

        # Failing Test Case (child too old)
        person_fail_age_ehs = deepcopy(person_pass_ehs)
        child2 = deepcopy(child1)
        child2["age"] = 4
        hh_fail_age_ehs = Household([person_fail_age_ehs, child2])

        # Failing Test Case (income too high)
        person_fail_income_ehs = deepcopy(person_pass_ehs)
        person_fail_income_ehs["annual_work_income"] = 40000
        person_fail_income_ehs["housing_type"] = HousingEnum.HOUSE_2B.value
        hh_fail_income_ehs = Household([person_fail_income_ehs, child1])

        for i, hh in enumerate(
            [
                hh_pass_ehs,
            ]
        ):
            result = cls.__call__(hh)
            assert result, f"EarlyHeadStart test {i} failed"
        for i, hh in enumerate(
            [
                hh_fail_age_ehs,
                hh_fail_income_ehs,
            ]
        ):
            result = cls.__call__(hh)
            assert not result, f"EarlyHeadStart test {i} failed"


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
        hh_income_qualifies = _income_eligible(
            hh.hh_annual_total_income(), hh.num_members()
        )

        if not has_toddler:
            return False
        if hh_income_qualifies:
            return True
        if not user_qualifies:
            return False
        if not spouse_qualifies:
            return False

        return True

    @classmethod
    def test_cases(cls):
        hh1 = nuclear_family()
        hh1.user()["work_hours_per_week"] = 10
        hh1.spouse()["enrolled_in_educational_training"] = True
        hh1.members[2]["age"] = 0
        assert cls.__call__(hh1)


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
            # eligible_child = False

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

                if not m["lived_together_last_6_months"]:
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

    @classmethod
    def test_cases(cls):
        """Tests eligibility for the 2023 tax year credit."""

        # Test 1: Basic eligibility (single filer)
        hh1 = nuclear_family()
        hh1.user()["annual_work_income"] = 50000
        hh1.spouse()["annual_work_income"] = 0
        hh1.user()["work_hours_per_week"] = 10
        hh1.spouse()["work_hours_per_week"] = 10
        hh1.members[2]["age"] = 10
        hh1.members[2]["has_ssn"] = True
        hh1.members[2]["relation"] = RelationEnum.CHILD
        hh1.members[2]["lived_together_last_6_months"] = True
        hh1.members[2]["provides_over_half_of_own_financial_support"] = False
        hh1.user()["filing_jointly"] = False
        assert cls.__call__(hh1), "Test 1 failed: Basic eligibility"

        # Test 2: Married filing jointly with combined income below $400,000
        hh2 = nuclear_family()
        hh2.user()["annual_work_income"] = 150000
        hh2.spouse()["annual_work_income"] = 200000
        hh2.members[2]["age"] = 8
        hh2.members[2]["has_ssn"] = True
        hh2.members[2]["relation"] = RelationEnum.ADOPTED_CHILD
        hh2.members[2]["lived_together_last_6_months"] = True
        hh2.members[2]["provides_over_half_of_own_financial_support"] = False
        hh2.user()["filing_jointly"] = True
        assert cls.__call__(hh2), "Test 2 failed: Married filing jointly"

        # Test 3: Child aged 16, but no SSN or ATIN
        hh3 = nuclear_family()
        hh3.user()["annual_work_income"] = 75000
        hh3.members[2]["age"] = 16
        hh3.members[2]["has_ssn"] = False
        hh3.members[2]["relation"] = RelationEnum.CHILD
        hh3.members[2]["lived_together_last_6_months"] = True
        hh3.members[2]["provides_over_half_of_own_financial_support"] = False
        assert not cls.__call__(hh3), "Test 3 failed: Child missing SSN or ATIN"

        # Test 4: Income exceeds $200,000 for single filer
        hh4 = nuclear_family()
        hh4.user()["annual_work_income"] = 250000
        hh4.spouse()["annual_work_income"] = 0
        hh4.members[2]["age"] = 12
        hh4.members[2]["has_ssn"] = True
        hh4.members[2]["relation"] = RelationEnum.CHILD
        hh4.members[2]["lived_together_last_6_months"] = True
        hh4.members[2]["provides_over_half_of_own_financial_support"] = False
        hh4.user()["filing_jointly"] = False
        assert not cls.__call__(hh4), "Test 4 failed: Income exceeds $200,000"

        # Test 5: Qualifying child is not a direct relation (e.g., foster child)
        hh5 = nuclear_family()
        hh5.user()["annual_work_income"] = 120000
        hh5.members[2]["age"] = 15
        hh5.members[2]["has_ssn"] = True
        hh5.members[2]["relation"] = RelationEnum.FOSTER_CHILD
        hh5.members[2]["lived_together_last_6_months"] = True
        hh5.members[2]["provides_over_half_of_own_financial_support"] = False
        hh5.user()["filing_jointly"] = False
        assert cls.__call__(hh5), "Test 5 failed: Foster child as qualifying child"

        # Test 6: Child provided more than half of their financial support
        hh6 = nuclear_family()
        hh6.user()["annual_work_income"] = 50000
        hh6.members[2]["age"] = 14
        hh6.members[2]["has_ssn"] = True
        hh6.members[2]["relation"] = RelationEnum.CHILD
        hh6.members[2]["lived_together_last_6_months"] = True
        hh6.members[2]["provides_over_half_of_own_financial_support"] = True
        assert not cls.__call__(
            hh6
        ), "Test 6 failed: Child providing own financial support"

        # Test 7: Child did not live with the filer for more than half of the year
        hh7 = nuclear_family()
        hh7.user()["annual_work_income"] = 50000
        hh7.members[2]["age"] = 10
        hh7.members[2]["has_ssn"] = True
        hh7.members[2]["relation"] = RelationEnum.CHILD
        hh7.members[2]["lived_together_last_6_months"] = False
        hh7.members[2]["provides_over_half_of_own_financial_support"] = False
        assert not cls.__call__(hh7), "Test 7 failed: Child did not live with filer"

        # Additional edge cases
        # Ensure households with no children fail
        hh8 = nuclear_family()
        hh8.user()["annual_work_income"] = 60000
        hh8.spouse()["annual_work_income"] = 0
        hh8.members = hh8.members[:-1]
        assert not cls.__call__(hh8), "Test 8 failed: No children in household"

        # Ensure a child above 16 fails
        hh9 = nuclear_family()
        hh9.user()["annual_work_income"] = 50000
        hh9.members[2]["age"] = 17
        hh9.members[2]["has_ssn"] = True
        hh9.members[2]["relation"] = RelationEnum.CHILD
        hh9.members[2]["lived_together_last_6_months"] = True
        hh9.members[2]["provides_over_half_of_own_financial_support"] = False
        assert not cls.__call__(hh9), "Test 9 failed: Child above 16"


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
            if hh.hh_annual_total_income() <= 50000:
                return True
            return False

        def _r4(hh) -> bool:
            income = hh.user().total_income() / 12
            rent = hh.user()["monthly_rent_spending"]
            if rent > income / 3:
                return True
            return False

        def _r5(hh) -> bool:
            if not hh.user()["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
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

    @classmethod
    def test_cases(cls):
        # Test case 1: All requirements met
        hh1 = nuclear_family()
        hh1.user()["age"] = 25  # Age 18 or older
        hh1.user()["name_is_on_lease"] = True  # Name on lease
        hh1.user()["annual_work_income"] = 20000
        hh1.spouse()["annual_work_income"] = 15000
        hh1.members[2]["annual_work_income"] = 0  # Combined income $50,000 or less
        hh1.user()["monthly_rent_spending"] = 10000  # More than 1/3 of income on rent
        hh1.user()[
            "place_of_residence"
        ] = PlaceOfResidenceEnum.NYC.value  # Lives in NYC
        hh1.user()[
            "housing_type"
        ] = HousingEnum.RENT_STABILIZED_APARTMENT.value  # Valid housing type
        hh1.user()["receives_ssi"] = True  # Receives SSI
        assert cls.__call__(hh1)

        # Test case 2: Age requirement not met
        hh2 = nuclear_family()
        hh2.user()["age"] = 17  # Under 18
        hh2.user()["name_is_on_lease"] = True
        hh2.user()["annual_work_income"] = 20000
        hh2.spouse()["annual_work_income"] = 15000
        hh2.user()["monthly_rent_spending"] = 800
        hh2.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.user()["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh2.user()["receives_ssi"] = True
        assert not cls.__call__(hh2)

        # Test case 3: Name on lease missing
        hh3 = nuclear_family()
        hh3.user()["age"] = 25
        hh3.user()["name_is_on_lease"] = False  # Name not on lease
        hh3.user()["annual_work_income"] = 20000
        hh3.spouse()["annual_work_income"] = 15000
        hh3.user()["monthly_rent_spending"] = 800
        hh3.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.user()["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh3.user()["receives_ssi"] = True
        assert not cls.__call__(hh3)

        # Test case 4: Combined income exceeds $50,000
        hh4 = nuclear_family()
        hh4.user()["age"] = 25
        hh4.user()["name_is_on_lease"] = True
        hh4.user()["annual_work_income"] = 40000  # Exceeds limit
        hh4.spouse()["annual_work_income"] = 20000
        hh4.user()["monthly_rent_spending"] = 800
        hh4.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.user()["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh4.user()["receives_ssi"] = True
        assert not cls.__call__(hh4)

        # Test case 5: Rent spending less than one-third of income
        hh5 = nuclear_family()
        hh5.user()["age"] = 25
        hh5.user()["name_is_on_lease"] = True
        hh5.user()["annual_work_income"] = 20000
        hh5.spouse()["annual_work_income"] = 15000
        hh5.user()["monthly_rent_spending"] = 500  # Less than 1/3
        hh5.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh5.user()["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh5.user()["receives_ssi"] = True
        assert not cls.__call__(hh5)

        # Test case 6: Invalid housing type
        hh6 = nuclear_family()
        hh6.user()["age"] = 25
        hh6.user()["name_is_on_lease"] = True
        hh6.user()["annual_work_income"] = 20000
        hh6.spouse()["annual_work_income"] = 15000
        hh6.user()["monthly_rent_spending"] = 800
        hh6.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh6.user()["housing_type"] = HousingEnum.HOUSE_2B.value  # Invalid housing type
        hh6.user()["receives_ssi"] = True
        assert not cls.__call__(hh6)

        # Test case 7: Missing income from eligible benefits
        hh7 = nuclear_family()
        hh7.user()["age"] = 25
        hh7.user()["name_is_on_lease"] = True
        hh7.user()["annual_work_income"] = 20000
        hh7.spouse()["annual_work_income"] = 15000
        hh7.user()["monthly_rent_spending"] = 800
        hh7.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh7.user()["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh7.user()["receives_ssi"] = False  # No eligible income source
        assert not cls.__call__(hh7)

        # Additional edge cases can be added as needed to cover every scenario.


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
                    RelationEnum.CHILD.value,
                    RelationEnum.STEPCHILD.value,
                    RelationEnum.FOSTER_CHILD.value,
                    RelationEnum.GRANDCHILD.value,
                    RelationEnum.ADOPTED_CHILD.value,
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
            else:
                raise NotImplementedError  # should never happen

        def _r4(hh) -> bool:
            qualifying_children = []

            for m in hh.members:
                if m["relation"] in [
                    # "child",
                    # "stepchild",
                    # "foster_child",
                    # "grandchild",
                    RelationEnum.CHILD.value,
                    RelationEnum.STEPCHILD.value,
                    RelationEnum.FOSTER_CHILD.value,
                    RelationEnum.GRANDCHILD.value,
                    RelationEnum.ADOPTED_CHILD.value,
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
                lived_together_last_6_months_members = []

                for m in hh.members:
                    if m["lived_together_last_6_months"]:
                        lived_together_last_6_months_members.append(m)
                if bool(lived_together_last_6_months_members):
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

    @classmethod
    def test_cases(cls):
        # Test 1: Married with qualifying children, earning under $63,398
        hh1 = nuclear_family()
        hh1.user()["annual_work_income"] = 40000
        hh1.spouse()["annual_work_income"] = 20000
        hh1.user()["filing_jointly"] = True
        hh1.spouse()["filing_jointly"] = True
        hh1.members[2]["relation"] = RelationEnum.CHILD.value
        hh1.members[2]["age"] = 10
        assert cls.__call__(hh1)

        # Test 2: Married with no qualifying children, earning under $24,210
        hh2 = nuclear_family()
        hh2.user()["annual_work_income"] = 12000
        hh2.spouse()["annual_work_income"] = 8000
        assert cls.__call__(hh2)

        # Test 3: Single with qualifying children, earning under $56,838
        hh3 = nuclear_family()
        hh3.spouse()["annual_work_income"] = 0  # Spouse does not exist
        hh3.user()["annual_work_income"] = 50000
        hh3.members[2]["relation"] = RelationEnum.CHILD.value
        hh3.members[2]["age"] = 8
        assert cls.__call__(hh3)

        # Test 4: Single with no qualifying children, earning under $17,640
        hh4 = nuclear_family()
        hh4.spouse()["annual_work_income"] = 0
        hh4.user()["annual_work_income"] = 16000
        hh4.members[2]["relation"] = RelationEnum.OTHER_NON_FAMILY.value
        hh4.members[2]["age"] = 10
        assert cls.__call__(hh4)

        # Test 5: No children, filer age between 25 and 64
        hh5 = nuclear_family()
        hh5.spouse()["annual_work_income"] = 0
        hh5.user()["annual_work_income"] = 15000
        hh5.user()["age"] = 30
        hh5.members[2]["relation"] = RelationEnum.OTHER_NON_FAMILY.value
        assert cls.__call__(hh5)

        # Test 6: Married Filing Separately, with a qualifying child
        hh6 = nuclear_family()
        hh6.spouse()["filing_jointly"] = False
        hh6.user()["filing_jointly"] = False
        hh6.user()["annual_work_income"] = 30000
        hh6.members[2]["relation"] = RelationEnum.CHILD.value
        hh6.members[2]["age"] = 5
        hh6.members[2]["lived_together_last_6_months"] = True
        assert cls.__call__(hh6)

        # Test 8: Invalid case - investment income over $11,000
        hh8 = nuclear_family()
        hh8.user()["annual_work_income"] = 50000
        hh8.user()["annual_investment_income"] = 12000
        hh8.members[2]["relation"] = RelationEnum.CHILD.value
        hh8.members[2]["age"] = 15
        assert not cls.__call__(hh8)

        # Test 9: Invalid case - no valid Social Security Number
        hh9 = nuclear_family()
        hh9.user()["has_ssn"] = False
        hh9.user()["annual_work_income"] = 50000
        hh9.members[2]["relation"] = RelationEnum.CHILD.value
        hh9.members[2]["age"] = 15
        assert not cls.__call__(hh9)

        # Test 10: Invalid case - no qualifying children and filer under 25
        hh10 = nuclear_family()
        hh10.user()["age"] = 20
        hh10.user()["annual_work_income"] = 15000
        hh10.members[2]["relation"] = RelationEnum.OTHER_NON_FAMILY.value
        assert not cls.__call__(hh10)

        # Test 11: Invalid case - married with no children earning over $24,210
        hh11 = nuclear_family()
        hh11.user()["annual_work_income"] = 20000
        hh11.spouse()["annual_work_income"] = 5000
        hh11.members[2]["relation"] = RelationEnum.OTHER_NON_FAMILY.value
        assert not cls.__call__(hh11)

        # Test 12: Invalid case - married filing separately with no qualifying child
        hh12 = nuclear_family()
        hh12.user()["annual_work_income"] = 30000
        hh12.spouse()["filing_jointly"] = False
        hh12.members[2]["relation"] = RelationEnum.OTHER_NON_FAMILY.value
        assert not cls.__call__(hh12)


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
            if hh.user()["housing_type"] == HousingEnum.TEMPORARY_HOUSING.value:
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
            if bool(
                [
                    m
                    for m in hh.members
                    if m["relation"] == RelationEnum.FOSTER_CHILD.value
                ]
            ):
                return True
            return False

        def _r6(hh) -> bool:
            hh_income = hh.hh_annual_total_income()
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

    @classmethod
    def test_cases(cls):
        # 1. Child aged 3-4 and living in temporary housing
        hh1 = nuclear_family()
        hh1.members[2]["age"] = 4
        hh1.user()["housing_type"] = HousingEnum.TEMPORARY_HOUSING.value
        assert cls.__call__(hh1)

        # 2. Child aged 3-4 and receiving HRA Cash Assistance
        hh2 = nuclear_family()
        hh2.members[2]["age"] = 3
        hh2.user()["receives_temporary_assistance"] = True
        assert cls.__call__(hh2)

        # 3. Child aged 3-4 and receiving SNAP
        hh3 = nuclear_family()
        hh3.members[2]["age"] = 4
        hh3.user()["receives_snap"] = True
        assert cls.__call__(hh3)

        # 4. Child aged 3-4 and receiving SSI
        hh4 = nuclear_family()
        hh4.members[2]["age"] = 3
        hh4.user()["receives_ssi"] = True
        assert cls.__call__(hh4)

        # 5. Child aged 3-4 and in foster care
        hh5 = nuclear_family()
        hh5.members[2]["age"] = 3
        hh5.members[2]["in_foster_care"] = True
        assert cls.__call__(hh5)

        # 6. Child aged 3-4 and family income below threshold for household size
        hh6 = nuclear_family()
        hh6.members[2]["age"] = 3
        hh6.user()["annual_work_income"] = 10000
        hh6.user()["annual_investment_income"] = 5000
        assert cls.__call__(hh6)

        # 7. Child aged 3-4, income just below threshold for household size of 3
        hh7 = nuclear_family()
        hh7.members[2]["age"] = 4
        hh7.user()["annual_work_income"] = 10000
        hh7.user()["annual_investment_income"] = 15000
        assert cls.__call__(hh7)

        # 8. Child aged 3-4, income just above threshold for household size of 3
        hh8 = nuclear_family()
        hh8.members[2]["age"] = 4
        hh8.user()["annual_work_income"] = 20000
        hh8.user()["annual_investment_income"] = 10000
        assert not cls.__call__(hh8)

        # 9. No eligible child, but household meets income requirements
        hh9 = nuclear_family()
        hh9.members[2]["age"] = 5
        hh9.user()["annual_work_income"] = 10000
        hh9.user()["annual_investment_income"] = 15000
        assert not cls.__call__(hh9)

        # 10. Child aged 3-4, multiple qualifying conditions (e.g., receiving SNAP and in temporary housing)
        hh10 = nuclear_family()
        hh10.members[2]["age"] = 4
        hh10.user()["receives_snap"] = True
        hh10.user()["housing_type"] = HousingEnum.TEMPORARY_HOUSING.value
        assert cls.__call__(hh10)

        # 11. Child aged 3-4, no qualifying conditions met
        hh11 = nuclear_family()
        hh11.user()["annual_work_income"] = 999999
        hh11.members[2]["age"] = 3
        assert not cls.__call__(hh11)

        # 12. Larger household size (5) with income below threshold
        hh12 = nuclear_family()
        # hh12.members.append({"age": 8})  # Adding another child
        hh12.members.append(hh12.members[2])
        hh12.members.append(hh12.members[2])
        hh12.members[2]["age"] = 4
        hh12.user()["annual_work_income"] = 30000
        assert cls.__call__(hh12)

        # 13. Larger household size (5) with income above threshold
        hh13 = nuclear_family()
        hh12.members.append(hh12.members[2])
        hh12.members.append(hh12.members[2])
        hh13.members[2]["age"] = 3
        hh13.user()["annual_work_income"] = 40000
        assert not cls.__call__(hh13)

        # 14. Edge case: Child aged 2 (not 3-4), income below threshold
        hh14 = nuclear_family()
        hh14.members[2]["age"] = 2
        hh14.user()["annual_work_income"] = 15000
        assert not cls.__call__(hh14)

        # 15. Edge case: Child aged 5 (not 3-4), meeting other qualifying conditions
        hh15 = nuclear_family()
        hh15.members[2]["age"] = 5
        hh15.user()["receives_snap"] = True
        assert not cls.__call__(hh15)

        # 16. Child aged 3-4, receiving SSI, income above threshold
        hh16 = nuclear_family()
        hh16.members[2]["age"] = 4
        hh16.user()["receives_ssi"] = True
        hh16.user()["annual_work_income"] = 50000
        assert cls.__call__(hh16)


class BasicSchoolTaxReliefProgram(BaseBenefitsProgram):
    """
    To be eligible for Basic STAR, you should be a homeowner of one of these types of housing:

        a house
        a condo
        a cooperative apartment
        a manufactured home
        a farmhouse
        a mixed-use property, including apartment buildings (only the owner-occupied portion is eligible)
    Age
        No age restriction
    Primary residence
        An owner must live on the property as their primary residence.
    Income
        The total income of only the owners and their spouses who live at the property must be:
        $500,000 or less for the credit
        $250,000 or less for the exemption (you cannot apply for the exemption anymore but you can restore it if you got it in 2015-16 but lost the benefit later.)
    """

    @staticmethod
    def __call__(hh):
        def _is_eligible_homeowner(hh):
            """Check if the user owns an eligible type of housing."""
            eligible_housing_types = [
                HousingEnum.HOUSE_2B.value,
                HousingEnum.CONDO.value,
                HousingEnum.COOPERATIVE_APARTMENT.value,
                HousingEnum.MANUFACTURED_HOME.value,
                HousingEnum.FARMHOUSE.value,
                HousingEnum.MIXED_USE_PROPERTY.value,
            ]
            return hh.user().get("housing_type") in eligible_housing_types

        def _basic_star_primary_residence(hh):
            """
            An owner must live on the property as their primary residence.
            """
            return hh.user().get("primary_residence", False)

        def _basic_star_income(hh):
            """
            The total income of the owners and their spouses who live at the property
            must be <= $500,000 (for the credit version).
            """
            owners = [hh.user()]
            spouse = hh.spouse()
            if spouse and spouse.get("primary_residence", False):
                owners.append(spouse)

            total_income = sum(owner.total_income() for owner in owners)
            return total_income <= 500000

        # Check eligibility for Basic STAR
        basic_star_eligible = (
            _is_eligible_homeowner(hh)
            and _basic_star_primary_residence(hh)
            and _basic_star_income(hh)
        )

        return basic_star_eligible

    @classmethod
    def test_cases(cls):
        # Test Case 1: Valid homeowner with eligible housing type and primary residence
        hh1 = nuclear_family()
        for member in hh1.members:
            member["housing_type"] = HousingEnum.CONDO.value
            member["primary_residence"] = True
        hh1.user()["annual_work_income"] = 100000
        hh1.spouse()["annual_work_income"] = 50000
        assert cls.__call__(hh1)

        # Test Case 2: Invalid homeowner with ineligible housing type
        hh2 = nuclear_family()
        for member in hh2.members:
            member["housing_type"] = HousingEnum.HOMELESS.value
            member["primary_residence"] = True
        hh2.user()["annual_work_income"] = 100000
        hh2.spouse()["annual_work_income"] = 50000
        assert not cls.__call__(hh2)

        # Test Case 3: Invalid household where total income exceeds the threshold
        hh3 = nuclear_family()
        for member in hh3.members:
            member["housing_type"] = HousingEnum.HOUSE_2B.value
            member["primary_residence"] = True
        hh3.user()["annual_work_income"] = 300000
        hh3.spouse()["annual_work_income"] = 250000
        assert not cls.__call__(hh3)

        # Test Case 4: Valid household with multiple children but within income threshold
        hh4 = nuclear_family()
        hh4.members.append(deepcopy(hh4.members[-1]))  # Add one more child
        for member in hh4.members:
            member["housing_type"] = HousingEnum.COOPERATIVE_APARTMENT.value
            member["primary_residence"] = True
        hh4.user()["annual_work_income"] = 200000
        hh4.spouse()["annual_work_income"] = 250000
        assert cls.__call__(hh4)

        # Test Case 5: Invalid household with ineligible primary residence status
        hh5 = nuclear_family()
        for member in hh5.members:
            member["housing_type"] = HousingEnum.FARMHOUSE.value
            member["primary_residence"] = False
        hh5.user()["annual_work_income"] = 100000
        hh5.spouse()["annual_work_income"] = 50000
        assert not cls.__call__(hh5)

        # Test Case 6: Valid household restoring exemption with past eligibility
        hh6 = nuclear_family()
        for member in hh6.members:
            member["housing_type"] = HousingEnum.MIXED_USE_PROPERTY.value
            member["primary_residence"] = True
        hh6.user()["annual_work_income"] = 0
        hh6.spouse()["annual_work_income"] = 0
        hh6.user()["annual_investment_income"] = 100000
        assert cls.__call__(hh6)


class EnhancedSchoolTaxReliefProgram(BaseBenefitsProgram):
    """
    To be eligible for Enhanced STAR, you should be a homeowner of one of these types of housing:

        a house
        a condo
        a cooperative apartment
        a manufactured home
        a farmhouse
        a mixed-use property, including apartment buildings (only the owner-occupied portion is eligible)

    Age
        All owners must be 65 or older as of December 31 of the year of the exemption.
        However, only one owner needs to be 65 or older if the property is jointly owned by only a married couple or only siblings.

    Primary residence
        At least one owner who's 65 or older must live on the property as their primary residence.

    Income
        Total income of all owners and resident spouses or registered domestic partners must be $98,700 or less.
    """

    @staticmethod
    def __call__(hh):
        def _is_eligible_homeowner(hh):
            """Check if the user owns an eligible type of housing."""
            eligible_housing_types = [
                HousingEnum.HOUSE_2B.value,
                HousingEnum.CONDO.value,
                HousingEnum.COOPERATIVE_APARTMENT.value,
                HousingEnum.MANUFACTURED_HOME.value,
                HousingEnum.FARMHOUSE.value,
                HousingEnum.MIXED_USE_PROPERTY.value,
            ]
            return hh.user().get("housing_type") in eligible_housing_types

        def _enhanced_star_age(hh):
            """
            - All owners must be 65+ unless jointly owned by a married couple or siblings,
              in which case at least one owner must be 65+.
            """
            owners = [hh.user()]
            co_owners = hh.features.get("co_owners", [])
            owners.extend(co_owners)

            # If all owners are 65 or older
            if all(owner["age"] >= 65 for owner in owners):
                return True

            # If exactly two owners: check married couple or siblings scenario
            if len(owners) == 2:
                user_owner = hh.user()
                # Check if they are a married couple filing jointly and at least one is 65
                if user_owner.get("filing_jointly") and any(
                    owner["age"] >= 65 for owner in owners
                ):
                    return True
                # Check if they are siblings and at least one is 65
                if all(
                    owner["relation"] == RelationEnum.SIBLING.value for owner in owners
                ) and any(owner["age"] >= 65 for owner in owners):
                    return True

            return False

        def _enhanced_star_primary_residence(hh):
            """
            At least one owner who is 65 or older must live on the property as their primary residence.
            """
            owners = [hh.user()]
            co_owners = hh.features.get("co_owners", [])
            owners.extend(co_owners)

            return any(
                owner["age"] >= 65 and owner.get("primary_residence", False)
                for owner in owners
            )

        def _enhanced_star_income(hh):
            """
            Total income of all owners and resident spouses/partners must be <= $98,700.
            """
            owners = [hh.user()]
            co_owners = hh.features.get("co_owners", [])
            owners.extend(co_owners)

            resident_spouses = []
            for owner in owners:
                if owner.get("primary_residence", False):
                    # If the owner is the main user, check for a spouse
                    if owner["relation"] == RelationEnum.SELF.value:
                        spouse = hh.spouse()
                        if spouse and spouse.get("primary_residence", False):
                            resident_spouses.append(spouse)

            total_income = sum(
                owner.total_income() for owner in owners + resident_spouses
            )
            return total_income <= 98700

        # Check eligibility for Enhanced STAR
        enhanced_star_eligible = (
            _is_eligible_homeowner(hh)
            and _enhanced_star_age(hh)
            and _enhanced_star_primary_residence(hh)
            and _enhanced_star_income(hh)
        )

        return enhanced_star_eligible

    @classmethod
    def test_cases(cls):
        # Test Case 1: Eligible - Single owner, age 65+, primary residence, income within limit
        hh1 = nuclear_family()
        hh1.members[0]["housing_type"] = HousingEnum.HOUSE_2B.value
        hh1.members[0]["age"] = 66
        hh1.members[0]["primary_residence"] = True
        hh1.members[0]["annual_work_income"] = 40000
        hh1.members[0]["annual_investment_income"] = 20000
        hh1.members[1]["annual_work_income"] = 15000
        hh1.members[1]["annual_investment_income"] = 10000
        assert cls.__call__(hh1)

        # Test Case 2: Eligible - Married couple, one age 65+, primary residence, income within limit
        hh2 = nuclear_family()
        hh2.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[0]["age"] = 65
        hh2.members[1]["age"] = 60
        hh2.members[0]["primary_residence"] = True
        hh2.members[1]["primary_residence"] = True
        hh2.members[0]["annual_work_income"] = 50000
        hh2.members[1]["annual_work_income"] = 30000
        assert cls.__call__(hh2)

        # Test Case 3: Not Eligible - All members under age 65
        hh3 = nuclear_family()
        hh3.members[0]["housing_type"] = HousingEnum.FARMHOUSE.value
        hh3.members[0]["age"] = 64
        hh3.members[1]["age"] = 64
        hh3.members[0]["primary_residence"] = True
        hh3.members[1]["primary_residence"] = True
        hh3.members[0]["annual_work_income"] = 40000
        hh3.members[1]["annual_work_income"] = 30000
        assert not cls.__call__(hh3)

        # Test Case 4: Not Eligible - Income exceeds the limit
        hh4 = nuclear_family()
        hh4.members[0]["housing_type"] = HousingEnum.MANUFACTURED_HOME.value
        hh4.members[0]["age"] = 67
        hh4.members[1]["age"] = 65
        hh4.members[0]["primary_residence"] = True
        hh4.members[1]["primary_residence"] = True
        hh4.members[0]["annual_work_income"] = 70000
        hh4.members[1]["annual_work_income"] = 40000
        assert not cls.__call__(hh4)

        # Test Case 5: Eligible - Mixed-use property, one owner age 65+, primary residence, income within limit
        hh5 = nuclear_family()
        hh5.members[0]["housing_type"] = HousingEnum.MIXED_USE_PROPERTY.value
        hh5.members[0]["age"] = 68
        hh5.members[1]["age"] = 60
        hh5.members[0]["primary_residence"] = True
        hh5.members[1]["primary_residence"] = True
        hh5.members[0]["annual_work_income"] = 20000
        hh5.members[1]["annual_work_income"] = 30000
        assert cls.__call__(hh5)

        # Test Case 6: Not Eligible - Primary residence condition not met
        hh6 = nuclear_family()
        hh6.members[0]["housing_type"] = HousingEnum.COOPERATIVE_APARTMENT.value
        hh6.members[0]["age"] = 70
        hh6.members[1]["age"] = 68
        hh6.members[0]["primary_residence"] = False
        hh6.members[1]["primary_residence"] = False
        hh6.members[0]["annual_work_income"] = 30000
        hh6.members[1]["annual_work_income"] = 20000
        assert not cls.__call__(hh6)


class SectionEightHousingChoiceVoucherProgram(BaseBenefitsProgram):
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

            return hh.hh_annual_total_income() <= income_limit

        return _income_eligible(hh)

        # ChatGPT Link - https://chatgpt.com/c/6748084a-bf98-8002-83e0-cc8e4d80c137
        hh_size = hh.num_members()
        total_income = hh.hh_annual_total_income()

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

    @classmethod
    def test_cases(cls):

        # Household of 3, combined income below threshold
        hh3 = nuclear_family()
        hh3.user()["annual_work_income"] = 40000
        hh3.spouse()["annual_work_income"] = 20000
        hh3.members[2]["annual_work_income"] = 5000
        assert cls.__call__(hh3)

        # Household of 3, combined income at threshold
        hh3.spouse()["annual_work_income"] = 24900
        assert cls.__call__(hh3)

        # Household of 3, combined income above threshold
        hh3.spouse()["annual_work_income"] = 30000
        assert not cls.__call__(hh3)

        # Household of 4, income below threshold with additional child
        hh4 = nuclear_family()
        hh4.members.append(hh4.members[-1])  # Add a fourth member
        hh4.user()["annual_work_income"] = 50000
        hh4.spouse()["annual_work_income"] = 20000
        assert cls.__call__(hh4)

        # Household of 4, income at threshold
        hh4.spouse()["annual_work_income"] = 27650
        assert cls.__call__(hh4)

        # Household of 4, income above threshold
        hh4.spouse()["annual_work_income"] = 30000
        assert not cls.__call__(hh4)

        # Test large household (8 members), income below threshold
        hh8 = nuclear_family()
        for _ in range(5):
            hh8.members.append(hh8.members[-1])  # Add 5 additional members
        hh8.user()["annual_work_income"] = 70000
        hh8.spouse()["annual_work_income"] = 30000
        assert cls.__call__(hh8)

        # Test large household (8 members), income above threshold
        hh8.spouse()["annual_work_income"] = 40000
        assert not cls.__call__(hh8)


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
        valid_housing_types = [
            HousingEnum.HOUSE_2B.value,
            HousingEnum.CONDO.value,
            HousingEnum.COOPERATIVE_APARTMENT.value,
        ]
        # housing_type = hh.get_housing_type()  # e.g., "one_family_home", "condo", etc.
        housing_type = hh.user()["housing_type"]
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
            rel1, rel2 = p1["relation"], p2["relation"]
            # Basic logic: If either reports "spouse" or "sibling", treat them accordingly
            return (RelationEnum.SPOUSE.value in [rel1, rel2]) or (
                RelationEnum.SIBLING.value in [rel1, rel2]
            )

        # 2A. Check the age requirement
        if len(owners) == 1:
            # If there is only one owner, that owner must be at least 65
            if owners[0]["age"] < 65:
                return False

        elif len(owners) == 2:
            # If there are exactly two owners, check if they are spouses or siblings
            if are_spouses_or_siblings(owners[0], owners[1]):
                # Only one must be 65+
                if not any(o["age"] >= 65 for o in owners):
                    return False
            else:
                # Otherwise, both must be 65+
                if not all(o["age"] >= 65 for o in owners):
                    return False

        else:
            # If more than two owners, all must be 65+
            if not all(o["age"] >= 65 for o in owners):
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
            if not o["had_previous_sche"] and o["months_owned_property"] < 12:
                return False

        return True

    @classmethod
    def test_cases(cls):
        # Test 1: All conditions met
        hh1 = nuclear_family()
        hh1.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh1.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh1.members[2]["housing_type"] = HousingEnum.CONDO.value
        hh1.members[0]["is_property_owner"] = True
        hh1.members[1]["is_property_owner"] = True
        hh1.members[0]["age"] = 70
        hh1.members[1]["age"] = 65
        hh1.members[0]["primary_residence"] = True
        hh1.members[1]["primary_residence"] = True
        hh1.members[0]["annual_work_income"] = 20000
        hh1.members[1]["annual_work_income"] = 10000
        hh1.members[0]["months_owned_property"] = 24
        hh1.members[1]["months_owned_property"] = 24
        assert cls.__call__(hh1)

        # Test 2: Property type not eligible
        hh2 = nuclear_family()
        hh2.members[0]["housing_type"] = HousingEnum.RENT_CONTROLLED_APARTMENT.value
        hh2.members[1]["housing_type"] = HousingEnum.RENT_CONTROLLED_APARTMENT.value
        hh2.members[2]["housing_type"] = HousingEnum.RENT_CONTROLLED_APARTMENT.value
        hh2.members[0]["age"] = 70
        hh2.members[1]["age"] = 65
        hh2.members[0]["is_property_owner"] = True
        hh2.members[1]["is_property_owner"] = True
        hh2.members[0]["primary_residence"] = True
        hh2.members[1]["primary_residence"] = True
        hh2.members[0]["annual_work_income"] = 20000
        hh2.members[1]["annual_work_income"] = 10000
        hh2.members[0]["months_owned_property"] = 24
        hh2.members[1]["months_owned_property"] = 24
        assert not cls.__call__(hh2)

        # Test 3: At least one owner under 65, no exception
        hh3 = nuclear_family()
        hh3.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh3.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh3.members[2]["housing_type"] = HousingEnum.CONDO.value
        hh3.members[0]["age"] = 70
        hh3.members[1]["age"] = 60
        hh3.members[1]["relation"] = RelationEnum.OTHER_NON_FAMILY.value
        hh3.members[0]["is_property_owner"] = True
        hh3.members[1]["is_property_owner"] = True
        hh3.members[0]["primary_residence"] = True
        hh3.members[1]["primary_residence"] = True
        hh3.members[0]["annual_work_income"] = 20000
        hh3.members[1]["annual_work_income"] = 10000
        hh3.members[0]["months_owned_property"] = 24
        hh3.members[1]["months_owned_property"] = 24
        assert not cls.__call__(hh3)

        # Test 4: Combined income exceeds the limit
        hh4 = nuclear_family()
        hh4.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh4.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh4.members[2]["housing_type"] = HousingEnum.CONDO.value
        hh4.members[0]["age"] = 70
        hh4.members[1]["age"] = 65
        hh4.members[0]["is_property_owner"] = True
        hh4.members[1]["is_property_owner"] = True
        hh4.members[0]["primary_residence"] = True
        hh4.members[1]["primary_residence"] = True
        hh4.members[0]["annual_work_income"] = 30000
        hh4.members[1]["annual_work_income"] = 30000
        hh4.members[0]["months_owned_property"] = 24
        hh4.members[1]["months_owned_property"] = 24
        assert not cls.__call__(hh4)

        # Test 5: Not all owners own the property
        hh5 = nuclear_family()
        hh5.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh5.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh5.members[2]["housing_type"] = HousingEnum.CONDO.value
        hh5.members[0]["age"] = 70
        hh5.members[1]["age"] = 65
        hh5.members[0]["is_property_owner"] = True
        hh5.members[1]["is_property_owner"] = True
        hh5.members[0]["primary_residence"] = True
        hh5.members[1]["primary_residence"] = False
        hh5.members[0]["annual_work_income"] = 20000
        hh5.members[1]["annual_work_income"] = 10000
        hh5.members[0]["months_owned_property"] = 24
        hh5.members[1]["months_owned_property"] = 24
        assert not cls.__call__(hh5)

        # Test 6: Ownership duration less than 12 months
        hh6 = nuclear_family()
        hh6.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh6.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh6.members[2]["housing_type"] = HousingEnum.CONDO.value
        hh6.members[0]["age"] = 70
        hh6.members[1]["age"] = 65
        hh6.members[0]["is_property_owner"] = True
        hh6.members[1]["is_property_owner"] = True
        hh6.members[0]["primary_residence"] = True
        hh6.members[1]["primary_residence"] = True
        hh6.members[0]["annual_work_income"] = 20000
        hh6.members[1]["annual_work_income"] = 10000
        hh6.members[0]["months_owned_property"] = 6
        hh6.members[1]["months_owned_property"] = 6
        assert not cls.__call__(hh6)


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
    def __call__(hh):
        num_members = len(hh.members)
        income = hh.hh_annual_total_income() / 12
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
                CitizenshipEnum.CITIZEN_OR_NATIONAL.value,
                CitizenshipEnum.LAWFUL_RESIDENT.value,
            ]:
                return False
            return True

        for m in hh.members:
            if user_is_eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Case 1: Single-member household, eligible
        hh1 = nuclear_family()
        hh1.members = hh1.members[:1]  # Single-member household
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh1.members[0]["annual_work_income"] = 2500  # Below threshold for family size 1
        assert cls.__call__(hh1)

        # Case 2: Family of 3, eligible
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value

        hh2.members[0]["citizenship"] = CitizenshipEnum.LAWFUL_RESIDENT.value
        hh2.members[1]["citizenship"] = CitizenshipEnum.LAWFUL_RESIDENT.value
        hh2.members[2]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value

        hh2.members[0]["annual_work_income"] = 2000
        hh2.members[1]["annual_work_income"] = 1000
        hh2.members[2][
            "annual_work_income"
        ] = 500  # Total income = 3500, below threshold for 3 members
        assert cls.__call__(hh2)

        # Case 3: Family of 4, not eligible (income too high)
        hh3 = nuclear_family()
        hh3.members.append(hh3.members[-1])  # Add a fourth member
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[3]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value

        hh3.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh3.members[1]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh3.members[2]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh3.members[3]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value

        hh3.members[0]["annual_work_income"] = 3000
        hh3.members[1]["annual_work_income"] = 3000
        hh3.members[2]["annual_work_income"] = 500
        hh3.members[3][
            "annual_work_income"
        ] = 999999  # Total income = 7000, above threshold for 4 members
        assert not cls.__call__(hh3)

        # Case 4: Non-NYC resident, not eligible
        hh4 = nuclear_family()
        hh4.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value

        hh4.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh4.members[1]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh4.members[2]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value

        hh4.members[0]["annual_work_income"] = 2000
        hh4.members[1]["annual_work_income"] = 1000
        hh4.members[2][
            "annual_work_income"
        ] = 500  # Total income = 3500, below threshold for 3 members
        assert not cls.__call__(hh4)

        # Case 5: Household with non-citizen members, not eligible
        hh5 = nuclear_family()
        hh5.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh5.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh5.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value

        hh5.members[0]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh5.members[1]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh5.members[2]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value

        hh5.members[0]["annual_work_income"] = 1000
        hh5.members[1]["annual_work_income"] = 1000
        hh5.members[2][
            "annual_work_income"
        ] = 1000  # Total income = 3000, below threshold for 3 members
        assert not cls.__call__(hh5)

        # Case 6: Family of 8, eligible
        hh6 = nuclear_family()
        for _ in range(5):
            hh6.members.append(hh6.members[-1])  # Add 5 additional members to make 8
        for member in hh6.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
            member["annual_work_income"] = (
                1000  # Total income = 8000, below threshold for 8 members
            )
        assert cls.__call__(hh6)


class IDNYC(BaseBenefitsProgram):
    """Anyone who lives in NYC and is age 10 and older is eligible to apply for an IDNYC card."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] != PlaceOfResidenceEnum.NYC.value:
                return False
            if m["age"] < 10:
                return False
            return True

        for m in hh.members:
            if eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: Default household, no one eligible
        hh1 = nuclear_family()
        for member in hh1.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        assert not cls.__call__(hh1)

        # Test case 2: Default household, user eligible (age 40, lives in NYC)
        hh2 = nuclear_family()
        for member in hh2.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.user()["age"] = 40
        hh2.spouse()["age"] = 40
        hh2.members[2]["age"] = 9
        assert cls.__call__(hh2)

        # Test case 3: Default household, child eligible (age 10, lives in NYC)
        hh3 = nuclear_family()
        for member in hh3.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.user()["age"] = 40
        hh3.spouse()["age"] = 40
        hh3.members[2]["age"] = 10
        assert cls.__call__(hh3)

        # Test case 4: Household with an additional eligible child
        hh4 = nuclear_family()
        for member in hh4.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.user()["age"] = 40
        hh4.spouse()["age"] = 40
        hh4.members[2]["age"] = 10
        hh4.members.append(hh4.members[-1])
        hh4.members[-1]["age"] = 15
        assert cls.__call__(hh4)

        # Test case 5: Household with no one eligible due to location
        hh5 = nuclear_family()
        for member in hh5.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh5.user()["age"] = 40
        hh5.spouse()["age"] = 40
        hh5.members[2]["age"] = 10
        assert not cls.__call__(hh5)

        # Test case 6: Household with all members eligible
        hh6 = nuclear_family()
        for member in hh6.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh6.user()["age"] = 40
        hh6.spouse()["age"] = 40
        hh6.members[2]["age"] = 10
        hh6.members.append(hh6.members[-1])
        hh6.members[-1]["age"] = 20
        assert cls.__call__(hh6)

        # Test case 7: Household with no one eligible due to age
        hh7 = nuclear_family()
        for member in hh7.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh7.user()["age"] = 9
        hh7.spouse()["age"] = 9
        hh7.members[2]["age"] = 9
        assert not cls.__call__(hh7)


class OfficeOfChildSupportServices(BaseBenefitsProgram):
    """To be eligible for child support services, you should be able to answer yes to these questions:

    Are you a singleparent or legal guardian of a child under the age of 21?
    Are you primarily responsible for the child's day-to-day life?"""

    @staticmethod
    def __call__(hh):
        has_spouse = False
        for m in hh.members:
            if m["relation"] == RelationEnum.SPOUSE.value:
                has_spouse = True
        if has_spouse:
            return False
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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Single parent, responsible for day-to-day life, child under 21
        hh1 = nuclear_family()
        hh1.members[0]["relation"] = RelationEnum.SELF.value
        hh1.members[1]["relation"] = RelationEnum.CHILD.value
        hh1.members[1]["age"] = 5
        hh1.members[0]["responsible_for_day_to_day"] = True
        assert cls.__call__(hh1)

        # Test Case 2: Both parents in household, not eligible
        hh2 = nuclear_family()
        hh2.members[0]["relation"] = RelationEnum.SELF.value
        hh2.members[1]["relation"] = RelationEnum.SPOUSE.value
        hh2.members[2]["relation"] = RelationEnum.CHILD.value
        hh2.members[2]["age"] = 10
        hh2.members[0]["responsible_for_day_to_day"] = True
        hh2.members[1]["responsible_for_day_to_day"] = True
        assert not cls.__call__(hh2)

        # Test Case 3: Single parent but child is 21 or older, not eligible
        hh3 = nuclear_family()
        hh3.members = hh3.members[:2]
        hh3.members[0]["relation"] = RelationEnum.SELF.value
        hh3.members[1]["relation"] = RelationEnum.CHILD.value
        hh3.members[1]["age"] = 21
        hh3.members[0]["responsible_for_day_to_day"] = True
        assert not cls.__call__(hh3)

        # Test Case 4: Legal guardian responsible for child under 21
        hh4 = nuclear_family()
        hh4.members[0]["relation"] = RelationEnum.OTHER_FAMILY.value
        hh4.members[1]["relation"] = RelationEnum.CHILD.value
        hh4.members[1]["age"] = 8
        hh4.members[0]["responsible_for_day_to_day"] = True
        assert cls.__call__(hh4)

        # Test Case 5: No children in household, not eligible
        hh5 = nuclear_family()
        hh5.members = hh5.members[:2]  # Remove child
        hh5.members[0]["relation"] = RelationEnum.SELF.value
        hh5.members[1]["relation"] = RelationEnum.SPOUSE.value
        hh5.members[0]["responsible_for_day_to_day"] = True
        assert not cls.__call__(hh5)


class HIVAIDSServicesAdministration(BaseBenefitsProgram):
    """To be eligible for HASA, you must have been diagnosed with HIV or with AIDS, as defined by the Centers for Disease Control and Prevention (CDC). You do not need to have symptoms to be eligible."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["hiv_aids"]:
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: User diagnosed with HIV
        hh1 = nuclear_family()
        hh1.user()["hiv_aids"] = True
        hh1.spouse()["hiv_aids"] = False
        hh1.members[2]["hiv_aids"] = False
        assert cls.__call__(hh1)

        # Test case 2: Spouse diagnosed with HIV
        hh2 = nuclear_family()
        hh2.user()["hiv_aids"] = False
        hh2.spouse()["hiv_aids"] = True
        hh2.members[2]["hiv_aids"] = False
        assert cls.__call__(hh2)

        # Test case 3: Child diagnosed with HIV
        hh3 = nuclear_family()
        hh3.user()["hiv_aids"] = False
        hh3.spouse()["hiv_aids"] = False
        hh3.members[2]["hiv_aids"] = True
        assert cls.__call__(hh3)

        # Test case 4: No one diagnosed with HIV
        hh4 = nuclear_family()
        hh4.user()["hiv_aids"] = False
        hh4.spouse()["hiv_aids"] = False
        hh4.members[2]["hiv_aids"] = False
        assert not cls.__call__(hh4)


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
            if not m["disabled"]:
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

    @classmethod
    def test_cases(cls):
        # Test case 1: Single eligible member
        hh1 = nuclear_family()
        hh1.members = hh1.members[:1]
        hh1.members[0]["age"] = 18  # User
        hh1.members[0]["can_manage_self"] = False
        hh1.members[0]["disabled"] = True
        hh1.members[0]["has_family_to_help"] = False
        assert cls.__call__(hh1)

        # Test case 2: No members meet age requirement
        hh2 = nuclear_family()
        hh2.members[0]["age"] = 17  # User
        hh2.members[1]["age"] = 16  # Spouse
        hh2.members[2]["age"] = 10  # Child
        assert not cls.__call__(hh2)

        # Test case 3: All members meet eligibility
        hh3 = nuclear_family()
        for member in hh3.members:
            member["age"] = 40
            member["can_manage_self"] = False
            member["disabled"] = True
            member["has_family_to_help"] = False
        assert cls.__call__(hh3)

        # Test case 4: Some members meet eligibility
        hh4 = nuclear_family()
        hh4.members[0]["age"] = 18
        hh4.members[0]["can_manage_self"] = False
        hh4.members[0]["disabled"] = True
        hh4.members[0]["has_family_to_help"] = False
        hh4.members[1]["age"] = 40
        hh4.members[1]["can_manage_self"] = True  # Spouse can manage self
        hh4.members[2]["age"] = 10
        assert cls.__call__(hh4)

        # Test case 5: No members have a disability
        hh5 = nuclear_family()
        for member in hh5.members:
            member["age"] = 40
            member["can_manage_self"] = True
            member["disabled"] = False
            member["has_family_to_help"] = True
        assert not cls.__call__(hh5)

        # Test case 6: Member has family willing to help
        hh6 = nuclear_family()
        hh6.members[0]["age"] = 30
        hh6.members[0]["can_manage_self"] = False
        hh6.members[0]["disabled"] = True
        hh6.members[0]["has_family_to_help"] = True  # Has family support
        assert not cls.__call__(hh6)

        # Test case 7: Member cannot manage resources without help
        hh7 = nuclear_family()
        hh7.members[0]["age"] = 18
        hh7.members[0]["can_manage_self"] = False
        hh7.members[0]["disabled"] = True
        hh7.members[0]["has_family_to_help"] = False
        assert cls.__call__(hh7)

        # Test case 8: Adding extra eligible members
        hh8 = nuclear_family()
        hh8.members[0]["age"] = 40
        hh8.members[0]["can_manage_self"] = False
        hh8.members[0]["disabled"] = True
        hh8.members[0]["has_family_to_help"] = False
        hh8.members.append(hh8.members[-1])  # Add another member
        hh8.members[-1]["age"] = 19
        hh8.members[-1]["can_manage_self"] = False
        hh8.members[-1]["disabled"] = True
        hh8.members[-1]["has_family_to_help"] = False
        assert cls.__call__(hh8)

        # Test case 9: Dropping last member to make the household ineligible
        hh9 = nuclear_family()
        hh9.members[0]["age"] = 40
        hh9.members[0]["can_manage_self"] = False
        hh9.members[0]["disabled"] = True
        hh9.members[0]["has_family_to_help"] = False
        hh9.members = hh9.members[:2]  # Drop child
        hh9.members[1]["age"] = 10  # Spouse is not eligible
        hh9.members[1]["can_manage_self"] = True
        assert cls.__call__(hh9)

        # Test case 10: Edge case where all conditions are barely met
        hh10 = nuclear_family()
        hh10.members[0]["age"] = 18
        hh10.members[0]["can_manage_self"] = False
        hh10.members[0]["disabled"] = True
        hh10.members[0]["has_family_to_help"] = False
        hh10.members[1]["age"] = 17  # Spouse under age
        hh10.members[2]["age"] = 10
        assert cls.__call__(hh10)


class AccessARideParatransitService(BaseBenefitsProgram):
    """You are eligible for Access-A-Ride if:
    you have a disability that prevents you from using accessible buses or subways for some or all of your trips OR:
    if you're recovering from surgery, have a long-term condition, or are seeking Paratransit Service during your visit to NYC.
    """

    @staticmethod
    def __call__(hh):
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

    @classmethod
    def test_cases(cls):
        # Test case 1: User has a disability that prevents them from using accessible buses or subways
        hh1 = nuclear_family()
        hh1.members[0]["can_access_subway_or_bus"] = False
        hh1.members[0]["disabled"] = True
        hh1.members[1]["can_access_subway_or_bus"] = True
        hh1.members[1]["disabled"] = False
        hh1.members[2]["can_access_subway_or_bus"] = True
        hh1.members[2]["disabled"] = False
        assert cls.__call__(hh1)

        # Test case 2: Spouse has a disability and cannot use accessible buses or subways
        hh2 = nuclear_family()
        hh2.members[1]["can_access_subway_or_bus"] = False
        hh2.members[1]["disabled"] = True
        hh2.members[0]["can_access_subway_or_bus"] = True
        hh2.members[0]["disabled"] = False
        hh2.members[2]["can_access_subway_or_bus"] = True
        hh2.members[2]["disabled"] = False
        assert cls.__call__(hh2)

        # Test case 3: User is recovering from surgery
        hh3 = nuclear_family()
        hh3.members[0]["recovering_from_surgery"] = True
        hh3.members[1]["recovering_from_surgery"] = False
        hh3.members[2]["recovering_from_surgery"] = False
        assert cls.__call__(hh3)

        # Test case 4: Spouse is recovering from surgery
        hh4 = nuclear_family()
        hh4.members[1]["recovering_from_surgery"] = True
        hh4.members[0]["recovering_from_surgery"] = False
        hh4.members[2]["recovering_from_surgery"] = False
        assert cls.__call__(hh4)


class BeaconPrograms(BaseBenefitsProgram):
    """Youth ages 5-21 who are enrolled in school can participate."""

    @staticmethod
    def __call__(hh):
        def is_eligible(m):
            if m["current_school_level"] != GradeLevelEnum.NONE.value:
                if m["age"] < 5 or m["age"] > 21:
                    return False
                else:
                    return True

        for m in hh.members:
            if is_eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: A child (age 10) enrolled in school
        hh = nuclear_family()
        hh.members[2]["age"] = 10
        hh.members[2]["current_school_level"] = GradeLevelEnum.FIVE.value
        assert cls.__call__(hh)

        # Test Case 2: Youth (age 21) enrolled in college
        hh.members[2]["age"] = 21
        hh.members[2]["current_school_level"] = GradeLevelEnum.COLLEGE.value
        assert cls.__call__(hh)

        # Test Case 3: Youth (age 5) enrolled in kindergarten
        hh.members[2]["age"] = 5
        hh.members[2]["current_school_level"] = GradeLevelEnum.K.value
        assert cls.__call__(hh)

        # Test Case 4: Youth (age 18) not enrolled in school
        hh.members[2]["age"] = 18
        hh.members[2]["current_school_level"] = GradeLevelEnum.NONE.value
        assert not cls.__call__(hh)

        # Test Case 5: Youth (age 22) enrolled in school (outside the age range)
        hh.members[2]["age"] = 22
        hh.members[2]["current_school_level"] = GradeLevelEnum.COLLEGE.value
        assert not cls.__call__(hh)


class NYCFreeTaxPrep(BaseBenefitsProgram):
    """For tax year 2023, youre eligible for NYC Free Tax Prep if you earned:

    $85,000 or less in 2023 and have dependents.
    $59,000 or less in 2023 and dont have dependents."""

    @staticmethod
    def __call__(hh):
        income = hh.hh_annual_total_income()
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

    @classmethod
    def test_cases(cls):
        # Test 1: Eligible Household with Dependents and Income Below Threshold
        hh1 = nuclear_family()
        hh1.user()["annual_work_income"] = 50000
        hh1.spouse()["annual_work_income"] = 30000
        hh1.members[2]["relation"] = RelationEnum.CHILD.value
        hh1.members[2]["age"] = 10
        assert cls.__call__(hh1)

        # Test 2: Ineligible Household with Dependents and Income Above Threshold
        hh2 = nuclear_family()
        hh2.user()["annual_work_income"] = 60000
        hh2.spouse()["annual_work_income"] = 30000
        hh2.members[2]["relation"] = RelationEnum.CHILD.value
        hh2.members[2]["age"] = 10
        assert not cls.__call__(hh2)

        # Test 3: Eligible Individual without Dependents and Income Below Threshold
        hh3 = nuclear_family()
        hh3.spouse()["relation"] = RelationEnum.OTHER_NON_FAMILY.value
        hh3.members = hh3.members[:1]  # Remove spouse and child
        hh3.user()["annual_work_income"] = 50000
        assert cls.__call__(hh3)

        # Test 4: Ineligible Individual without Dependents and Income Above Threshold
        hh4 = nuclear_family()
        hh4.spouse()["relation"] = RelationEnum.OTHER_NON_FAMILY.value
        hh4.members = hh4.members[:1]  # Remove spouse and child
        hh4.user()["annual_work_income"] = 60000
        assert not cls.__call__(hh4)


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

    @classmethod
    def test_cases(cls):
        # Test 1: Eligible household with a 5th-grade child and NYC residency
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["current_school_level"] = GradeLevelEnum.FIVE.value
        assert cls.__call__(hh1)

        # Test 2: Household with a 5th-grade child but lives outside NYC
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[2]["current_school_level"] = GradeLevelEnum.FIVE.value
        assert not cls.__call__(hh2)

        # Test 3: Household living in NYC but the child is not in 5th grade
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["current_school_level"] = GradeLevelEnum.SIX.value
        assert not cls.__call__(hh3)

        # Test 4: Household with multiple children, one eligible and others not
        hh4 = nuclear_family()
        hh4.members.append(deepcopy(hh4.members[-1]))  # Add a second child
        hh4.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[3]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[2][
            "current_school_level"
        ] = GradeLevelEnum.FIVE.value  # First child eligible
        hh4.members[3][
            "current_school_level"
        ] = GradeLevelEnum.SIX.value  # Second child not eligible
        assert cls.__call__(hh4)


class KindergartenAndElementarySchool(BaseBenefitsProgram):
    """All NYC children age 4-5 are eligible for kindergarten and are guaranteed placement."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] >= 4 and m["age"] <= 5:
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: Household with a 4-year-old child
        hh1 = nuclear_family()
        hh1.members[2]["age"] = 4
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert cls.__call__(hh1)  # Should pass, as the child is 4 and resides in NYC

        # Test case 2: Household with a 5-year-old child
        hh2 = nuclear_family()
        hh2.members[2]["age"] = 5
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert cls.__call__(hh2)  # Should pass, as the child is 5 and resides in NYC

        # Test case 3: Household with a 6-year-old child
        hh3 = nuclear_family()
        hh3.members[2]["age"] = 6
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert not cls.__call__(hh3)  # Should fail, as the child is older than 5


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
            # if m["college"]:
            if m["current_school_level"] == GradeLevelEnum.COLLEGE.value:
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
            if not m["is_parent"]:
                return False
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

    @classmethod
    def test_cases(cls):

        # Test 1: High school equivalency and college prep track eligibility (live in NYC and fatherhood)
        hh1 = nuclear_family()
        hh1.user()["sex"] = SexEnum.MALE.value
        hh1.user()["age"] = 25
        hh1.user()["is_parent"] = True
        hh1.user()["education_level"] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value

        assert cls.__call__(hh1)

        # Test 2: High school equivalency track eligibility (age between 18 and 30, lives in NYC, fatherhood)
        hh2 = nuclear_family()
        hh2.user()["sex"] = SexEnum.MALE.value
        for member in hh2.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["is_parent"] = True
        hh2.user()["age"] = 25
        hh2.spouse()["age"] = 25
        hh2.members[2]["age"] = 5  # child remains under 18
        assert cls.__call__(hh2)

        # Test 3: College prep track eligibility (age 18-30, high school diploma, less than 12 college credits, not enrolled in college)
        hh3 = nuclear_family()
        for member in hh3.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["is_parent"] = True
        hh3.user()["age"] = 22
        hh3.user()["sex"] = SexEnum.MALE.value
        hh3.user()[
            "high_school_equivalent"
        ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        hh3.user()["college_credits"] = 6
        hh3.user()["enrolled_in_educational_training"] = False
        assert cls.__call__(hh3)

        # Test 4: Ineligible due to age (not between 18 and 30)
        hh4 = nuclear_family()
        hh4.user()["sex"] = SexEnum.MALE.value
        for member in hh4.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["is_parent"] = True
        hh4.user()["age"] = 40
        hh4.spouse()["age"] = 40
        hh4.members[2]["age"] = 10
        assert not cls.__call__(hh4)

        # Test 5: Ineligible due to residence (does not live in NYC)
        hh5 = nuclear_family()
        hh5.user()["sex"] = SexEnum.MALE.value
        for member in hh5.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
            member["is_parent"] = True
        hh5.user()["age"] = 22
        assert not cls.__call__(hh5)

        # Test 6: Ineligible due to lack of fatherhood status
        hh6 = nuclear_family()
        hh6.user()["sex"] = SexEnum.MALE.value
        for member in hh6.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["is_parent"] = False
        hh6.user()["age"] = 22
        assert not cls.__call__(hh6)

        # Test 7: Eligible for high school equivalency track even tho in college
        hh7 = nuclear_family()
        hh7.user()["sex"] = SexEnum.MALE.value
        for member in hh7.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["is_parent"] = True
        hh7.user()["age"] = 24
        hh7.user()[
            "high_school_equivalent"
        ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        hh7.user()["college_credits"] = 6
        hh7.user()["enrolled_in_educational_training"] = True
        hh7.user()["current_school_level"] = GradeLevelEnum.COLLEGE.value
        assert cls.__call__(hh7)

        # Test 8: eligible since eligible for hs track even though ineligible for college track
        hh8 = nuclear_family()
        hh8.user()["sex"] = SexEnum.MALE.value
        for member in hh8.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["is_parent"] = True
        hh8.user()["age"] = 25
        hh8.user()[
            "high_school_equivalent"
        ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        hh8.user()["college_credits"] = 15
        hh8.user()["enrolled_in_educational_training"] = False
        assert cls.__call__(hh8)


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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Valid case - Parent with a baby < 3 months old, lives in NYCHA development, gets ACS, and lives in DHS shelter.
        hh1 = nuclear_family()
        hh1.user()["is_parent"] = True
        hh1.members[2]["age"] = 0  # Baby less than 3 months old
        hh1.members[0]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[1]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[2]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[0]["acs"] = True
        hh1.members[1]["acs"] = True
        hh1.members[2]["acs"] = True
        hh1.members[0]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh1.members[1]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh1.members[2]["housing_type"] = HousingEnum.DHS_SHELTER.value
        assert cls.__call__(hh1)

        # Test Case 2: Invalid - Parent with a baby < 3 months old but does not live in NYCHA development.
        hh2 = nuclear_family()
        hh2.user()["is_parent"] = True
        hh2.members[2]["age"] = 0  # Baby less than 3 months old
        hh2.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[2]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[0]["acs"] = True
        hh2.members[1]["acs"] = True
        hh2.members[2]["acs"] = True
        assert cls.__call__(hh2)

        # Test Case 3: Invalid - Parent with a baby < 3 months old but does not live in DHS shelter.
        hh3 = nuclear_family()
        hh3.user()["is_parent"] = True
        hh3.members[2]["age"] = 0  # Baby less than 3 months old
        hh3.members[0]["acs"] = True
        hh3.members[1]["acs"] = True
        hh3.members[2]["acs"] = True
        hh3.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh3.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh3.members[2]["housing_type"] = HousingEnum.CONDO.value
        assert cls.__call__(hh3)


class ChildrenAndYouthWithSpecialHealthCareNeeds(BaseBenefitsProgram):
    """Eligible children must:

    Be age 21 or younger

    Live in New York City

    Have been diagnosed with or may have a serious or chronic health condition, physical disability, or developmental or emotional/behavioral condition

    Need extra health care and assistance"""

    @staticmethod
    def __call__(hh):
        def is_eligible(m):
            if m["age"] > 21:
                return False
            if m["place_of_residence"] != PlaceOfResidenceEnum.NYC.value:
                return False
            if m["can_care_for_self"]:
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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Child under 21 living in NYC with chronic health condition
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["age"] = 20
        hh1.members[2]["chronic_health_condition"] = True
        hh1.members[2]["can_care_for_self"] = False
        assert cls.__call__(hh1)

        # Test Case 3: Child under 21 living in NYC with emotional/behavioral condition
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["age"] = 10
        hh3.members[2]["emotional_behavioral_condition"] = True
        hh3.members[2]["can_care_for_self"] = False
        assert cls.__call__(hh3)

        # Test Case 5: Ineligible household (child older than 21)
        hh5 = nuclear_family()
        hh5.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh5.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh5.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh5.members[2]["age"] = 22  # Not eligible due to age
        hh5.members[2]["chronic_health_condition"] = True
        assert not cls.__call__(hh5)


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

    @classmethod
    def test_cases(cls):
        hh = nuclear_family()
        hh.members[2]["age"] = 4  # Child is 4 years old
        hh.members[2][
            "emotional_behavioral_condition"
        ] = True  # Child has an emotional condition
        hh.members[2][
            "mental_health_condition"
        ] = True  # Child has a mental health condition
        assert cls.__call__(hh)  # Test should pass for the household
        hh = nuclear_family()
        hh.members[2]["age"] = 5  # Child is 5 years old
        hh.members[2][
            "mental_health_condition"
        ] = True  # Child has a mental health disorder
        assert cls.__call__(hh)  # Test should pass for the household
        hh = nuclear_family()
        hh.members[2]["age"] = 10  # Child is 10 years old
        hh.members[2][
            "emotional_behavioral_condition"
        ] = True  # Child has an emotional condition
        hh.members[2][
            "mental_health_condition"
        ] = True  # Child has a mental health condition
        assert cls.__call__(hh)  # Test should pass for the household


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

    @classmethod
    def test_cases(cls):
        # Test case 1: Eligible child with all requirements met
        hh1 = nuclear_family()
        hh1.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value

        hh1.members[2]["age"] = 10  # Under 19 years old
        hh1.members[2][
            "health_insurance"
        ] = False  # Not covered by other health insurance
        assert cls.__call__(hh1)  # Should pass since the child meets all requirements

        # Test case 2: Child over the age limit
        hh2 = nuclear_family()
        hh2.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value

        hh2.members[2]["age"] = 20  # Over 19 years old
        hh2.members[2]["health_insurance"] = False
        assert not cls.__call__(
            hh2
        )  # Should fail since the child is over the age limit

        # Test case 3: Child covered by other health insurance
        hh3 = nuclear_family()
        hh3.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value

        hh3.members[2]["age"] = 15  # Under 19 years old
        hh3.members[2]["health_insurance"] = True  # Covered by other health insurance
        assert not cls.__call__(
            hh3
        )  # Should fail since the child has other health insurance


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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Household with a single youth (age 17) living in New York City
        hh1 = nuclear_family()
        hh1.members[2]["age"] = 17  # The child is 17 years old
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2][
            "citizenship"
        ] = (
            CitizenshipEnum.UNLAWFUL_RESIDENT.value
        )  # Child is an undocumented immigrant
        assert cls.__call__(hh1)

        # Test Case 2: Household with multiple children, including a youth (age 16) and a non-youth (age 20)
        hh2 = nuclear_family()
        hh2.members[2]["age"] = 16  # The child is 16 years old
        hh2.members.append(deepcopy(hh2.members[-1]))  # Add another member
        hh2.members[3]["age"] = 20  # New member is an adult
        for member in hh2.members:
            member["place_of_residence"] = (
                PlaceOfResidenceEnum.NYC.value
            )  # All members live in NYC
            member["citizenship"] = (
                CitizenshipEnum.UNLAWFUL_RESIDENT.value
            )  # All members are undocumented immigrants
        assert cls.__call__(hh2)


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
                    if m["mental_health_condition"]:
                        return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: Child aged 10 with emotional, behavioral, or mental health challenges
        hh1 = nuclear_family()
        hh1.members[2]["age"] = 10  # Set the child's age to 10
        hh1.members[2][
            "emotional_behavioral_condition"
        ] = True  # Child has emotional challenges
        assert cls.__call__(hh1)

        # Test case 2: Child aged 5 with mental health condition
        hh2 = nuclear_family()
        hh2.members[2]["age"] = 5  # Set the child's age to 5
        hh2.members[2][
            "mental_health_condition"
        ] = True  # Child has mental health challenges
        assert cls.__call__(hh2)

        # Test case 3: Child aged 24 with behavioral challenges
        hh3 = nuclear_family()
        hh3.members[2]["age"] = 24  # Set the child's age to 24
        hh3.members[2][
            "emotional_behavioral_condition"
        ] = True  # Child has behavioral challenges
        assert cls.__call__(hh3)


class FamilyAssessmentProgram(BaseBenefitsProgram):
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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Family with a child struggling to relate to others
        hh1 = nuclear_family()
        hh1.user()["struggles_to_relate"] = True
        hh1.spouse()["struggles_to_relate"] = False
        hh1.members[2]["age"] = 10
        hh1.members[2]["struggles_to_relate"] = True
        assert cls.__call__(hh1)

        # Test Case 2: Family with a 17-year-old child and parents not struggling to relate
        hh2 = nuclear_family()
        hh2.user()["struggles_to_relate"] = False
        hh2.spouse()["struggles_to_relate"] = False
        hh2.members[2]["age"] = 17
        hh2.members[2]["struggles_to_relate"] = True
        assert cls.__call__(hh2)

        # Test Case 3: Family with two children, one struggling to relate
        hh3 = nuclear_family()
        hh3.members.append(deepcopy(hh3.members[-1]))
        hh3.members[2]["age"] = 15
        hh3.members[2]["struggles_to_relate"] = False
        hh3.members[3]["age"] = 12
        hh3.members[3]["struggles_to_relate"] = True
        assert cls.__call__(hh3)


class CornerstonePrograms(BaseBenefitsProgram):
    """Cornerstone Programs are available to NYCHA residents who are Kindergarten-age or older."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["housing_type"] == HousingEnum.NYCHA_DEVELOPMENT.value:
                if m["age"] >= 5:
                    return True

        for m in hh.members:
            if eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: Household with all members living in NYCHA development, with one child of kindergarten age
        hh1 = nuclear_family()
        hh1.members[0]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[1]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[2]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[2]["age"] = 5  # Kindergarten-age child
        assert cls.__call__(hh1)

        # Test Case 2: Household not living in NYCHA development, with a child of kindergarten age
        hh2 = nuclear_family()
        hh2.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[2]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[2]["age"] = 6  # Older child, still eligible
        assert not cls.__call__(hh2)


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

    @classmethod
    def test_cases(cls):
        # Test case 1: Household meets all criteria (NYC residence, child is 3 years old or younger)
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["age"] = 3
        assert cls.__call__(hh1), "Test case 1 failed: Household should be eligible."

        # Test case 2: Household does not meet criteria (child is 4 years old)
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["age"] = 4
        assert not cls.__call__(
            hh2
        ), "Test case 2 failed: Household should not be eligible."

        # Test case 3: Household does not meet criteria (does not live in NYC)
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[2]["age"] = 2
        assert not cls.__call__(
            hh3
        ), "Test case 3 failed: Household should not be eligible."


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
            income = hh.hh_annual_total_income()
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

    @classmethod
    def test_cases(cls):
        # Test 1: Single person household with US citizenship and income within limit
        hh1 = nuclear_family()
        hh1.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh1.members[0]["age"] = 30
        hh1.members[0]["annual_work_income"] = 50000
        hh1.members = hh1.members[:1]  # Single person household
        assert cls.__call__(hh1)

        # Test 2: Two-person household with one citizen and one lawful resident, both meeting income limit
        hh2 = nuclear_family()
        hh2.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh2.members[1]["citizenship"] = CitizenshipEnum.LAWFUL_RESIDENT.value
        hh2.members[0]["age"] = 35
        hh2.members[1]["age"] = 40
        hh2.members[0]["annual_work_income"] = 40000
        hh2.members[1]["annual_work_income"] = 40000
        assert cls.__call__(hh2)

        # Test 3: Household with three members and income exceeding limit for family size
        hh3 = nuclear_family()
        hh3.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh3.members[0]["age"] = 25
        hh3.members[1]["age"] = 25
        hh3.members[2]["age"] = 5
        hh3.members[0]["annual_work_income"] = 60000
        hh3.members[1]["annual_work_income"] = 60000
        hh3.members[2]["annual_work_income"] = 0
        assert not cls.__call__(hh3)

        # Test 4: Single person household with no legal status
        hh4 = nuclear_family()
        hh4.members[0]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh4.members[0]["age"] = 22
        hh4.members[0]["annual_work_income"] = 20000
        hh4.members = hh4.members[:1]  # Single person household
        assert not cls.__call__(hh4)

        # Test 5: Family with one child, spouse not meeting age requirement
        hh5 = nuclear_family()
        hh5.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh5.members[0]["age"] = 30
        hh5.members[1]["age"] = 16  # Spouse is a minor
        hh5.members[1]["citizenship"] = CitizenshipEnum.LAWFUL_RESIDENT.value
        hh5.members[2]["age"] = 3
        hh5.members[0]["annual_work_income"] = 30000
        hh5.members[1]["annual_work_income"] = 0
        hh5.members[2]["annual_work_income"] = 0
        assert not cls.__call__(hh5)

        # Test 6: Large family meeting all requirements
        hh6 = nuclear_family()
        hh6.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh6.members[1]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh6.members[0]["age"] = 35
        hh6.members[1]["age"] = 32
        hh6.members[2]["age"] = 8
        hh6.members.append(deepcopy(hh6.members[-1]))  # Add a 4th member
        hh6.members[-1]["age"] = 6
        hh6.members[0]["annual_work_income"] = 30000
        hh6.members[1]["annual_work_income"] = 30000
        hh6.members[2]["annual_work_income"] = 0
        hh6.members[3]["annual_work_income"] = 0
        assert cls.__call__(hh6)


class SchoolAgeAndEarlyChildhoodFamilyAndCommunityEngagementFACECenters(
    BaseBenefitsProgram
):
    """All NYC families who care for a child with disabilities are eligible to receive services for free. Trainings are open to everyone."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] != PlaceOfResidenceEnum.NYC.value:
                return False
            if m["age"] < 18:
                if m["disabled"]:
                    return True

        for m in hh.members:
            if eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: Household with one child with disabilities
        hh1 = nuclear_family()
        hh1.members[2]["age"] = 5
        hh1.members[2]["can_care_for_self"] = False
        hh1.members[2]["disabled"] = True
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert cls.__call__(hh1)

        # Test Case 2: Household without any child with disabilities
        hh2 = nuclear_family()
        hh2.members[2]["age"] = 5
        hh2.members[2]["can_care_for_self"] = True
        hh2.members[2]["disabled"] = False
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert not cls.__call__(hh2)

        # Test Case 3: Household with multiple children, one disabled
        hh3 = nuclear_family()
        hh3.members.append(deepcopy(hh3.members[-1]))
        hh3.members[2]["age"] = 5
        hh3.members[2]["can_care_for_self"] = False
        hh3.members[2]["disabled"] = True
        hh3.members[3]["age"] = 7
        hh3.members[3]["can_care_for_self"] = True
        hh3.members[3]["disabled"] = False
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[3]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert cls.__call__(hh3)

        # Test Case 4: Household not residing in NYC
        hh4 = nuclear_family()
        hh4.members[2]["age"] = 5
        hh4.members[2]["can_care_for_self"] = False
        hh4.members[2]["disabled"] = True
        hh4.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        assert not cls.__call__(hh4)


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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Single household with one child under age 4 in NYC
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["age"] = 3  # Child under age 4
        assert cls.__call__(hh1)

        # Test Case 2: Household with multiple children under age 4 in NYC
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members.append(deepcopy(hh2.members[-1]))  # Add another child
        hh2.members[2]["age"] = 3
        hh2.members[3]["age"] = 2
        assert cls.__call__(hh2)

        # Test Case 3: Household with a child under age 4, outside NYC
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[2]["age"] = 3  # Child under age 4
        assert not cls.__call__(hh3)

        # Test Case 4: Household in NYC, but child is older than 4
        hh4 = nuclear_family()
        hh4.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[2]["age"] = 5  # Child over age 4
        assert not cls.__call__(hh4)


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
                if m["months_pregnant"] > 0:
                    return True
            return False

        def threshold(hh):
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
            num_preg = sum(m["months_pregnant"] > 0 for m in hh.members)
            family_size = len(hh.members) + num_preg
            income = hh.hh_annual_total_income()
            if family_size > 8:
                return income <= thresholds[8] + 11997 * (family_size - 8)
            return income <= thresholds[family_size]

        if threshold(hh):
            for m in hh.members:
                if eligible(m):
                    return True
        return False

    @classmethod
    def test_cases(cls):
        # Test 1: A nuclear family with income below the threshold for a family of 3
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["annual_work_income"] = 20000
        hh1.members[1]["annual_work_income"] = 10000
        hh1.members[2]["annual_work_income"] = 0
        hh1.members[0]["months_pregnant"] = 5
        assert cls.__call__(hh1)

        # Test 2: A family of 4, with income slightly above the threshold
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members.append(deepcopy(hh2.members[-1]))  # Add a fourth member
        hh2.members[0]["annual_work_income"] = 35000
        hh2.members[1]["annual_work_income"] = 25000
        hh2.members[2]["annual_work_income"] = 0
        hh2.members[3]["annual_work_income"] = 0
        hh2.members[0]["months_pregnant"] = 0
        assert not cls.__call__(hh2)

        # Test 3: A single individual with income below the threshold for a family size of 1
        hh3 = nuclear_family()
        hh3.members = hh3.members[:1]  # Keep only the first member
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[0]["annual_work_income"] = 30000
        hh3.members[0]["months_pregnant"] = 1
        assert cls.__call__(hh3)

        # Test 4: A family of 5, income below the threshold, no pregnancy
        hh4 = nuclear_family()
        hh4.members.append(deepcopy(hh4.members[-1]))  # Add a fourth member
        hh4.members.append(deepcopy(hh4.members[-1]))  # Add a fifth member
        for member in hh4.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["annual_work_income"] = 15000
        hh4.members[0]["months_pregnant"] = 0
        assert not cls.__call__(hh4)


class AcceleratedStudyInAssociatePrograms(BaseBenefitsProgram):
    """To be eligible for CUNY ASAP, you should have all of the following:

    Have been accepted to a CUNY college
    Be a New York City resident or eligible for in-state tuition
    Be proficient in Math and/or English (reading and writing)
    Have no more than 15 college credits and a minimum GPA of 2.0."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["accepted_to_cuny"]:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    if m["eligible_for_instate_tuition"]:
                        if m["proficient_in_math"]:
                            if m["proficient_in_english_reading_and_writing"]:
                                if m["college_credits"] <= 15:
                                    if m["gpa"] >= 2.0:
                                        return True

        for m in hh.members:
            if eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test 1: All requirements met for the user
        hh1 = nuclear_family()
        hh1.user()["accepted_to_cuny"] = True
        hh1.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.user()["eligible_for_instate_tuition"] = True
        hh1.user()["proficient_in_math"] = True
        hh1.user()["proficient_in_english_reading_and_writing"] = True
        hh1.user()["college_credits"] = 15
        hh1.user()["gpa"] = 2.5
        assert cls.__call__(hh1)

        # Test 2: Missing CUNY acceptance
        hh2 = nuclear_family()
        hh2.user()["accepted_to_cuny"] = False
        hh2.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.user()["eligible_for_instate_tuition"] = True
        hh2.user()["proficient_in_math"] = True
        hh2.user()["proficient_in_english_reading_and_writing"] = True
        hh2.user()["college_credits"] = 15
        hh2.user()["gpa"] = 2.5
        assert not cls.__call__(hh2)

        # Test 3: Too many college credits
        hh3 = nuclear_family()
        hh3.user()["accepted_to_cuny"] = True
        hh3.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.user()["eligible_for_instate_tuition"] = True
        hh3.user()["proficient_in_math"] = True
        hh3.user()["proficient_in_english_reading_and_writing"] = True
        hh3.user()["college_credits"] = 16
        hh3.user()["gpa"] = 2.5
        assert not cls.__call__(hh3)

        # Test 4: Low GPA
        hh4 = nuclear_family()
        hh4.user()["accepted_to_cuny"] = True
        hh4.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.user()["eligible_for_instate_tuition"] = True
        hh4.user()["proficient_in_math"] = True
        hh4.user()["proficient_in_english_reading_and_writing"] = True
        hh4.user()["college_credits"] = 15
        hh4.user()["gpa"] = 1.8
        assert not cls.__call__(hh4)

        # Test 5: Spouse meets the criteria
        hh5 = nuclear_family()
        hh5.spouse()["accepted_to_cuny"] = True
        hh5.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh5.spouse()["eligible_for_instate_tuition"] = True
        hh5.spouse()["proficient_in_math"] = True
        hh5.spouse()["proficient_in_english_reading_and_writing"] = True
        hh5.spouse()["college_credits"] = 10
        hh5.spouse()["gpa"] = 3.0
        assert cls.__call__(hh5)


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

    @classmethod
    def test_cases(cls):
        # Test 1: Household meets all criteria for CUNY Start eligibility
        hh1 = nuclear_family()
        hh1.members[0][
            "high_school_equivalent"
        ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        hh1.members[1][
            "high_school_equivalent"
        ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        hh1.members[2][
            "high_school_equivalent"
        ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        hh1.members[0]["accepted_to_cuny"] = True
        hh1.members[1]["accepted_to_cuny"] = True
        hh1.members[2]["accepted_to_cuny"] = True
        hh1.members[0]["eligible_for_instate_tuition"] = True
        hh1.members[1]["eligible_for_instate_tuition"] = True
        hh1.members[2]["eligible_for_instate_tuition"] = True
        assert cls.__call__(hh1)

        # Test 2: Household where the user meets the criteria, but the spouse and child do not
        hh2 = nuclear_family()
        hh2.members[0][
            "high_school_equivalent"
        ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        hh2.members[0]["accepted_to_cuny"] = True
        hh2.members[0]["eligible_for_instate_tuition"] = True
        hh2.members[1][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh2.members[2][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        assert cls.__call__(hh2)

        # Test 3: Household where the spouse meets the criteria, but the user and child do not
        hh3 = nuclear_family()
        hh3.members[0][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh3.members[1]["high_school_equivalent"] = EducationLevelEnum.HSE_DIPLOMA.value
        hh3.members[1]["accepted_to_cuny"] = True
        hh3.members[1]["eligible_for_instate_tuition"] = True
        hh3.members[2][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        assert cls.__call__(hh3)

        # Test 4: Household with additional children, and one child meets the criteria
        hh4 = nuclear_family()
        hh4.members.append(deepcopy(hh4.members[-1]))
        hh4.members[0][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh4.members[1][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh4.members[2][
            "high_school_equivalent"
        ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        hh4.members[2]["accepted_to_cuny"] = True
        hh4.members[2]["eligible_for_instate_tuition"] = True
        hh4.members[3][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        assert cls.__call__(hh4)

        # Test 5: Household where no members meet the criteria
        hh5 = nuclear_family()
        hh5.members[0][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh5.members[1][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh5.members[2][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh5.members[0]["accepted_to_cuny"] = False
        hh5.members[1]["accepted_to_cuny"] = False
        hh5.members[2]["accepted_to_cuny"] = False
        hh5.members[0]["eligible_for_instate_tuition"] = False
        hh5.members[1]["eligible_for_instate_tuition"] = False
        hh5.members[2]["eligible_for_instate_tuition"] = False
        assert not cls.__call__(hh5)


class AdvanceEarn(BaseBenefitsProgram):
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
                    if m["current_school_level"] == GradeLevelEnum.NONE.value:
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
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: All requirements met for all household members
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value

        hh1.members[0]["age"] = 20
        hh1.members[1]["age"] = 20
        hh1.members[2]["age"] = 16

        hh1.members[0][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh1.members[1][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh1.members[2][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value

        hh1.members[0]["work_hours_per_week"] = 0
        hh1.members[1]["work_hours_per_week"] = 0
        hh1.members[2]["work_hours_per_week"] = 0

        hh1.members[0]["authorized_to_work_in_us"] = True
        hh1.members[1]["authorized_to_work_in_us"] = True
        hh1.members[2]["authorized_to_work_in_us"] = True

        assert cls.__call__(hh1)

        # Test Case 2: One member does not meet age requirement
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value

        hh2.members[0]["age"] = 15  # Does not meet the age requirement
        hh2.members[1]["age"] = 20
        hh2.members[2]["age"] = 16

        hh2.members[0][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh2.members[1][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        hh2.members[2][
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value

        hh2.members[0]["work_hours_per_week"] = 0
        hh2.members[1]["work_hours_per_week"] = 0
        hh2.members[2]["work_hours_per_week"] = 0

        hh2.members[0]["authorized_to_work_in_us"] = True
        hh2.members[1]["authorized_to_work_in_us"] = True
        hh2.members[2]["authorized_to_work_in_us"] = True

        assert cls.__call__(hh2)

        # Test Case 3: Member has a high school diploma
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value

        hh3.members[0]["age"] = 20
        hh3.members[1]["age"] = 20
        hh3.members[2]["age"] = 16

        # hh3.members[0][
        #     "high_school_equivalent"
        # ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value  # Does not meet requirement
        # hh3.members[1][
        #     "high_school_equivalent"
        # ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        # hh3.members[2][
        #     "high_school_equivalent"
        # ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value

        hh3.members[0]["current_school_level"] = GradeLevelEnum.TWELVE.value
        hh3.members[1]["current_school_level"] = GradeLevelEnum.TWELVE.value
        hh3.members[2]["current_school_level"] = GradeLevelEnum.TWELVE.value

        hh3.members[0]["work_hours_per_week"] = 0
        hh3.members[1]["work_hours_per_week"] = 0
        hh3.members[2]["work_hours_per_week"] = 0

        hh3.members[0]["authorized_to_work_in_us"] = True
        hh3.members[1]["authorized_to_work_in_us"] = True
        hh3.members[2]["authorized_to_work_in_us"] = True

        assert not cls.__call__(hh3)


class TrainEarn(BaseBenefitsProgram):
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
            if m["sex"] == SexEnum.FEMALE.value:
                return True
            return False

        def student(m):
            if m["current_school_level"] != GradeLevelEnum.NONE.value:
                return True
            if m["proficient_in_english_reading_and_writing"]:
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
            if m["months_pregnant"] > 0:
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
            income = hh.hh_annual_total_income()
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

    @classmethod
    def test_cases(cls):
        # Test 1: Valid case for NYC resident, 18 years old, not working, and household meets income limit
        hh = nuclear_family()
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh.user()["age"] = 18
        hh.user()["work_hours_per_week"] = 0
        hh.members[0]["annual_work_income"] = 0
        hh.members = hh.members[:1]
        # hh.members[1]["annual_work_income"] = 0
        # hh.members[2]["annual_work_income"] = 0
        assert cls.__call__(hh)

        # Test 2: Invalid due to not being a resident of NYC
        hh = nuclear_family()
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        assert not cls.__call__(hh)

        # Test 3: Invalid due to age (too young)
        hh = nuclear_family()
        hh.user()["age"] = 15
        assert not cls.__call__(hh)

        # Test 4: Valid for a high school dropout with low household income
        hh = nuclear_family()
        hh.user()["age"] = 20
        hh.user()["work_hours_per_week"] = 0
        hh.user()[
            "high_school_equivalent"
        ] = EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        # hh.user()["annual_work_income"] = 5000
        hh.spouse()["annual_work_income"] = 3000
        assert cls.__call__(hh)

        # Test 5: Invalid due to currently attending high school
        hh = nuclear_family()
        hh.user()["age"] = 19
        hh.user()["current_school_level"] = GradeLevelEnum.TWELVE.value
        assert not cls.__call__(hh)

        # Test 6: Valid for an English language learner with a high school diploma
        hh = nuclear_family()
        hh.user()[
            "high_school_equivalent"
        ] = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
        hh.user()["proficient_in_english_reading_and_writing"] = False
        hh.user()["annual_work_income"] = 0
        hh.user()["age"] = 18
        hh.spouse()["annual_work_income"] = 1000
        assert cls.__call__(hh)

        # Test 7: Valid for a foster youth with no income
        hh = nuclear_family()
        hh.user()["in_foster_care"] = True
        hh.user()["age"] = 18
        hh.user()["work_hours_per_week"] = 0
        hh.members[0]["annual_work_income"] = 0
        hh.members[1]["annual_work_income"] = 0
        assert cls.__call__(hh)

        # Test 8: Valid for a runaway youth with a justice system involvement
        hh = nuclear_family()
        hh.user()["is_runaway"] = True
        hh.user()["age"] = 18
        hh.user()["involved_in_justice_system"] = True
        hh.user()["work_hours_per_week"] = 0
        assert cls.__call__(hh)

        # Test 9: Invalid due to working more than allowed
        hh = nuclear_family()
        hh.user()["work_hours_per_week"] = 25
        hh.user()["annual_work_income"] = 999999
        assert not cls.__call__(hh)

        # Test 10: Valid for a household receiving SNAP
        hh = nuclear_family()
        hh.user()["receives_snap"] = True
        hh.user()["age"] = 18
        hh.user()["work_hours_per_week"] = 0
        assert cls.__call__(hh)

        # Test 11: Valid for a pregnant individual
        hh = nuclear_family()
        hh.user()["months_pregnant"] = 5
        hh.user()["age"] = 18
        hh.user()["work_hours_per_week"] = 0
        assert cls.__call__(hh)

        # Test 12: Valid for a low-income parent
        hh = nuclear_family()
        hh.user()["is_parent"] = True
        hh.user()["age"] = 18
        hh.user()["annual_work_income"] = 0
        hh.spouse()["annual_work_income"] = 5000
        assert cls.__call__(hh)

        # Test 13: Invalid due to household income above the threshold
        hh = nuclear_family()
        hh.user()["age"] = 18
        hh.user()["annual_work_income"] = 20000
        hh.spouse()["annual_work_income"] = 15000
        hh.members[2]["annual_work_income"] = 2000
        assert not cls.__call__(hh)

        # Test 14: Valid for a disabled individual
        hh = nuclear_family()
        hh.user()["age"] = 18
        hh.user()["disabled"] = True
        hh.user()["work_hours_per_week"] = 0
        assert cls.__call__(hh)

        # Test 15: Invalid due to non-registration for selective service for eligible males
        hh = nuclear_family()
        hh.user()["age"] = 20
        hh.user()["sex"] = SexEnum.MALE.value
        hh.user()["selective_service"] = False
        assert not cls.__call__(hh)

        # Test 16: Valid for a selective service-registered male
        hh = nuclear_family()
        hh.user()["age"] = 20
        hh.user()["sex"] = SexEnum.MALE.value
        hh.user()["selective_service"] = True
        hh.user()["work_hours_per_week"] = 0
        assert cls.__call__(hh)

        # Test 17: Valid for a family with household income within limits for size 3
        hh = nuclear_family()
        hh.user()["age"] = 18
        hh.members[0]["annual_work_income"] = 0
        hh.members[1]["annual_work_income"] = 0
        hh.members[2]["annual_work_income"] = 0
        assert cls.__call__(hh)

        # Test 18: Invalid for a household with income above the threshold for size 4
        hh = nuclear_family()
        hh.user()["age"] = 18
        hh.members.append(deepcopy(hh.members[-1]))
        hh.members[0]["annual_work_income"] = 0
        hh.members[1]["annual_work_income"] = 15000
        hh.members[2]["annual_work_income"] = 5000
        hh.members[3]["annual_work_income"] = 1000
        assert cls.__call__(hh)


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

    @classmethod
    def test_cases(cls):
        # Test 1: All users are NYC residents, and the third member is a minor
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["age"] = 10  # Minor child
        assert cls.__call__(hh1)

        # Test 2: At least one user has a disability and resides in NYC
        hh2 = nuclear_family()
        hh2.members[0]["disabled"] = True
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert cls.__call__(hh2)

        # Test 3: A household with a member identifying as LGBTQ+ and a minor child in NYC
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[0]["struggles_to_relate"] = True  # LGBTQ+ identifier
        hh3.members[2]["age"] = 15  # Minor child
        assert cls.__call__(hh3)


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
                        if m["work_or_volunteer_experience"]:
                            if m["gpa"] >= 3.0:
                                return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: Valid case where all conditions are met by the user
        hh1 = nuclear_family()
        hh1.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.user()["current_school_level"] = GradeLevelEnum.COLLEGE.value
        hh1.user()["age"] = 18
        hh1.user()["work_or_volunteer_experience"] = True
        hh1.user()["gpa"] = 3.5
        assert cls.__call__(hh1)

        # Test case 2: Valid case where conditions are met by the spouse
        hh2 = nuclear_family()
        hh2.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.spouse()["current_school_level"] = GradeLevelEnum.TWELVE.value
        hh2.spouse()["age"] = 20
        hh2.spouse()["work_or_volunteer_experience"] = True
        hh2.spouse()["gpa"] = 3.1
        assert cls.__call__(hh2)

        # Test case 3: Valid case where conditions are met by a child
        hh3 = nuclear_family()
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["current_school_level"] = GradeLevelEnum.TWELVE.value
        hh3.members[2]["age"] = 16
        hh3.members[2]["work_or_volunteer_experience"] = True
        hh3.members[2]["gpa"] = 3.2
        assert cls.__call__(hh3)

        # Test case 4: Invalid case where no one lives in NYC
        hh4 = nuclear_family()
        hh4.user()["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.spouse()["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.user()["current_school_level"] = GradeLevelEnum.COLLEGE.value
        hh4.user()["age"] = 18
        hh4.user()["work_or_volunteer_experience"] = True
        hh4.user()["gpa"] = 3.5
        assert not cls.__call__(hh4)

        # Test case 5: Invalid case where no one is enrolled in school
        hh5 = nuclear_family()
        hh5.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh5.user()["current_school_level"] = GradeLevelEnum.NONE.value
        hh5.user()["age"] = 18
        hh5.user()["work_or_volunteer_experience"] = True
        hh5.user()["gpa"] = 3.5
        assert not cls.__call__(hh5)

        # Test case 6: Invalid case where no one is between 16 and 22 years old
        hh6 = nuclear_family()
        hh6.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh6.user()["current_school_level"] = GradeLevelEnum.COLLEGE.value
        hh6.user()["age"] = 23
        hh6.user()["work_or_volunteer_experience"] = True
        hh6.user()["gpa"] = 3.5
        assert not cls.__call__(hh6)

        # Test case 7: Invalid case where GPA is below 3.00 for everyone
        hh7 = nuclear_family()
        hh7.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh7.user()["current_school_level"] = GradeLevelEnum.COLLEGE.value
        hh7.user()["age"] = 18
        hh7.user()["work_or_volunteer_experience"] = True
        hh7.user()["gpa"] = 2.9
        assert not cls.__call__(hh7)


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

    @classmethod
    def test_cases(cls):
        # Test 1: Valid case - youth is 14 years old and enrolled in high school
        hh1 = nuclear_family()
        hh1.members[2]["age"] = 14
        hh1.members[2]["current_school_level"] = GradeLevelEnum.NINE.value
        assert cls.__call__(hh1)

        # Test 3: Invalid case - youth is 13 years old
        hh3 = nuclear_family()
        hh3.members[2]["age"] = 13
        hh3.members[2]["current_school_level"] = GradeLevelEnum.EIGHT.value
        assert not cls.__call__(hh3)

        # Test 4: Invalid case - youth is 22 years old
        hh4 = nuclear_family()
        hh4.members[2]["age"] = 22
        hh4.members[2]["current_school_level"] = GradeLevelEnum.COLLEGE.value
        assert not cls.__call__(hh4)

        # Test 5: Invalid case - youth is 17 years old but not enrolled in school
        hh5 = nuclear_family()
        hh5.members[2]["age"] = 17
        hh5.members[2]["current_school_level"] = GradeLevelEnum.NONE.value
        assert not cls.__call__(hh5)

        # Test 6: Valid case - household with two youths, one eligible, one not
        hh6 = nuclear_family()
        hh6.members.append(deepcopy(hh6.members[-1]))
        hh6.members[2]["age"] = 15
        hh6.members[2]["current_school_level"] = GradeLevelEnum.TEN.value
        hh6.members[3]["age"] = 22
        hh6.members[3]["current_school_level"] = GradeLevelEnum.COLLEGE.value
        assert cls.__call__(hh6)


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

        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: NYCHA resident old enough to work, lives in Jobs Plus neighborhood
        hh1 = nuclear_family()
        hh1.members[0]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[1]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[2]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.user()["age"] = 25
        hh1.user()["lives_in_jobs_plus_neighborhood"] = True
        assert cls.__call__(hh1)

        # Test Case 2: NYCHA resident old enough to work, but does not live in Jobs Plus neighborhood
        hh2 = nuclear_family()
        hh2.members[0]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh2.members[1]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh2.members[2]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh2.user()["age"] = 30
        hh2.user()["lives_in_jobs_plus_neighborhood"] = False
        assert not cls.__call__(hh2)

        # Test Case 3: NYCHA resident too young to work, lives in Jobs Plus neighborhood
        hh3 = nuclear_family()
        hh3.members[0]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh3.members[1]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh3.members[2]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh3.user()["age"] = 15
        hh3.user()["lives_in_jobs_plus_neighborhood"] = True
        assert not cls.__call__(hh3)

        # Test Case 4: Mixed household with some NYCHA residents eligible and others not
        hh4 = nuclear_family()
        hh4.members[0]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh4.members[1]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh4.members[2]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh4.user()["age"] = 20
        hh4.user()["lives_in_jobs_plus_neighborhood"] = True
        hh4.spouse()["age"] = 16
        hh4.spouse()["lives_in_jobs_plus_neighborhood"] = False
        assert cls.__call__(hh4)


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

        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: NYC resident, 8th-grade student
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["current_school_level"] = GradeLevelEnum.EIGHT.value
        assert cls.__call__(hh1)

        # Test Case 2: NYC resident, first-time 9th-grade student
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["current_school_level"] = GradeLevelEnum.NINE.value
        assert cls.__call__(hh2)

        # Test Case 3: Non-NYC resident, 8th-grade student
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[2]["current_school_level"] = GradeLevelEnum.EIGHT.value
        assert not cls.__call__(hh3)


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

        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: All conditions met for all members
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["current_school_level"] = GradeLevelEnum.EIGHT.value
        assert cls.__call__(hh1)

        # Test Case 2: Only the child meets grade level criteria, but not NYC residency
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[2]["current_school_level"] = GradeLevelEnum.NINE.value
        assert not cls.__call__(hh2)

        # Test Case 3: One child meets the criteria in a larger family
        hh3 = nuclear_family()
        hh3.members.append(deepcopy(hh3.members[-1]))  # Add another child
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[3]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["current_school_level"] = GradeLevelEnum.EIGHT.value
        hh3.members[3]["current_school_level"] = GradeLevelEnum.COLLEGE.value
        assert cls.__call__(hh3)


class VeteransAffairsSupportedHousing(BaseBenefitsProgram):
    """To be eligible for HUD-VASH, you must:

    be eligible for VA health care.
    be homeless."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["housing_type"] == HousingEnum.HOMELESS.value:
                if m["va_healthcare"]:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: All members meet the requirements for HUD-VASH
        hh1 = nuclear_family()
        for member in hh1.members:
            member["va_healthcare"] = True
            member["housing_type"] = HousingEnum.HOMELESS.value
        assert cls.__call__(hh1)

        # Test case 2: One member meets the requirements for HUD-VASH
        hh2 = nuclear_family()
        hh2.members[0]["va_healthcare"] = True
        hh2.members[0]["housing_type"] = HousingEnum.HOMELESS.value
        for member in hh2.members[1:]:
            member["va_healthcare"] = False
            member["housing_type"] = HousingEnum.HOUSE_2B.value
        assert cls.__call__(hh2)

        # Test case 3: No members meet the requirements for HUD-VASH
        hh3 = nuclear_family()
        for member in hh3.members:
            member["va_healthcare"] = False
            member["housing_type"] = HousingEnum.HOUSE_2B.value
        assert not cls.__call__(hh3)


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
                        HousingEnum.NYCHA_DEVELOPMENT.value,
                        HousingEnum.SECTION_8.value,
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
                    hh_income = hh.hh_annual_total_income()
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

    @classmethod
    def test_cases(cls):
        # Test Case 1: A household with a child under 6 and a U.S. citizen, no air conditioner, meets income requirements
        hh1 = nuclear_family()
        hh1.user()["age"] = 40
        hh1.spouse()["age"] = 38
        hh1.members[2]["age"] = 5
        hh1.user()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh1.spouse()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh1.members[2]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh1.members[0]["ac"] = False
        hh1.members[1]["ac"] = False
        hh1.members[2]["ac"] = False
        hh1.user()["annual_work_income"] = 0
        hh1.spouse()["annual_work_income"] = 0
        hh1.members[2]["annual_work_income"] = 0
        assert cls.__call__(hh1)

        # Test Case 2: A household with someone age 60 or older, heat included in rent, no air conditioner
        hh2 = nuclear_family()
        hh2.user()["age"] = 60
        hh2.spouse()["age"] = 58
        hh2.members[2]["age"] = 10
        hh2.user()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh2.spouse()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh2.members[2]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh2.members[0]["ac"] = False
        hh2.members[1]["ac"] = False
        hh2.members[2]["ac"] = False
        hh2.members[0]["heat_included_in_rent"] = True
        hh2.members[1]["heat_included_in_rent"] = True
        hh2.members[2]["heat_included_in_rent"] = True
        assert cls.__call__(hh2)

        # Test Case 3: A household with a medical condition exacerbated by heat, SNAP benefits, and no air conditioner
        hh3 = nuclear_family()
        hh3.user()["age"] = 35
        hh3.spouse()["age"] = 33
        hh3.members[2]["age"] = 8
        hh3.user()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh3.spouse()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh3.members[2]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh3.members[0]["heat_exacerbated_condition"] = True
        hh3.members[1]["heat_exacerbated_condition"] = True
        hh3.members[2]["heat_exacerbated_condition"] = False
        hh3.members[0]["ac"] = False
        hh3.members[1]["ac"] = False
        hh3.members[2]["ac"] = False
        hh3.user()["receives_snap"] = True
        hh3.spouse()["receives_snap"] = True
        hh3.members[2]["receives_snap"] = True
        assert cls.__call__(hh3)

        # Test Case 4: A household with no eligible conditions, should fail
        hh4 = nuclear_family()
        hh4.user()["age"] = 30
        hh4.spouse()["age"] = 28
        hh4.members[2]["age"] = 10
        hh4.user()["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh4.spouse()["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh4.members[2]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh4.members[0]["ac"] = True
        hh4.members[1]["ac"] = True
        hh4.members[2]["ac"] = True
        assert not cls.__call__(hh4)

        # Test Case 5: A household exceeding income limits, should fail
        hh5 = nuclear_family()
        hh5.user()["age"] = 40
        hh5.spouse()["age"] = 38
        hh5.members[2]["age"] = 5
        hh5.user()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh5.spouse()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh5.members[2]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh5.members[0]["ac"] = False
        hh5.members[1]["ac"] = False
        hh5.members[2]["ac"] = False
        hh5.user()["annual_work_income"] = 50000
        hh5.spouse()["annual_work_income"] = 50000
        hh5.members[2]["annual_work_income"] = 0
        assert not cls.__call__(hh5)

        # Test Case 6: A household with a working air conditioner, should fail
        hh6 = nuclear_family()
        hh6.user()["age"] = 70
        hh6.spouse()["age"] = 68
        hh6.members[2]["age"] = 10
        hh6.user()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh6.spouse()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh6.members[2]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh6.members[0]["ac"] = True
        hh6.members[1]["ac"] = True
        hh6.members[2]["ac"] = True
        assert not cls.__call__(hh6)

        # Test Case 7: A single-member household meeting all conditions
        hh7 = nuclear_family()
        hh7.members = hh7.members[:1]
        hh7.user()["age"] = 65
        hh7.user()["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh7.user()["ac"] = False
        hh7.user()["annual_work_income"] = 30000
        hh7.user()["receives_snap"] = True
        assert cls.__call__(hh7)

        # Test Case 8: A large household meeting all conditions
        hh8 = nuclear_family()
        hh8.members.append(deepcopy(hh8.members[-1]))  # Add a 4th member
        hh8.user()["age"] = 50
        hh8.spouse()["age"] = 48
        hh8.members[2]["age"] = 3
        hh8.members[3]["age"] = 70
        for member in hh8.members:
            member["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
            member["ac"] = False
            member["annual_work_income"] = 15000
        hh8.user()["receives_snap"] = True
        assert cls.__call__(hh8)


class NYCCare(BaseBenefitsProgram):
    """To be eligible for NYC Care, you must:

    Live in New York City.
    Not qualify for any health insurance plan available in New York State."""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if not m["qualify_for_health_insurance"]:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: All members live in NYC and no member qualifies for health insurance
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["qualify_for_health_insurance"] = False
        hh1.members[1]["qualify_for_health_insurance"] = False
        hh1.members[2]["qualify_for_health_insurance"] = False
        assert cls.__call__(hh1)

        # Test Case 2: At least one member does not live in NYC
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[0]["qualify_for_health_insurance"] = False
        hh2.members[1]["qualify_for_health_insurance"] = False
        hh2.members[2]["qualify_for_health_insurance"] = False
        assert cls.__call__(hh2)

        # Test Case 3: At least one member qualifies for health insurance
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[0]["qualify_for_health_insurance"] = True
        hh3.members[1]["qualify_for_health_insurance"] = True
        hh3.members[2]["qualify_for_health_insurance"] = True
        assert not cls.__call__(hh3)

        # Test Case 4: Household with additional members, all meeting the requirements
        hh4 = nuclear_family()
        hh4.members.append(deepcopy(hh4.members[-1]))  # Add another child
        hh4.members.append(deepcopy(hh4.members[-1]))  # Add another child
        for member in hh4.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["qualify_for_health_insurance"] = False
        assert cls.__call__(hh4)


class ActionNYC(BaseBenefitsProgram):
    """ActionNYC is for all New Yorkers, regardless of immigration status. Your documented status does not affect your eligbility to use ActionNYC."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: Basic household eligibility with mixed immigration status
        hh1 = nuclear_family()
        hh1.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh1.members[1]["citizenship"] = CitizenshipEnum.LAWFUL_RESIDENT.value
        hh1.members[2]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value

        assert cls.__call__(
            hh1
        ), "Household should qualify regardless of immigration status."

        # Test Case 2: Household with only undocumented members
        hh2 = nuclear_family()
        hh2.members[0]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh2.members[1]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh2.members[2]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value

        assert cls.__call__(
            hh2
        ), "Household with only undocumented members should qualify."

        # Test Case 3: Extended household with mixed statuses and a new child
        hh3 = nuclear_family()
        hh3.members.append(deepcopy(hh3.members[-1]))  # Add a second child
        hh3.members[3]["age"] = 5

        hh3.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh3.members[1]["citizenship"] = CitizenshipEnum.LAWFUL_RESIDENT.value
        hh3.members[2]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh3.members[3]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value

        assert cls.__call__(
            hh3
        ), "Extended household with mixed statuses should qualify."


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
            income = hh.hh_annual_total_income()
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
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["age"] >= 18 and m["age"] <= 64:
                    return True
            return False

        for m in hh.members:
            if eligible(m):
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test 1: Household is eligible if any member is a New Yorker, regardless of immigration status
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh1.members[1]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh1.members[2]["citizenship"] = CitizenshipEnum.LAWFUL_RESIDENT.value
        assert cls.__call__(hh1)

        # Test 2: Household is eligible if a child is a New Yorker and other members have varying citizenship statuses
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[0]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh2.members[1]["citizenship"] = CitizenshipEnum.LAWFUL_RESIDENT.value
        hh2.members[2]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        assert not cls.__call__(hh2)

        # Test 3: Household is not eligible if no members are residents of NYC
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh3.members[0]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh3.members[1]["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh3.members[2]["citizenship"] = CitizenshipEnum.LAWFUL_RESIDENT.value
        assert not cls.__call__(hh3)


class WeSpeakNYC(BaseBenefitsProgram):
    """Anyone is eligible to sign up for an online class. Classes and materials are created for intermediate English language learners ages 16 and above."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] >= 16:
                if m["proficient_in_english_reading_and_writing"]:
                    return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: All members meet the criteria
        hh1 = nuclear_family()
        hh1.members[0]["age"] = 30  # User
        hh1.members[1]["age"] = 35  # Spouse
        hh1.members[2]["age"] = 16  # Child

        hh1.members[0]["proficient_in_english_reading_and_writing"] = True
        hh1.members[1]["proficient_in_english_reading_and_writing"] = True
        hh1.members[2]["proficient_in_english_reading_and_writing"] = True

        assert cls.__call__(hh1)

        # Test Case 2: Only one member meets the criteria
        hh2 = nuclear_family()
        hh2.members[0]["age"] = 18  # User
        hh2.members[1]["age"] = 40  # Spouse
        hh2.members[2]["age"] = 10  # Child

        hh2.members[0]["proficient_in_english_reading_and_writing"] = True
        hh2.members[1]["proficient_in_english_reading_and_writing"] = False
        hh2.members[2]["proficient_in_english_reading_and_writing"] = False

        assert cls.__call__(hh2)

        # Test Case 3: None of the members meet the age requirement
        hh3 = nuclear_family()
        hh3.members[0]["age"] = 15  # User
        hh3.members[1]["age"] = 14  # Spouse
        hh3.members[2]["age"] = 12  # Child

        hh3.members[0]["proficient_in_english_reading_and_writing"] = True
        hh3.members[1]["proficient_in_english_reading_and_writing"] = True
        hh3.members[2]["proficient_in_english_reading_and_writing"] = True

        assert not cls.__call__(hh3)


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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Household in NYC and at risk of homelessness
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["at_risk_of_homelessness"] = True
        hh1.members[1]["at_risk_of_homelessness"] = True
        hh1.members[2]["at_risk_of_homelessness"] = True
        assert cls.__call__(hh1)

        # Test Case 2: Household not in NYC but at risk of homelessness
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[0]["at_risk_of_homelessness"] = True
        hh2.members[1]["at_risk_of_homelessness"] = True
        hh2.members[2]["at_risk_of_homelessness"] = True
        assert not cls.__call__(hh2)

        # Test Case 3: Household in NYC but not at risk of homelessness
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[0]["at_risk_of_homelessness"] = False
        hh3.members[1]["at_risk_of_homelessness"] = False
        hh3.members[2]["at_risk_of_homelessness"] = False
        assert not cls.__call__(hh3)


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
    NY State Emergency COVID-19 and Paid Sick and Family Leave covers people under mandatory quarantine or isolation orders or whose minor dependent is.
    """

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["place_of_residence"] != PlaceOfResidenceEnum.NYC.value:
                return False
            if m["federal_work_study"]:
                return False
            if m["scholarship"]:
                return False
            if m["government_job"]:
                return False
            if m["is_therapist"]:
                return False
            if m["contractor"]:
                return False
            if m["wep"]:
                return False
            if m["collective_bargaining"]:
                return False
            return True

        for m in hh.members:
            if eligible(m):
                if m["annual_work_income"] > 0:
                    return True
        return False

    @classmethod
    def test_cases(cls):
        # Test 1: A user in NYC, not part of any exclusion, should pass.
        hh1 = nuclear_family()
        for member in hh1.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["student_in_federal_work_study"] = False
            member["compensated_by_scholarship"] = False
            member["government_employee"] = False
            member["is_therapist"] = False
            member["contractor"] = False
            member["wep"] = False
            member["collective_bargaining"] = False
            member["annual_work_income"] = 100000
        assert cls.__call__(hh1)

        # Test 2: A user living outside NYC should fail.
        hh2 = nuclear_family()
        for member in hh2.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
            member["annual_work_income"] = 100000

        assert not cls.__call__(hh2)

        # Test 3: A user in a federal work study program should fail.
        hh3 = nuclear_family()
        # hh3.user()["student_in_federal_work_study"] = True
        for member in hh3.members:
            member["federal_work_study"] = True
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["annual_work_income"] = 100000
        assert not cls.__call__(hh3)

        # Test 4: A user compensated by scholarship should fail.
        hh4 = nuclear_family()
        for member in hh4.members:
            member["scholarship"] = True
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["annual_work_income"] = 100000
        assert not cls.__call__(hh4)

        # Test 5: A government employee should fail.
        hh5 = nuclear_family()
        for member in hh5.members:
            member["government_job"] = True
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["annual_work_income"] = 100000
        assert not cls.__call__(hh5)

        # Test 6: A licensed therapist meeting pay threshold should fail.
        hh6 = nuclear_family()
        for member in hh6.members:
            member["is_therapist"] = True
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["annual_work_income"] = 100000
        assert not cls.__call__(hh6)

        # Test 7: A user participating in WEP should fail.
        hh7 = nuclear_family()
        for member in hh7.members:
            member["wep"] = True
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["annual_work_income"] = 100000
        assert not cls.__call__(hh7)

        # Test 8: A user with a collective bargaining agreement waiving the law should fail.
        hh8 = nuclear_family()
        for member in hh8.members:
            member["collective_bargaining"] = True
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["annual_work_income"] = 100000
        assert not cls.__call__(hh8)


class STEMMattersNYC(BaseBenefitsProgram):
    """Available to students entering grades 1 through 12 in NYC public and charter schools in September 2024."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["current_school_level"] in [
                GradeLevelEnum.ONE.value,
                GradeLevelEnum.TWO.value,
                GradeLevelEnum.THREE.value,
                GradeLevelEnum.FOUR.value,
                GradeLevelEnum.FIVE.value,
                GradeLevelEnum.SIX.value,
                GradeLevelEnum.SEVEN.value,
                GradeLevelEnum.EIGHT.value,
                GradeLevelEnum.NINE.value,
                GradeLevelEnum.TEN.value,
                GradeLevelEnum.ELEVEN.value,
                GradeLevelEnum.TWELVE.value,
            ]:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: Household with all members in NYC public/charter school and enrolled in grades 1 through 12
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["current_school_level"] = GradeLevelEnum.FIVE.value  # Grade 5
        hh1.members[0][
            "current_school_level"
        ] = GradeLevelEnum.NONE.value  # User not in school
        hh1.members[1][
            "current_school_level"
        ] = GradeLevelEnum.NONE.value  # Spouse not in school
        assert cls.__call__(hh1)

        # Test Case 2: Household with a child in a charter school, grade 12, and living in NYC
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["current_school_level"] = GradeLevelEnum.TWELVE.value  # Grade 12
        hh2.members[2][
            "enrolled_in_educational_training"
        ] = True  # Confirmed enrollment
        assert cls.__call__(hh2)

        # Test Case 3: Household with a child in grade 1 and receiving public education in NYC
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["current_school_level"] = GradeLevelEnum.ONE.value  # Grade 1
        hh3.members[2]["receives_cash_assistance"] = False  # No additional assistance
        assert cls.__call__(hh3)


class COVIDnineteenFuneralAssistance(BaseBenefitsProgram):
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

    @classmethod
    def test_cases(cls):
        # Test Case 1: All members are U.S. citizens, incurred COVID-19 funeral expenses, and the death certificate attributes the death to COVID-19.
        hh1 = nuclear_family()
        for member in hh1.members:
            member["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
            member["covid_funeral_expenses"] = True
        assert cls.__call__(hh1)

        # Test Case 3: The user incurred COVID-19 funeral expenses and the death certificate attributes the death to COVID-19, but the user is not a qualified noncitizen.
        hh3 = nuclear_family()
        hh3.user()["citizenship"] = CitizenshipEnum.UNLAWFUL_RESIDENT.value
        hh3.user()["covid_funeral_expenses"] = True
        for member in hh3.members[1:]:
            member["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
            member["covid_funeral_expenses"] = False
        assert not cls.__call__(hh3)

        # Test Case 4: A household with multiple members where only one child incurred COVID-19 funeral expenses, is a U.S. citizen, and the death certificate attributes the death to COVID-19.
        hh4 = nuclear_family()
        hh4.members.append(deepcopy(hh4.members[-1]))  # Add another child
        hh4.members[-1]["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        hh4.members[-1]["covid_funeral_expenses"] = True
        for member in hh4.members[:-1]:
            member["citizenship"] = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
            member["covid_funeral_expenses"] = False
        assert cls.__call__(hh4)


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
        income = hh.hh_annual_total_income()
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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Household receives SNAP
        hh1 = nuclear_family()
        hh1.members[0]["receives_snap"] = True
        hh1.members[1]["receives_snap"] = True
        hh1.members[2]["receives_snap"] = True
        assert cls.__call__(hh1)

        # Test Case 2: Household receives Medicaid
        hh2 = nuclear_family()
        hh2.members[0]["receives_medicaid"] = True
        hh2.members[1]["receives_medicaid"] = True
        hh2.members[2]["receives_medicaid"] = True
        assert cls.__call__(hh2)

        # Test Case 3: Household receives Veterans Pension and Survivors Benefit
        hh3 = nuclear_family()
        hh3.members[0]["receives_vpsb"] = True
        hh3.members[1]["receives_vpsb"] = True
        hh3.members[2]["receives_vpsb"] = True
        assert cls.__call__(hh3)

        # Test Case 4: Household income is equal to or less than requirements for 3 members
        hh4 = nuclear_family()
        hh4.members[0]["annual_work_income"] = 10000
        hh4.members[1]["annual_work_income"] = 5000
        hh4.members[2]["annual_work_income"] = 2000
        assert cls.__call__(hh4)

        # Test Case 5: Household participates in Federal Public Housing Assistance (FPHA)
        hh5 = nuclear_family()
        hh5.members[0]["receives_fpha"] = True
        hh5.members[1]["receives_fpha"] = True
        hh5.members[2]["receives_fpha"] = True
        assert cls.__call__(hh5)

        # Test Case 6: Household with more than 8 members and meets income requirement
        hh6 = nuclear_family()
        for _ in range(6):  # Add 6 more children to make household size 9
            hh6.members.append(deepcopy(hh6.members[-1]))
        for member in hh6.members:
            member["annual_work_income"] = (
                3000  # Total income: $27,000 (less than $71,172 for 9 members)
            )
        assert cls.__call__(hh6)


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
            income = hh.hh_annual_total_income() / 12
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
                if m["work_hours_per_week"] >= 10:
                    return True
                if m["enrolled_in_educational_training"]:
                    return True
                if m["enrolled_in_vocational_training"]:
                    return True
                if m["days_looking_for_work"] > 0:
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

    @classmethod
    def test_cases(cls):
        # Test 1: Household qualifies due to receiving Cash Assistance
        hh1 = nuclear_family()
        hh1.members[0]["receives_cash_assistance"] = True
        hh1.members[1]["receives_cash_assistance"] = True
        hh1.members[2]["receives_cash_assistance"] = True
        assert cls.__call__(hh1)

        # Test 2: Household qualifies due to being homeless
        hh2 = nuclear_family()
        hh2.members[0]["housing_type"] = HousingEnum.HOMELESS.value
        hh2.members[1]["housing_type"] = HousingEnum.HOMELESS.value
        hh2.members[2]["housing_type"] = HousingEnum.HOMELESS.value
        assert cls.__call__(hh2)

        # Test 3: Household qualifies due to low income (family size 3)
        hh3 = nuclear_family()
        hh3.members[0]["annual_work_income"] = 20000
        hh3.members[1]["annual_work_income"] = 30000
        hh3.members[2]["annual_work_income"] = 0
        assert cls.__call__(hh3)

        # Test 4: Household qualifies due to work hours (10+ hours per week)
        hh4 = nuclear_family()
        hh4.members[0]["work_hours_per_week"] = 10
        hh4.members[1]["work_hours_per_week"] = 15
        hh4.members[2]["work_hours_per_week"] = 0
        assert cls.__call__(hh4)

        # Test 5: Household qualifies due to educational training
        hh5 = nuclear_family()
        hh5.members[0]["enrolled_in_educational_training"] = True
        hh5.members[1]["enrolled_in_educational_training"] = True
        hh5.members[2]["enrolled_in_educational_training"] = False
        assert cls.__call__(hh5)

        # Test 6: Household qualifies due to living in temporary housing
        hh6 = nuclear_family()
        hh6.members[0]["housing_type"] = HousingEnum.TEMPORARY_HOUSING.value
        hh6.members[1]["housing_type"] = HousingEnum.TEMPORARY_HOUSING.value
        hh6.members[2]["housing_type"] = HousingEnum.TEMPORARY_HOUSING.value
        assert cls.__call__(hh6)

        # Test 7: Household qualifies due to attending domestic violence services
        hh7 = nuclear_family()
        hh7.members[0]["attending_service_for_domestic_violence"] = True
        hh7.members[1]["attending_service_for_domestic_violence"] = False
        hh7.members[2]["attending_service_for_domestic_violence"] = False
        assert cls.__call__(hh7)

        # Test 8: Household qualifies due to receiving treatment for substance abuse
        hh8 = nuclear_family()
        hh8.members[0]["receiving_treatment_for_substance_abuse"] = True
        hh8.members[1]["receiving_treatment_for_substance_abuse"] = False
        hh8.members[2]["receiving_treatment_for_substance_abuse"] = False
        assert cls.__call__(hh8)


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

    @classmethod
    def test_cases(cls):
        # Test Case 1: All members live in NYC and user is at least 18
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["age"] = 18
        hh1.members[1]["age"] = 40
        hh1.members[2]["age"] = 10
        assert cls.__call__(hh1)

        # Test Case 2: At least one member works in NYC and is 18 or older
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[0]["works_outside_home"] = True
        hh2.members[0]["work_hours_per_week"] = 20
        hh2.members[0]["age"] = 20
        assert not cls.__call__(hh2)

        # Test Case 3: User and spouse live in NYC but child does not meet the age requirement
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[0]["age"] = 25
        hh3.members[1]["age"] = 40
        hh3.members[2]["age"] = 10
        assert cls.__call__(hh3)

        hh4 = nuclear_family()
        hh4.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.members[1]["works_outside_home"] = True
        hh4.members[1]["work_hours_per_week"] = 30
        hh4.members[1]["age"] = 30
        assert not cls.__call__(hh4)


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
                if m["age"] < 19 and m["enrolled_in_educational_training"]:
                    return True
                if m["age"] < 19 and m["enrolled_in_vocational_training"]:
                    return True
                if m["months_pregnant"] > 0:
                    return True
            return False

        def r2(hh):
            for m in hh.members:
                if m["receives_cash_assistance"]:
                    return True
            return False

        def r3(hh):
            for m in hh.members:
                if m["housing_type"] == HousingEnum.HRA_SHELTER.value:
                    return True
                if m["housing_type"] == HousingEnum.DHS_SHELTER.value:
                    if m["eligible_for_hra_shelter"]:
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

    @classmethod
    def test_cases(cls):

        # Test Case 1: Family with a child under 18
        hh1 = nuclear_family()
        hh1.members[2]["age"] = 17
        hh1.members[0]["receives_cash_assistance"] = True
        hh1.members[0]["housing_type"] = HousingEnum.HRA_SHELTER.value
        hh1.members[1]["housing_type"] = HousingEnum.HRA_SHELTER.value
        hh1.members[2]["housing_type"] = HousingEnum.HRA_SHELTER.value
        assert cls.__call__(hh1)

        # Test Case 2: Family with a 19-year-old enrolled in vocational training
        hh2 = nuclear_family()
        hh2.members[2]["age"] = 18
        hh2.members[2]["current_school_level"] = GradeLevelEnum.COLLEGE.value
        hh2.members[2]["enrolled_in_vocational_training"] = True
        hh2.members[0]["receives_cash_assistance"] = True
        hh2.members[0]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh2.members[1]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh2.members[2]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh2.members[0]["eligible_for_hra_shelter"] = True
        assert cls.__call__(hh2)

        # Test Case 3: Family with a pregnant member
        hh3 = nuclear_family()
        hh3.members = hh3.members[:2]
        hh3.members[0]["months_pregnant"] = 5
        hh3.members[0]["housing_type"] = HousingEnum.RENT_CONTROLLED_APARTMENT.value
        hh3.members[1]["housing_type"] = HousingEnum.RENT_CONTROLLED_APARTMENT.value
        # hh3.members[2]["housing_type"] = HousingEnum.RENT_CONTROLLED_APARTMENT.value
        hh3.members[0]["currently_being_evicted"] = True
        hh3.members[1]["currently_being_evicted"] = True
        hh3.members[0]["receives_cash_assistance"] = True
        hh3.members[1]["receives_cash_assistance"] = True
        assert cls.__call__(hh3)

        # Test Case 4: Family evicted within the last 12 months
        hh4 = nuclear_family()
        hh4.members[2]["age"] = 10
        hh4.members[0]["evicted_months_ago"] = 8
        hh4.members[0]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh4.members[1]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh4.members[2]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh4.members[0]["receives_cash_assistance"] = True
        hh4.members[1]["receives_cash_assistance"] = True
        assert cls.__call__(hh4)

        # Test Case 5: Family meeting multiple conditions (child under 18 and in shelter)
        hh5 = nuclear_family()
        hh5.members[2]["age"] = 15
        hh5.members[0]["receives_cash_assistance"] = True
        hh5.members[0]["housing_type"] = HousingEnum.HRA_SHELTER.value
        hh5.members[1]["housing_type"] = HousingEnum.HRA_SHELTER.value
        hh5.members[2]["housing_type"] = HousingEnum.HRA_SHELTER.value
        hh5.members[0]["receives_cash_assistance"] = True
        hh5.members[1]["receives_cash_assistance"] = True
        assert cls.__call__(hh5)

        # Test Case 6: Family with eviction threat and child in vocational training
        hh6 = nuclear_family()
        hh6.members[2]["age"] = 18
        hh6.members[2]["current_school_level"] = GradeLevelEnum.TWELVE.value
        hh6.members[2]["enrolled_in_vocational_training"] = True
        hh6.members[0]["receives_cash_assistance"] = True
        hh6.members[0]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh6.members[1]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh6.members[2]["housing_type"] = HousingEnum.DHS_SHELTER.value
        hh6.members[0]["currently_being_evicted"] = True
        hh6.members[1]["currently_being_evicted"] = True
        hh6.members[0]["receives_cash_assistance"] = True
        hh6.members[1]["receives_cash_assistance"] = True

        assert cls.__call__(hh6)


class NYSPaidFamilyLeave(BaseBenefitsProgram):
    """You can take Paid Family Leave if you:

    Are a resident of New York State.
    Work for a private employer in New York State or for a public employer who has opted in.
    Meet the time-worked requirements before taking Paid Family Leave:
    Full-time employees who regularly work 20 or more hours/week can take PFL after working 26 consecutive weeks.
    Part-time employees who regularly work less than 20 hours/week can take PFL after working 175 days. These days don't need to be consecutive.
    """

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                if m["employer_opt_in"]:
                    if m["work_hours_per_week"] >= 20:
                        if m["consecutive_work_weeks"] >= 26:
                            return True
                    else:
                        if m["nonconsecutive_work_days"] >= 175:
                            return True
        return False

    @classmethod
    def test_cases(cls):
        # Test 1: Full-time employee meets the requirements
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["work_hours_per_week"] = 40
        hh1.members[1]["work_hours_per_week"] = 40
        hh1.members[0]["consecutive_work_weeks"] = 30
        hh1.members[1]["consecutive_work_weeks"] = 30
        hh1.members[0]["employer_opt_in"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["employer_opt_in"] = PlaceOfResidenceEnum.NYC.value
        assert cls.__call__(hh1)

        # Test 2: Part-time employee meets the requirements
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[0]["work_hours_per_week"] = 10
        hh2.members[1]["work_hours_per_week"] = 15
        hh2.members[0]["nonconsecutive_work_days"] = 180
        hh2.members[1]["nonconsecutive_work_days"] = 180
        hh2.members[0]["employer_opt_in"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["employer_opt_in"] = PlaceOfResidenceEnum.NYC.value
        assert cls.__call__(hh2)

        # Test 3: Does not meet time-worked requirements (full-time)
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[0]["work_hours_per_week"] = 40
        hh3.members[1]["work_hours_per_week"] = 40
        hh3.members[0]["consecutive_work_weeks"] = 20
        hh3.members[1]["consecutive_work_weeks"] = 20
        hh3.members[0]["employer_opt_in"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["employer_opt_in"] = PlaceOfResidenceEnum.NYC.value
        assert not cls.__call__(hh3)

        # Test 4: Does not meet time-worked requirements (part-time)
        hh4 = nuclear_family()
        hh4.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[0]["work_hours_per_week"] = 10
        hh4.members[1]["work_hours_per_week"] = 15
        hh4.members[0]["nonconsecutive_work_days"] = 150
        hh4.members[1]["nonconsecutive_work_days"] = 150
        hh4.members[0]["employer_opt_in"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[1]["employer_opt_in"] = PlaceOfResidenceEnum.NYC.value
        assert not cls.__call__(hh4)

    # Test 5: Does not meet time-worked requirements (part-time)


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
                if not m["can_care_for_self"]:
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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Valid household with an eligible adult meeting all conditions
        hh1 = nuclear_family()
        hh1.user()["age"] = 30
        hh1.user()["can_care_for_self"] = False
        hh1.user()["developmental_condition"] = True
        hh1.user()["developmental_mental_day_treatment"] = True
        assert cls.__call__(hh1)

        # Test Case 2: Household with an adult with mental illness but not attending treatment
        hh2 = nuclear_family()
        hh2.user()["age"] = 40
        hh2.user()["can_care_for_self"] = False
        hh2.user()["mental_health_condition"] = True
        hh2.user()["developmental_mental_day_treatment"] = False
        assert not cls.__call__(hh2)

        # Test Case 3: Household with an individual with a history of substance abuse less than 5 years clean
        hh3 = nuclear_family()
        hh3.user()["age"] = 25
        hh3.user()["years_sober"] = 3
        hh3.user()["can_care_for_self"] = False
        hh3.user()["developmental_mental_day_treatment"] = True
        assert not cls.__call__(hh3)

        # Test Case 4: Household where an individual is wheelchair-bound
        hh4 = nuclear_family()
        hh4.user()["age"] = 30
        hh4.user()["can_care_for_self"] = False
        hh4.user()["wheelchair"] = True
        assert not cls.__call__(hh4)

        # Test Case 5: Household with a bedridden individual
        hh5 = nuclear_family()
        hh5.user()["age"] = 30
        hh5.user()["can_care_for_self"] = False
        hh5.user()["bedridden"] = True
        assert not cls.__call__(hh5)

        # Test Case 6: Household with an individual with arson history
        hh6 = nuclear_family()
        hh6.user()["age"] = 35
        hh6.user()["can_care_for_self"] = False
        hh6.user()["arson"] = True
        assert not cls.__call__(hh6)

        # Test Case 7: Household with an individual with verbal abuse history
        hh7 = nuclear_family()
        hh7.user()["age"] = 50
        hh7.user()["can_care_for_self"] = False
        hh7.user()["verbal_abuse"] = True
        assert not cls.__call__(hh7)

        # Test Case 8: Household where all users are eligible
        hh8 = nuclear_family()
        hh8.user()["age"] = 45
        hh8.spouse()["age"] = 40
        hh8.user()["can_care_for_self"] = False
        hh8.spouse()["can_care_for_self"] = False
        hh8.user()["mental_health_condition"] = True
        hh8.spouse()["developmental_condition"] = True
        hh8.user()["developmental_mental_day_treatment"] = True
        hh8.spouse()["developmental_mental_day_treatment"] = True
        assert cls.__call__(hh8)


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
    This requirement is waived for U. S. military veterans with a DD-214 that verifies honorable service.
    """

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
                8: 164200,
            }
            income = hh.hh_annual_total_income()
            family_size = len(hh.members)
            if family_size > 8:
                return income <= thresholds[8] + 9950 * (family_size - 8)
            return income <= thresholds[family_size]

        def r1(m):
            if m["first_time_home_buyer"]:
                return True
            if m["honorable_service"]:
                return True
            return False

        if income(hh):
            for m in hh.members:
                if r1(m):
                    return True
        return False

    @classmethod
    def test_cases(cls):

        # Test Case 1: Single-person household, income within the limit
        hh1 = nuclear_family()
        hh1.members = hh1.members[:1]  # Only the user remains
        hh1.user()["annual_work_income"] = 87000
        hh1.user()["first_time_home_buyer"] = True
        assert cls.__call__(hh1)

        # Test Case 2: Two-person household, income within the limit
        hh2 = nuclear_family()
        hh2.members[0]["annual_work_income"] = 60000
        hh2.members[1]["annual_work_income"] = 35000
        hh2.members[0]["first_time_home_buyer"] = True
        hh2.members[1]["first_time_home_buyer"] = True
        assert cls.__call__(hh2)

        # Test Case 3: Three-person household, income over the limit
        hh3 = nuclear_family()
        # hh3.members.append(deepcopy(hh3.members[-1]))  # Add a third member
        hh3.members[0]["annual_work_income"] = 40000
        hh3.members[1]["annual_work_income"] = 40000
        hh3.members[2]["annual_work_income"] = 40000
        hh3.members[0]["first_time_home_buyer"] = True
        hh3.members[1]["first_time_home_buyer"] = True
        hh3.members[2]["first_time_home_buyer"] = True
        assert not cls.__call__(hh3)

        # Test Case 4: Household with a military veteran, income within the limit
        hh4 = nuclear_family()
        hh4.members[0]["annual_work_income"] = 85000
        hh4.members[0]["conflict_veteran"] = True
        hh4.members[0]["first_time_home_buyer"] = False
        hh4.members[0]["honorable_service"] = True
        assert cls.__call__(hh4)

        # Test Case 5: Four-person household, income at the upper limit
        hh5 = nuclear_family()
        hh5.members.append(deepcopy(hh5.members[-1]))  # Add a fourth member
        hh5.members[0]["annual_work_income"] = 30000
        hh5.members[1]["annual_work_income"] = 40000
        hh5.members[2]["annual_work_income"] = 20000
        hh5.members[3]["annual_work_income"] = 2400
        for member in hh5.members:
            member["first_time_home_buyer"] = True
        assert cls.__call__(hh5)

        # Test Case 6: Veteran household exceeding income limit
        hh6 = nuclear_family()
        hh6.members[0]["annual_work_income"] = 125000
        hh6.members[0]["conflict_veteran"] = True
        hh6.members[0]["honorable_service"] = True
        hh6.members[0]["first_time_home_buyer"] = False
        assert not cls.__call__(hh6)

        # Test Case 7: Eight-person household, income below limit
        hh7 = nuclear_family()
        for _ in range(5):  # Add five more members
            hh7.members.append(deepcopy(hh7.members[-1]))
        for member in hh7.members:
            member["annual_work_income"] = 20000
            member["first_time_home_buyer"] = True
        assert cls.__call__(hh7)

        # Test Case 8: Single-person household, income over the limit
        hh8 = nuclear_family()
        hh8.members = hh8.members[:1]  # Only the user remains
        hh8.user()["annual_work_income"] = 90000
        hh8.user()["first_time_home_buyer"] = True
        assert not cls.__call__(hh8)


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
        income = hh.hh_annual_total_income()
        thresholds = {
            1: 86960,
            2: 99440,
            3: 111840,
            4: 124240,
            5: 134160,
            6: 144080,
            7: 154080,
            8: 164000,
        }
        if family_size > 8:
            return income <= thresholds[8] + 9920 * (family_size - 8)
        return income <= thresholds[family_size]

    @classmethod
    def test_cases(cls):
        # Test Case 1: Household of 3 with income within limits for Federally Assisted Rental
        hh1 = nuclear_family()
        hh1.members[0]["annual_work_income"] = 40_000
        hh1.members[1]["annual_work_income"] = 30_000
        hh1.members[2]["annual_work_income"] = 0
        total_income = sum(member["annual_work_income"] for member in hh1.members)
        assert (
            total_income <= 111_840
        )  # Limit for a household of 3 for Federally Assisted Rental
        assert cls.__call__(hh1)

        # Test Case 2: Household of 2 exceeding income limits for Federally Assisted Cooperative
        hh2 = nuclear_family()
        hh2.members.pop()  # Remove the child to make the household size 2
        hh2.members[0]["annual_work_income"] = 80_000
        hh2.members[1]["annual_work_income"] = 90_000
        total_income = sum(member["annual_work_income"] for member in hh2.members)
        assert (
            total_income > 155_375
        )  # Limit for a household of 2 for Federally Assisted Cooperative
        assert not cls.__call__(hh2)

        # Test Case 3: Household of 4 meeting limits for Non-Federally Assisted
        hh3 = nuclear_family()
        hh3.members.append(
            deepcopy(hh3.members[-1])
        )  # Add another child to make household size 4
        hh3.members[0]["annual_work_income"] = 50_000
        hh3.members[1]["annual_work_income"] = 60_000
        hh3.members[2]["annual_work_income"] = 0
        hh3.members[3]["annual_work_income"] = 0
        total_income = sum(member["annual_work_income"] for member in hh3.members)
        assert (
            total_income <= 194_125
        )  # Limit for a household of 4 for Non-Federally Assisted
        assert cls.__call__(hh3)


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

    @classmethod
    def test_cases(cls):
        # Test case 1: All members of the household are renters and eligible regardless of immigration status
        hh1 = nuclear_family()
        assert not cls.__call__(hh1)
        hh2 = nuclear_family()
        hh2.members[0]["monthly_rent_spending"] = 1000
        assert cls.__call__(hh2)


class TextTwoWork(BaseBenefitsProgram):
    """TXT-2-WORK is available for:
    NYC residents
    HRA clients who receive assistance including temporary cash, SNAP, or housing assistance
    """

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

    @classmethod
    def test_cases(cls):
        # Test Case 1: Household resides in NYC
        hh1 = nuclear_family()
        for member in hh1.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert cls.__call__(hh1), "Household in NYC should pass eligibility"

        # Test Case 2: Household receives temporary cash assistance
        hh2 = nuclear_family()
        for member in hh2.members:
            member["receives_temporary_assistance"] = True
        assert cls.__call__(
            hh2
        ), "Household receiving temporary cash assistance should pass eligibility"

        # Test Case 3: Household receives SNAP
        hh3 = nuclear_family()
        for member in hh3.members:
            member["receives_snap"] = True
        assert cls.__call__(hh3), "Household receiving SNAP should pass eligibility"

        # Test Case 4: Household receives housing assistance
        hh4 = nuclear_family()
        for member in hh4.members:
            member["receives_fpha"] = True
        assert cls.__call__(
            hh4
        ), "Household receiving housing assistance should pass eligibility"


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
            income = hh.hh_annual_total_income()
            thresholds = {
                1: 60240,
                2: 81760,
                3: 103280,
                4: 124800,
                5: 146320,
                6: 167840,
                7: 189360,
                8: 210880,
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

    @classmethod
    def test_cases(cls):
        # Case 1: Single person, eligible
        hh1 = nuclear_family()
        hh1.user()["age"] = 55
        hh1.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.user()["annual_work_income"] = 60000
        hh1.user()["annual_investment_income"] = 240
        hh1.spouse()["age"] = 40  # Irrelevant spouse data
        hh1.members = hh1.members[0:1]  # Remove other members
        assert cls.__call__(hh1)

        # Case 2: Two-person household, eligible
        hh2 = nuclear_family()
        hh2.user()["age"] = 60
        hh2.spouse()["age"] = 57
        hh2.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.user()["annual_work_income"] = 40000
        hh2.spouse()["annual_work_income"] = 20000
        assert cls.__call__(hh2)

        # Case 3: Three-person household, eligible
        hh3 = nuclear_family()
        hh3.user()["age"] = 65
        hh3.spouse()["age"] = 60
        hh3.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["age"] = 15
        hh3.user()["annual_work_income"] = 50000
        hh3.spouse()["annual_work_income"] = 30000
        hh3.members[2]["annual_work_income"] = 3280
        assert cls.__call__(hh3)

        # Case 4: Single person, ineligible due to age
        hh4 = nuclear_family()
        hh4.user()["age"] = 50
        hh4.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.user()["annual_work_income"] = 60000
        hh4.spouse()["age"] = 40  # Irrelevant spouse data
        hh4.members = hh4.members[0:1]  # Remove other members
        assert not cls.__call__(hh4)

        # Case 5: Single person, ineligible due to residence
        hh5 = nuclear_family()
        hh5.user()["age"] = 55
        hh5.user()["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh5.user()["annual_work_income"] = 60000
        hh5.spouse()["age"] = 40  # Irrelevant spouse data
        hh5.members = hh5.members[0:1]  # Remove other members
        assert not cls.__call__(hh5)

        # Case 6: Three-person household, ineligible due to income
        hh6 = nuclear_family()
        hh6.user()["age"] = 70
        hh6.spouse()["age"] = 68
        hh6.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh6.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh6.members[2]["age"] = 15
        hh6.user()["annual_work_income"] = 60000
        hh6.spouse()["annual_work_income"] = 60000
        hh6.members[2]["annual_work_income"] = 10000
        assert not cls.__call__(hh6)

        # Case 7: Four-person household, eligible with added members
        hh7 = nuclear_family()
        hh7.user()["age"] = 67
        hh7.spouse()["age"] = 66
        hh7.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh7.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh7.members[2]["age"] = 15
        hh7.user()["annual_work_income"] = 40000
        hh7.spouse()["annual_work_income"] = 20000
        hh7.members.append(deepcopy(hh7.members[-1]))  # Add another child
        hh7.members[-1]["age"] = 10
        hh7.members[-1]["annual_work_income"] = 4800
        assert cls.__call__(hh7)

        # Case 8: Four-person household, ineligible due to age of all users
        hh8 = nuclear_family()
        hh8.user()["age"] = 40
        hh8.spouse()["age"] = 38
        hh8.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh8.spouse()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh8.members[2]["age"] = 10
        hh8.user()["annual_work_income"] = 40000
        hh8.spouse()["annual_work_income"] = 20000
        hh8.members.append(deepcopy(hh8.members[-1]))  # Add another child
        hh8.members[-1]["age"] = 10
        hh8.members[-1]["annual_work_income"] = 4800
        assert not cls.__call__(hh8)


class BigAppleConnect(BaseBenefitsProgram):
    """You can enroll in Big Apple Connect if you live in a NYCHA development."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["housing_type"] == HousingEnum.NYCHA_DEVELOPMENT.value:
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: Household lives in a NYCHA development
        hh1 = nuclear_family()
        hh1.members[0]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[1]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        hh1.members[2]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        assert cls.__call__(hh1)

        # Test case 2: Household does not live in a NYCHA development
        hh2 = nuclear_family()
        hh2.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[2]["housing_type"] = HousingEnum.CONDO.value
        assert not cls.__call__(hh2)


class SeniorCitizenRentIncreaseExemption(BaseBenefitsProgram):
    """To be eligible for SCRIE, you should be able to answer "yes" to all of these questions:

    Are you 62 or older?
    Is your name on the lease?
    Is your combined household income $50,000 or less in a year?
    Do you spend more than one-third of your monthly income on rent?
    Do you live in NYC in one of these types of housing?
    a rent stabilized apartment
    a rent controlled apartment
    a rent regulated hotel or single room occupancy unit
    a Mitchell-Lama development
    a Limited Dividend Housing Company development
    a Redevelopment Company development
    a Housing Development Fund Company development"""

    @staticmethod
    def __call__(hh):
        def eligible(m):
            if m["age"] >= 62:
                if m["name_is_on_lease"]:
                    if m["monthly_rent_spending"] > m.total_income() / 12:
                        if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                            if m["housing_type"] in [
                                HousingEnum.RENT_STABILIZED_APARTMENT.value,
                                HousingEnum.RENT_CONTROLLED_APARTMENT.value,
                                HousingEnum.RENT_REGULATED_HOTEL.value,
                                HousingEnum.MITCHELL_LAMA_DEVELOPMENT.value,
                                HousingEnum.LIMITED_DIVIDEND_DEVELOPMENT.value,
                                HousingEnum.REDEVELOPMENT_COMPANY_DEVELOPMENT.value,
                                HousingEnum.HDFC_DEVELOPMENT.value,
                            ]:
                                return True

        if hh.hh_annual_total_income() <= 50000:
            for m in hh.members:
                if eligible(m):
                    return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: All conditions met
        hh1 = nuclear_family()
        hh1.members[0]["age"] = 62
        hh1.members[1]["age"] = 62
        hh1.members[2]["age"] = 10
        hh1.members[0]["name_is_on_lease"] = True
        hh1.members[1]["name_is_on_lease"] = True
        hh1.members[2]["name_is_on_lease"] = True
        hh1.members[0]["annual_work_income"] = 25000
        hh1.members[1]["annual_work_income"] = 20000
        hh1.members[2]["annual_work_income"] = 0
        hh1.members[0]["monthly_rent_spending"] = 99999
        hh1.members[1]["monthly_rent_spending"] = 99999
        hh1.members[2]["monthly_rent_spending"] = 99999
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh1.members[1]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh1.members[2]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        assert cls.__call__(hh1)

        # Test case 2: Age requirement not met
        hh2 = nuclear_family()
        hh2.members[0]["age"] = 61
        hh2.members[1]["age"] = 60
        hh2.members[2]["age"] = 10
        hh2.members[0]["name_is_on_lease"] = True
        hh2.members[1]["name_is_on_lease"] = True
        hh2.members[2]["name_is_on_lease"] = True
        hh2.members[0]["annual_work_income"] = 25000
        hh2.members[1]["annual_work_income"] = 20000
        hh2.members[2]["annual_work_income"] = 0
        hh2.members[0]["monthly_rent_spending"] = 99999
        hh2.members[1]["monthly_rent_spending"] = 99999
        hh2.members[2]["monthly_rent_spending"] = 99999
        hh2.members[0]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh2.members[1]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh2.members[2]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert not cls.__call__(hh2)

        # Test case 3: Lease not in name
        hh3 = nuclear_family()
        hh3.members[0]["age"] = 62
        hh3.members[1]["age"] = 62
        hh3.members[2]["age"] = 10
        hh3.members[0]["name_is_on_lease"] = False
        hh3.members[1]["name_is_on_lease"] = False
        hh3.members[2]["name_is_on_lease"] = False
        hh3.members[0]["annual_work_income"] = 25000
        hh3.members[1]["annual_work_income"] = 20000
        hh3.members[2]["annual_work_income"] = 0
        hh3.members[0]["monthly_rent_spending"] = 99999
        hh3.members[1]["monthly_rent_spending"] = 99999
        hh3.members[2]["monthly_rent_spending"] = 99999
        hh3.members[0]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh3.members[1]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh3.members[2]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert not cls.__call__(hh3)

        # Test case 4: Income exceeds $50,000
        hh4 = nuclear_family()
        hh4.members[0]["age"] = 62
        hh4.members[1]["age"] = 62
        hh4.members[2]["age"] = 10
        hh4.members[0]["name_is_on_lease"] = True
        hh4.members[1]["name_is_on_lease"] = True
        hh4.members[2]["name_is_on_lease"] = True
        hh4.members[0]["annual_work_income"] = 30000
        hh4.members[1]["annual_work_income"] = 25000
        hh4.members[2]["annual_work_income"] = 0
        hh4.members[0]["monthly_rent_spending"] = 99999
        hh4.members[1]["monthly_rent_spending"] = 99999
        hh4.members[2]["monthly_rent_spending"] = 99999
        hh4.members[0]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh4.members[1]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh4.members[2]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert not cls.__call__(hh4)

        # Test case 5: Rent spending not more than one-third of income
        hh5 = nuclear_family()
        hh5.members[0]["age"] = 62
        hh5.members[1]["age"] = 62
        hh5.members[2]["age"] = 10
        hh5.members[0]["name_is_on_lease"] = True
        hh5.members[1]["name_is_on_lease"] = True
        hh5.members[2]["name_is_on_lease"] = True
        hh5.members[0]["annual_work_income"] = 30000
        hh5.members[1]["annual_work_income"] = 20000
        hh5.members[2]["annual_work_income"] = 0
        hh5.members[0]["monthly_rent_spending"] = 1000
        hh5.members[1]["monthly_rent_spending"] = 1000
        hh5.members[2]["monthly_rent_spending"] = 1000
        hh5.members[0]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh5.members[1]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh5.members[2]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert not cls.__call__(hh5)

        # Test case 6: Not living in qualifying housing type
        hh6 = nuclear_family()
        hh6.members[0]["age"] = 62
        hh6.members[1]["age"] = 62
        hh6.members[2]["age"] = 10
        hh6.members[0]["name_is_on_lease"] = True
        hh6.members[1]["name_is_on_lease"] = True
        hh6.members[2]["name_is_on_lease"] = True
        hh6.members[0]["annual_work_income"] = 25000
        hh6.members[1]["annual_work_income"] = 20000
        hh6.members[2]["annual_work_income"] = 0
        hh6.members[0]["monthly_rent_spending"] = 99999
        hh6.members[1]["monthly_rent_spending"] = 99999
        hh6.members[2]["monthly_rent_spending"] = 99999
        hh6.members[0]["housing_type"] = HousingEnum.HOUSE_2B.value
        hh6.members[1]["housing_type"] = HousingEnum.HOUSE_2B.value
        hh6.members[2]["housing_type"] = HousingEnum.HOUSE_2B.value
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert not cls.__call__(hh6)

        # Test case 7: Only one member meets all conditions
        hh7 = nuclear_family()
        hh7.members[0]["age"] = 62
        hh7.members[1]["age"] = 50
        hh7.members[2]["age"] = 10
        hh7.members[0]["name_is_on_lease"] = True
        hh7.members[1]["name_is_on_lease"] = False
        hh7.members[2]["name_is_on_lease"] = False
        hh7.members[0]["annual_work_income"] = 25000
        hh7.members[1]["annual_work_income"] = 25000
        hh7.members[2]["annual_work_income"] = 0
        hh7.members[0]["monthly_rent_spending"] = 99999
        hh7.members[1]["monthly_rent_spending"] = 1000
        hh7.members[2]["monthly_rent_spending"] = 1000
        hh7.members[0]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh7.members[1]["housing_type"] = HousingEnum.HOUSE_2B.value
        hh7.members[2]["housing_type"] = HousingEnum.HOUSE_2B.value
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        assert cls.__call__(hh7)

        # Test case 8: Household with additional children, still meets criteria
        hh8 = nuclear_family()
        hh8.members.append(deepcopy(hh8.members[-1]))
        hh8.members.append(deepcopy(hh8.members[-1]))
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        for member in hh8.members:
            member["age"] = 62 if member == hh8.members[0] else 10
            member["name_is_on_lease"] = True
            member["annual_work_income"] = 25000 if member == hh8.members[0] else 0
            member["monthly_rent_spending"] = 99999
            member["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        assert cls.__call__(hh8)


class PreKForAll(BaseBenefitsProgram):
    """All NYC children age 3 or 4 are eligible. This includes children with disabilities or who are learning English.
    Children do not need to be toilet trained to attend pre-K."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] == 3 or m["age"] == 4:
                return True
        return False

    @classmethod
    def test_cases(cls):

        # Test 1: Household with one 3-year-old child
        hh1 = nuclear_family()
        hh1.members[2]["age"] = 3
        assert cls.__call__(hh1)

        # Test 3: Household with one child younger than 3
        hh3 = nuclear_family()
        hh3.members[2]["age"] = 2
        assert not cls.__call__(hh3)

    # Test 2: Household with one 4-year-old child


class DisabledHomeownersExemption(BaseBenefitsProgram):
    """To be eligible for DHE, you must meet these requirements:

    Own a one-, two-, or three-family home, condo, or coop apartment.
    All property owners are people with disabilities. However, if you own the property with a spouse or sibling, only one of you need to have a disability to qualify.
    You must live on the property as your primary residence.
    The combined income for all owners must be less than or equal to $58,399.
    Your property cannot be within a housing development controlled by a Limited Profit Housing Company, Mitchell-Lama, Limited Dividend Housing Company, or redevelopment company. Contact your property manager if you're not sure.
    """

    @staticmethod
    def __call__(hh):
        if hh.user()["housing_type"] not in [
            HousingEnum.HOUSE_2B.value,
            HousingEnum.CONDO.value,
            HousingEnum.COOPERATIVE_APARTMENT.value,
        ]:
            return False
        owner_indices = []
        spouse_or_sibling_owner = False
        for m in hh.members:
            if m["is_property_owner"]:
                owner_indices.append(hh.members.index(m))
                if (
                    m["relation"] == RelationEnum.SPOUSE.value
                    or m["relation"] == RelationEnum.SIBLING.value
                ):
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
        if hh.hh_annual_total_income() > 58399:
            return False
        return True

    @classmethod
    def test_cases(cls):
        # Test 1: Eligible case with one disabled owner
        hh1 = nuclear_family()
        hh1.members[0]["is_property_owner"] = True
        hh1.members[1]["is_property_owner"] = True
        hh1.members[0]["primary_residence"] = True
        hh1.members[1]["primary_residence"] = True
        hh1.members[0]["disabled"] = True
        hh1.members[1]["disabled"] = False
        hh1.members[0]["annual_work_income"] = 20000
        hh1.members[1]["annual_work_income"] = 20000
        hh1.members[2]["annual_work_income"] = 0
        hh1.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh1.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh1.members[2]["housing_type"] = HousingEnum.CONDO.value
        assert cls.__call__(hh1)

        # Test 2: Ineligible case due to high combined income
        hh2 = nuclear_family()
        hh2.members[0]["is_property_owner"] = True
        hh2.members[1]["is_property_owner"] = True
        hh2.members[0]["primary_residence"] = True
        hh2.members[1]["primary_residence"] = True
        hh2.members[0]["disabled"] = True
        hh2.members[1]["disabled"] = False
        hh2.members[0]["annual_work_income"] = 40000
        hh2.members[1]["annual_work_income"] = 40000
        hh2.members[2]["annual_work_income"] = 0
        hh2.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh2.members[2]["housing_type"] = HousingEnum.CONDO.value
        assert not cls.__call__(hh2)

        # Test 3: Ineligible case due to not living on property
        hh3 = nuclear_family()
        hh3.members[0]["is_property_owner"] = True
        hh3.members[1]["is_property_owner"] = True
        hh3.members[0]["primary_residence"] = False
        hh3.members[1]["primary_residence"] = False
        hh3.members[0]["disabled"] = True
        hh3.members[1]["disabled"] = False
        hh3.members[0]["annual_work_income"] = 20000
        hh3.members[1]["annual_work_income"] = 20000
        hh3.members[2]["annual_work_income"] = 0
        hh3.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh3.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh3.members[2]["housing_type"] = HousingEnum.CONDO.value
        assert not cls.__call__(hh3)

        # Test 4: Ineligible case due to housing type in Limited Profit Housing
        hh4 = nuclear_family()
        hh4.members[0]["is_property_owner"] = True
        hh4.members[1]["is_property_owner"] = True
        hh4.members[0]["primary_residence"] = True
        hh4.members[1]["primary_residence"] = True
        hh4.members[0]["disabled"] = True
        hh4.members[1]["disabled"] = False
        hh4.members[0]["annual_work_income"] = 20000
        hh4.members[1]["annual_work_income"] = 20000
        hh4.members[2]["annual_work_income"] = 0
        hh4.members[0]["housing_type"] = HousingEnum.MITCHELL_LAMA_DEVELOPMENT.value
        hh4.members[1]["housing_type"] = HousingEnum.MITCHELL_LAMA_DEVELOPMENT.value
        hh4.members[2]["housing_type"] = HousingEnum.MITCHELL_LAMA_DEVELOPMENT.value
        assert not cls.__call__(hh4)

        # Test 5: Eligible case with all owners disabled
        hh5 = nuclear_family()
        hh5.members[0]["is_property_owner"] = True
        hh5.members[1]["is_property_owner"] = True
        hh5.members[0]["primary_residence"] = True
        hh5.members[1]["primary_residence"] = True
        hh5.members[0]["disabled"] = True
        hh5.members[1]["disabled"] = True
        hh5.members[0]["annual_work_income"] = 20000
        hh5.members[1]["annual_work_income"] = 20000
        hh5.members[2]["annual_work_income"] = 0
        hh5.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh5.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh5.members[2]["housing_type"] = HousingEnum.CONDO.value
        assert cls.__call__(hh5)

        # Test 6: Ineligible case due to no disabled owner
        hh6 = nuclear_family()
        hh6.members[0]["is_property_owner"] = True
        hh6.members[1]["is_property_owner"] = True
        hh6.members[0]["primary_residence"] = True
        hh6.members[1]["primary_residence"] = True
        hh6.members[0]["disabled"] = False
        hh6.members[1]["disabled"] = False
        hh6.members[0]["annual_work_income"] = 20000
        hh6.members[1]["annual_work_income"] = 20000
        hh6.members[2]["annual_work_income"] = 0
        hh6.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh6.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh6.members[2]["housing_type"] = HousingEnum.CONDO.value
        assert not cls.__call__(hh6)

        # Test 7: Eligible case with three family home
        hh7 = nuclear_family()
        hh7.members[0]["is_property_owner"] = True
        hh7.members[1]["is_property_owner"] = True
        hh7.members[0]["primary_residence"] = True
        hh7.members[1]["primary_residence"] = True
        hh7.members[0]["disabled"] = True
        hh7.members[1]["disabled"] = False
        hh7.members[0]["annual_work_income"] = 20000
        hh7.members[1]["annual_work_income"] = 20000
        hh7.members[2]["annual_work_income"] = 0
        hh7.members[0]["housing_type"] = HousingEnum.HOUSE_2B.value
        hh7.members[1]["housing_type"] = HousingEnum.HOUSE_2B.value
        hh7.members[2]["housing_type"] = HousingEnum.HOUSE_2B.value
        assert cls.__call__(hh7)

        # Test 8: Ineligible case due to more than three family home
        hh8 = nuclear_family()
        hh8.members[0]["is_property_owner"] = True
        hh8.members[1]["is_property_owner"] = True
        hh8.members[0]["primary_residence"] = True
        hh8.members[1]["primary_residence"] = True
        hh8.members[0]["disabled"] = True
        hh8.members[1]["disabled"] = False
        hh8.members[0]["annual_work_income"] = 20000
        hh8.members[1]["annual_work_income"] = 20000
        hh8.members[2]["annual_work_income"] = 0
        hh8.members[0]["housing_type"] = HousingEnum.HOUSE_4B.value
        hh8.members[1]["housing_type"] = HousingEnum.HOUSE_4B.value
        hh8.members[2]["housing_type"] = HousingEnum.HOUSE_4B.value
        assert not cls.__call__(hh8)


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

    @classmethod
    def test_cases(cls):
        # Test case 1: Valid veteran household with eligibility
        hh1 = nuclear_family()
        hh1.members[0]["is_property_owner"] = True
        hh1.members[0]["primary_residence"] = True
        hh1.members[0]["conflict_veteran"] = True
        hh1.members[0]["honorable_service"] = True
        hh1.members[1]["is_property_owner"] = True
        hh1.members[1]["primary_residence"] = True
        hh1.members[1]["conflict_veteran"] = True
        hh1.members[1]["honorable_service"] = True
        hh1.members[2]["primary_residence"] = True
        assert cls.__call__(hh1)

        # Test case 2: Invalid veteran household - missing property ownership
        hh2 = nuclear_family()
        hh2.members[0]["is_property_owner"] = False
        hh2.members[0]["primary_residence"] = True
        hh2.members[0]["conflict_veteran"] = True
        hh2.members[0]["honorable_service"] = True
        hh2.members[1]["is_property_owner"] = False
        hh2.members[1]["primary_residence"] = True
        hh2.members[1]["conflict_veteran"] = True
        hh2.members[1]["honorable_service"] = True
        hh2.members[2]["primary_residence"] = True
        assert not cls.__call__(hh2)

        # Test case 3: Invalid veteran household - not a primary residence
        hh3 = nuclear_family()
        hh3.members[0]["is_property_owner"] = True
        hh3.members[0]["primary_residence"] = False
        hh3.members[0]["conflict_veteran"] = True
        hh3.members[0]["honorable_service"] = True
        hh3.members[1]["is_property_owner"] = True
        hh3.members[1]["primary_residence"] = False
        hh3.members[1]["conflict_veteran"] = True
        hh3.members[1]["honorable_service"] = True
        hh3.members[2]["primary_residence"] = True
        assert not cls.__call__(hh3)

        # Test case 4: Invalid veteran household - no qualifying service
        hh4 = nuclear_family()
        hh4.members[0]["is_property_owner"] = True
        hh4.members[0]["primary_residence"] = True
        hh4.members[0]["conflict_veteran"] = False
        hh4.members[0]["honorable_service"] = True
        hh4.members[1]["is_property_owner"] = True
        hh4.members[1]["primary_residence"] = True
        hh4.members[1]["conflict_veteran"] = False
        hh4.members[1]["honorable_service"] = True
        hh4.members[2]["primary_residence"] = True
        assert not cls.__call__(hh4)

    # Test case 5: Invalid veteran household - no conflict


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
    Your family is at or under the following gross monthly income guidelines for your household size in the table above.
    """

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
            total_financial_resources = sum(
                [x["available_financial_resources"] for x in hh.members]
            )
            if age_req:
                if total_financial_resources < 3750:
                    return True
            else:
                if total_financial_resources < 2500:
                    return True
            return False

        def r4(hh):
            for m in hh.members:
                if m["receives_snap"]:
                    return True
                if m["receives_temporary_assistance"]:
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
            income = hh.hh_annual_total_income() / 12
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

    @classmethod
    def test_cases(cls):
        from copy import deepcopy

        hh = nuclear_family()

        # Test 1: Household with a child under age 6 and income below limit
        hh.user()["age"] = 40
        hh.spouse()["age"] = 40
        hh.members[2]["age"] = 5
        hh.members[0]["annual_work_income"] = 12000
        hh.members[1]["annual_work_income"] = 12000
        hh.members[0]["electricity_shut_off"] = True
        hh.members[1]["electricity_shut_off"] = True
        hh.members[2]["electricity_shut_off"] = True
        hh.members[0]["heating_electrical_bill_in_name"] = True
        hh.members[1]["heating_electrical_bill_in_name"] = True
        hh.members[2]["heating_electrical_bill_in_name"] = True

        assert cls.__call__(hh)

        # Test 2: Household with a senior (age 60+) and income below limit
        hh = nuclear_family()
        hh.members[0]["age"] = 60
        hh.members[1]["age"] = 40
        hh.members[2]["age"] = 10
        hh.members[0]["annual_work_income"] = 25000
        hh.members[1]["annual_work_income"] = 0
        hh.members[0]["electricity_shut_off"] = True
        hh.members[1]["electricity_shut_off"] = True
        hh.members[2]["electricity_shut_off"] = True
        hh.members[0]["heating_electrical_bill_in_name"] = True
        assert cls.__call__(hh)

        # Test 3: Household with a disabled member and income below limit
        hh = nuclear_family()
        hh.members[0]["age"] = 40
        hh.members[1]["age"] = 40
        hh.members[2]["age"] = 10
        hh.members[0]["disabled"] = True
        hh.members[1]["annual_work_income"] = 12000
        hh.members[2]["annual_work_income"] = 12000
        hh.members[0]["heating_electrical_bill_in_name"] = True
        hh.members[0]["electricity_shut_off"] = True
        assert cls.__call__(hh)

        # Test 4: Household with electricity shut off and resources under limit
        hh = nuclear_family()
        hh.members[0]["electricity_shut_off"] = True
        hh.members[1]["electricity_shut_off"] = True
        hh.members[2]["electricity_shut_off"] = True
        hh.members[0]["available_financial_resources"] = 2000
        hh.members[1]["available_financial_resources"] = 0
        hh.members[2]["available_financial_resources"] = 0
        hh.members[0][""] = 2000
        hh.members[0]["heating_electrical_bill_in_name"] = True
        assert cls.__call__(hh)

        # Test 5: Household without heating resources and resources under limit
        hh = nuclear_family()
        hh.members[0]["out_of_fuel"] = True
        hh.members[1]["out_of_fuel"] = True
        hh.members[2]["out_of_fuel"] = True
        hh.members[0]["available_financial_resources"] = 2400
        hh.members[1]["available_financial_resources"] = 0
        hh.members[2]["available_financial_resources"] = 0
        hh.members[0]["heating_electrical_bill_in_name"] = True
        hh.members[0]["receives_ssi"] = True
        assert cls.__call__(hh)

        # Test 6: Household with heating bill in member's name and income under limit
        hh = nuclear_family()
        hh.members[0]["heating_electrical_bill_in_name"] = True
        hh.members[1]["heating_electrical_bill_in_name"] = True
        hh.members[2]["heating_electrical_bill_in_name"] = True
        hh.members[0]["annual_work_income"] = 20000
        hh.members[1]["annual_work_income"] = 0
        hh.members[2]["annual_work_income"] = 0
        hh.members[0]["electricity_shut_off"] = True
        assert cls.__call__(hh)

        # Test 7: Household eligible through SNAP benefits
        hh = nuclear_family()
        hh.members[0]["receives_snap"] = True
        hh.members[1]["receives_snap"] = True
        hh.members[2]["receives_snap"] = True
        hh.members[0]["annual_work_income"] = 40000
        hh.members[1]["annual_work_income"] = 40000
        hh.members[2]["annual_work_income"] = 0
        hh.members[0]["electricity_shut_off"] = True
        hh.members[0]["heating_electrical_bill_in_name"] = True
        assert cls.__call__(hh)

        # Test 8: Large household meeting income limit
        hh = nuclear_family()
        hh.members.append(deepcopy(hh.members[-1]))
        hh.members.append(deepcopy(hh.members[-1]))
        hh.members.append(deepcopy(hh.members[-1]))
        for member in hh.members:
            member["age"] = 30
            member["annual_work_income"] = 1000
        hh.members[0]["receives_snap"] = True
        hh.members[0]["electricity_shut_off"] = True
        hh.members[0]["heating_electrical_bill_in_name"] = True
        assert cls.__call__(hh)


class NYSUnemploymentInsurance(BaseBenefitsProgram):
    """You are eligible for Unemployment Insurance (UI) if you:

    lost your job through no fault of your own (for example, you got laid off).
    worked within the last 18 months, and able to work immediately.
    are authorized to work in the US andwere authorized to work when you lost your job.
    """

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["lost_job"]:
                if -1 != m["months_since_worked"] <= 18:
                    if m["can_work_immediately"]:
                        if m["authorized_to_work_in_us"]:
                            return True
        return False

    @classmethod
    def test_cases(cls):
        # Test 1: All conditions met for UI eligibility
        hh1 = nuclear_family()
        hh1.user()["lost_job"] = True
        hh1.user()["months_since_worked"] = 6
        hh1.user()["can_work_immediately"] = True
        hh1.user()["authorized_to_work_in_us"] = True
        hh1.user()["was_authorized_to_work_when_job_lost"] = True
        hh1.spouse()["lost_job"] = False
        hh1.spouse()["can_work_immediately"] = False
        hh1.spouse()["authorized_to_work_in_us"] = True
        assert cls.__call__(hh1)

        # Test 2: At least one household member meets all conditions
        hh2 = nuclear_family()
        hh2.user()["lost_job"] = False
        hh2.spouse()["lost_job"] = True
        hh2.spouse()["months_since_worked"] = 12
        hh2.spouse()["can_work_immediately"] = True
        hh2.spouse()["authorized_to_work_in_us"] = True
        hh2.spouse()["was_authorized_to_work_when_job_lost"] = True
        hh2.members[2]["lost_job"] = False
        assert cls.__call__(hh2)

        # Test 3: No member meets "can work immediately"
        hh3 = nuclear_family()
        hh3.user()["lost_job"] = True
        hh3.user()["months_since_worked"] = 3
        hh3.user()["can_work_immediately"] = False
        hh3.user()["authorized_to_work_in_us"] = True
        hh3.user()["was_authorized_to_work_when_job_lost"] = True
        hh3.spouse()["lost_job"] = True
        hh3.spouse()["months_since_worked"] = 8
        hh3.spouse()["can_work_immediately"] = False
        assert not cls.__call__(hh3)

        # Test 4: No member authorized to work in the US
        hh4 = nuclear_family()
        hh4.user()["lost_job"] = True
        hh4.user()["months_since_worked"] = 6
        hh4.user()["can_work_immediately"] = True
        hh4.user()["authorized_to_work_in_us"] = False
        hh4.user()["was_authorized_to_work_when_job_lost"] = False
        hh4.spouse()["lost_job"] = True
        hh4.spouse()["can_work_immediately"] = True
        hh4.spouse()["authorized_to_work_in_us"] = False
        hh4.spouse()["was_authorized_to_work_when_job_lost"] = False
        assert not cls.__call__(hh4)


class SummerMeals(BaseBenefitsProgram):
    """Summer Meals is available to anyone age 18 or younger."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] < 18:
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: Household with a child under 18
        hh1 = nuclear_family()
        hh1.members[2]["age"] = 10  # The child is 10 years old
        hh1.members[0]["housing_type"] = HousingEnum.CONDO.value
        hh1.members[1]["housing_type"] = HousingEnum.CONDO.value
        hh1.members[2]["housing_type"] = HousingEnum.CONDO.value
        assert cls.__call__(hh1)
        # Test Case 2: Household with no members age 18 or younger
        hh2 = nuclear_family()
        hh2.members[0]["age"] = 40  # User
        hh2.members[1]["age"] = 40  # Spouse
        hh2.members[2]["age"] = 20  # Child is now 20 years old
        hh2.members[0]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh2.members[1]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        hh2.members[2]["housing_type"] = HousingEnum.RENT_STABILIZED_APARTMENT.value
        assert not cls.__call__(hh2)


class NYCHAResidentEconomicEmpowermentAndSustainability(BaseBenefitsProgram):
    """Anyone who lives in NYCHA housing is eligible for REES."""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["housing_type"] == HousingEnum.NYCHA_DEVELOPMENT.value:
                return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: All members live in NYCHA housing
        hh1 = nuclear_family()
        for member in hh1.members:
            member["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value
        assert cls.__call__(hh1)

        # Test Case 2: Only one member lives in NYCHA housing
        hh2 = nuclear_family()
        hh2.members[0]["housing_type"] = HousingEnum.NYCHA_DEVELOPMENT.value  # User
        hh2.members[1]["housing_type"] = HousingEnum.CONDO.value  # Spouse
        hh2.members[2]["housing_type"] = HousingEnum.HOUSE_4B.value  # Child
        assert cls.__call__(hh2)


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
            income = hh.hh_annual_total_income()
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
                if m["age"] >= 55:
                    if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                        if m["work_hours_per_week"] <= 0:
                            return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: Single member household meets all criteria
        hh1 = nuclear_family()
        hh1.members = hh1.members[:1]  # Single-member household
        hh1.members[0]["age"] = 55
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["annual_work_income"] = 0  # Below $18,825
        hh1.members[0]["work_hours_per_week"] = 0  # Unemployed
        assert cls.__call__(hh1)

        # Test case 2: Three-member household, income slightly above the limit
        hh2 = nuclear_family()
        hh2.members[0]["age"] = 60
        hh2.members[1]["age"] = 45
        hh2.members[2]["age"] = 10
        for member in hh2.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[0]["annual_investment_income"] = 32000
        hh2.members[1]["annual_investment_income"] = 2000
        hh2.members[2]["annual_investment_income"] = 0
        assert not cls.__call__(hh2)  # Total income $34,000 > $32,275

        # Test case 3: Five-member household meeting all criteria
        hh3 = nuclear_family()
        hh3.members.append(deepcopy(hh3.members[-1]))  # Add 4th member
        hh3.members.append(deepcopy(hh3.members[-1]))  # Add 5th member
        hh3.members[0]["age"] = 57
        for member in hh3.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["annual_investment_income"] = 1000  # Total $5,000 < $45,725
            member["work_hours_per_week"] = 0  # Unemployed
        assert cls.__call__(hh3)

        # Test case 4: Household with a member under 55 as the only unemployed person
        hh4 = nuclear_family()
        hh4.members[0]["age"] = 40
        hh4.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[0]["annual_work_income"] = 0
        hh4.members[0]["work_hours_per_week"] = 0
        hh4.members[1]["age"] = 60
        hh4.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[1]["annual_work_income"] = 18000  # Below $25,550 (2 members)
        hh4.members[1]["work_hours_per_week"] = 30
        assert not cls.__call__(hh4)  # Only the 40-year-old is unemployed

        # Test case 5: Household member does not live in NYC
        hh5 = nuclear_family()
        hh5.members[0]["age"] = 56
        hh5.members[0][
            "place_of_residence"
        ] = PlaceOfResidenceEnum.Jersey.value  # Not in NYC
        hh5.members[0]["annual_work_income"] = 18000  # Below $18,825
        hh5.members[0]["work_hours_per_week"] = 0  # Unemployed
        assert not cls.__call__(hh5)

        # Test case 6: Larger household with mixed eligibility
        hh6 = nuclear_family()
        hh6.members.append(deepcopy(hh6.members[-1]))  # Add 4th member
        hh6.members.append(deepcopy(hh6.members[-1]))  # Add 5th member
        hh6.members[0]["age"] = 58
        hh6.members[1]["age"] = 60
        hh6.members[2]["age"] = 40
        hh6.members[3]["age"] = 20
        hh6.members[4]["age"] = 10
        for member in hh6.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh6.members[0][
            "annual_work_income"
        ] = 0  # Meets income and unemployment criteria
        hh6.members[1][
            "annual_work_income"
        ] = 0  # Meets income and unemployment criteria
        hh6.members[2]["annual_work_income"] = 45000  # Over $52,450 for 6 members
        hh6.members[3]["annual_work_income"] = 0
        hh6.members[4]["annual_work_income"] = 0
        assert cls.__call__(hh6)  # Two older adults meet criteria


class WorkforceoneCareerCenters(BaseBenefitsProgram):
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

    @classmethod
    def test_cases(cls):
        # Test case 1: All requirements met
        hh1 = nuclear_family()
        hh1.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh1.members[0]["age"] = 25
        hh1.members[1]["age"] = 40
        hh1.members[2]["age"] = 18
        hh1.members[0]["authorized_to_work_in_us"] = True
        hh1.members[1]["authorized_to_work_in_us"] = True
        hh1.members[2]["authorized_to_work_in_us"] = True
        assert cls.__call__(hh1)

        # Test case 2: Household outside NYC
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh2.members[0]["age"] = 30
        hh2.members[1]["age"] = 35
        hh2.members[2]["age"] = 18
        hh2.members[0]["authorized_to_work_in_us"] = True
        hh2.members[1]["authorized_to_work_in_us"] = True
        hh2.members[2]["authorized_to_work_in_us"] = True
        assert not cls.__call__(hh2)

        # Test case 3: No household members are authorized to work in the U.S.
        hh3 = nuclear_family()
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[0]["age"] = 25
        hh3.members[1]["age"] = 40
        hh3.members[2]["age"] = 18
        hh3.members[0]["authorized_to_work_in_us"] = False
        hh3.members[1]["authorized_to_work_in_us"] = False
        hh3.members[2]["authorized_to_work_in_us"] = False
        assert not cls.__call__(hh3)

        # Test case 4: Household has a mix of members meeting and not meeting criteria
        hh4 = nuclear_family()
        hh4.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[1]["place_of_residence"] = PlaceOfResidenceEnum.Jersey.value
        hh4.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[0]["age"] = 17  # Below age requirement
        hh4.members[1]["age"] = 40
        hh4.members[2]["age"] = 18
        hh4.members[0]["authorized_to_work_in_us"] = True
        hh4.members[1]["authorized_to_work_in_us"] = True
        hh4.members[2]["authorized_to_work_in_us"] = False
        assert not cls.__call__(hh4)


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
            income = hh.hh_annual_total_income()
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

        if not income(hh):
            return False
        for m in hh.members:
            if m["age"] >= 60:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: Eligible household with 1 member meeting all criteria
        hh1 = nuclear_family()
        hh1.members[0]["age"] = 60  # User is 60 years or older
        hh1.members[0][
            "place_of_residence"
        ] = PlaceOfResidenceEnum.NYC.value  # NY resident
        hh1.members[0]["annual_work_income"] = 19000  # Income below limit for 1 person
        hh1.members = hh1.members[:1]  # Single-member household
        assert cls.__call__(hh1)

        # Test case 2: Household of 2 members with combined income below the threshold
        hh2 = nuclear_family()
        hh2.members[0]["age"] = 61  # User is 60 years or older
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[0]["annual_work_income"] = 15000
        hh2.members[1]["annual_work_income"] = 10000  # Spouse income
        hh2.members = hh2.members[:2]  # Two-member household
        assert cls.__call__(hh2)

        # Test case 3: Household of 3 members meeting income requirements
        hh3 = nuclear_family()
        hh3.members[0]["age"] = 60
        hh3.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh3.members[0]["annual_work_income"] = 10000
        hh3.members[1]["annual_work_income"] = 12000
        hh3.members[2]["annual_work_income"] = 1000  # Child's income
        hh3.members.append(deepcopy(hh3.members[2]))  # Add another child
        hh3.members[3]["annual_work_income"] = 0
        assert cls.__call__(hh3)

        # Test case 4: Ineligible household due to age (all under 60)
        hh4 = nuclear_family()
        hh4.members[0]["age"] = 50  # User under 60
        hh4.members[1]["age"] = 55  # Spouse under 60
        hh4.members[2]["age"] = 10  # Child
        hh4.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh4.members[0]["annual_work_income"] = 25000
        hh4.members[1]["annual_work_income"] = 15000
        assert not cls.__call__(hh4)

        # Test case 5: Ineligible household due to income
        hh5 = nuclear_family()
        hh5.members[0]["age"] = 65  # User is 60 years or older
        hh5.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh5.members[0][
            "annual_work_income"
        ] = 50000  # Income exceeds limit for 1 member
        hh5.members = hh5.members[:1]  # Single-member household
        assert not cls.__call__(hh5)

        # Test case 6: Household with non-citizens eligible due to residency and income
        hh6 = nuclear_family()
        hh6.members[0]["age"] = 62
        hh6.members[1]["age"] = 63
        hh6.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh6.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh6.members[0][
            "citizenship"
        ] = CitizenshipEnum.UNLAWFUL_RESIDENT.value  # Non-citizen
        hh6.members[1][
            "citizenship"
        ] = CitizenshipEnum.UNLAWFUL_RESIDENT.value  # Non-citizen
        hh6.members[0]["annual_work_income"] = 10000
        hh6.members[1]["annual_work_income"] = 12000
        assert cls.__call__(hh6)


class LearnEarn(BaseBenefitsProgram):
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
            income = hh.hh_annual_total_income()
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

        # if not income:
        #     return False

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
            if income(hh):
                return True
            return False

        for m in hh.members:
            if m["age"] >= 16 and m["age"] <= 21:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    if m["current_school_level"] in [9, 10, 11, 12]:
                        if m["has_ssn"]:
                            if m["authorized_to_work_in_us"]:
                                if (
                                    m["selective_service"]
                                    or not m["is_eligible_for_selective_service"]
                                ):
                                    if other(m):
                                        return True
        return False

    @classmethod
    def test_cases(cls):
        # Test case 1: Age between 16-21, NYC high school senior, meets all conditions
        hh = nuclear_family()
        hh.user()["age"] = 17
        hh.spouse()["age"] = 18
        hh.members[2]["age"] = 0
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh.user()["current_school_level"] = GradeLevelEnum.TWELVE.value
        hh.user()["has_ssn"] = True
        hh.user()["authorized_to_work_in_us"] = True
        hh.user()["selective_service"] = True
        hh.user()["receives_snap"] = True
        assert cls.__call__(hh)

        # Test case 2: Homeless youth qualifies
        hh = nuclear_family()
        hh.user()["age"] = 20
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh.user()["has_ssn"] = True
        hh.user()["authorized_to_work_in_us"] = True
        hh.user()["is_runaway"] = True
        hh.user()["current_school_level"] = GradeLevelEnum.TWELVE.value
        assert cls.__call__(hh)

        # Test case 3: Foster care youth
        hh = nuclear_family()
        hh.user()["age"] = 19
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh.user()["has_ssn"] = True
        hh.user()["authorized_to_work_in_us"] = True
        hh.user()["in_foster_care"] = True
        hh.user()["current_school_level"] = GradeLevelEnum.TWELVE.value
        assert cls.__call__(hh)

        # Test case 4: Pregnant youth
        hh = nuclear_family()
        hh.user()["age"] = 20
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh.user()["has_ssn"] = True
        hh.user()["authorized_to_work_in_us"] = True
        hh.user()["is_parent"] = True
        hh.user()["current_school_level"] = GradeLevelEnum.TWELVE.value
        assert cls.__call__(hh)

        # Test case 5: Youth with a disability
        hh = nuclear_family()
        hh.user()["age"] = 18
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh.user()["has_ssn"] = True
        hh.user()["authorized_to_work_in_us"] = True
        hh.user()["disabled"] = True
        hh.user()["current_school_level"] = GradeLevelEnum.ELEVEN.value
        assert cls.__call__(hh)

        # Test case 6: Low-income household
        hh = nuclear_family()
        hh.user()["age"] = 20
        hh.spouse()["age"] = 19
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh.user()["has_ssn"] = True
        hh.user()["authorized_to_work_in_us"] = True
        hh.user()["annual_work_income"] = 5000
        hh.spouse()["annual_work_income"] = 5000
        hh.user()["current_school_level"] = GradeLevelEnum.TWELVE.value
        hh.spouse()["current_school_level"] = GradeLevelEnum.TWELVE.value
        hh.members[2]["age"] = 5
        assert cls.__call__(hh)

        # Test case 7: Justice-involved youth
        hh = nuclear_family()
        hh.user()["age"] = 19
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh.user()["has_ssn"] = True
        hh.user()["authorized_to_work_in_us"] = True
        hh.user()["involved_in_justice_system"] = True
        hh.user()["current_school_level"] = GradeLevelEnum.TWELVE.value
        assert cls.__call__(hh)

        # Test case 8: Household income at the limit for 4 members
        hh = nuclear_family()
        hh.user()["age"] = 20
        hh.spouse()["age"] = 19
        hh.user()["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh.user()["current_school_level"] = GradeLevelEnum.TWELVE.value
        hh.user()["has_ssn"] = True
        hh.user()["authorized_to_work_in_us"] = True
        hh.user()["annual_work_income"] = 15000
        hh.spouse()["annual_work_income"] = 15000
        hh.members.append(deepcopy(hh.members[-1]))
        hh.members[3]["age"] = 1
        hh.members[3]["annual_work_income"] = 0
        assert cls.__call__(hh)


class NYCNurseFamilyPartnership(BaseBenefitsProgram):
    """You're eligible for NYC NFP if you can answer 'yes' to the below questions:

    Are you 28 weeks pregnant or less with your first baby?
    Do you live in New York City?
    Are you eligible for Medicaid?
    This program is available to all eligible parents, regardless of age, immigration status, or gender identity.
    """

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if 0 < m["months_pregnant"] < 28:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    if m["eligible_for_medicaid"]:
                        return True
        return False

    @classmethod
    def test_cases(cls):
        # NYCNurseFamilyPartnership Tests

        # Passing Test Case
        person_pass = Person.default_person(is_self=True)
        person_pass["place_of_residence"] = "New York City"
        person_pass["eligible_for_medicaid"] = True
        person_pass["months_pregnant"] = 20
        hh_pass_nfp = Household([person_pass])

        # Failing Test Case (not living in NYC)
        person_fail_residence = deepcopy(person_pass)
        person_fail_residence["place_of_residence"] = "Jersey"
        hh_fail_residence = Household([person_fail_residence])

        # Failing Test Case (not eligible for Medicaid)
        person_fail_medicaid = deepcopy(person_pass)
        person_fail_medicaid["eligible_for_medicaid"] = False
        hh_fail_medicaid = Household([person_fail_medicaid])

        # Failing Test Case (too far along in pregnancy)
        person_fail_pregnancy = deepcopy(person_pass)
        person_fail_pregnancy["months_pregnant"] = 30
        hh_fail_pregnancy = Household([person_fail_pregnancy])

        for i, hh in enumerate(
            [
                hh_pass_nfp,
            ]
        ):
            result = cls.__call__(hh)
            assert result, f"NYCNurseFamilyPartnership test {i} failed"
        for i, hh in enumerate(
            [
                hh_fail_residence,
                hh_fail_medicaid,
                hh_fail_pregnancy,
            ]
        ):
            result = cls.__call__(hh)
            assert not result, f"NYCNurseFamilyPartnership test {i} failed"


class SummerYouthEmploymentProgram(BaseBenefitsProgram):
    """You are eligible if you:live in New York City
    are 14-24 years old"""

    @staticmethod
    def __call__(hh):
        for m in hh.members:
            if m["age"] >= 14 and m["age"] <= 24:
                if m["place_of_residence"] == PlaceOfResidenceEnum.NYC.value:
                    return True
        return False

    @classmethod
    def test_cases(cls):
        # Test Case 1: All household members meet the eligibility criteria
        hh1 = nuclear_family()
        for member in hh1.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["age"] = 20  # Age between 14-24
        assert cls.__call__(hh1)

        # Test Case 2: Only the child meets the age eligibility criteria
        hh2 = nuclear_family()
        hh2.members[0]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[1]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[2]["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
        hh2.members[0]["age"] = 40  # Parent, out of age range
        hh2.members[1]["age"] = 40  # Spouse, out of age range
        hh2.members[2]["age"] = 16  # Child, within age range
        assert cls.__call__(hh2)

        # Test Case 3: No household members meet the age eligibility criteria
        hh3 = nuclear_family()
        for member in hh3.members:
            member["place_of_residence"] = PlaceOfResidenceEnum.NYC.value
            member["age"] = 30  # Out of age range
        assert not cls.__call__(hh3)

        # Test Case 4: Household members live outside New York City
        hh4 = nuclear_family()
        for member in hh4.members:
            member["place_of_residence"] = (
                PlaceOfResidenceEnum.Jersey.value
            )  # Outside NYC
            member["age"] = 20  # Within age range
        assert not cls.__call__(hh4)

    # SummerYouthEmploymentProgram Tests
    @classmethod
    def test_cases(cls):
        # Passing Test Case
        person_pass_syep = Person.default_person(is_self=True)
        person_pass_syep["place_of_residence"] = "New York City"
        person_pass_syep["age"] = 16
        hh_pass_syep = Household([person_pass_syep])

        # Failing Test Case (not living in NYC)
        person_fail_residence_syep = deepcopy(person_pass_syep)
        person_fail_residence_syep["place_of_residence"] = "Jersey"
        hh_fail_residence_syep = Household([person_fail_residence_syep])

        # Failing Test Case (age out of range)
        person_fail_age_syep = deepcopy(person_pass_syep)
        person_fail_age_syep["age"] = 25
        hh_fail_age_syep = Household([person_fail_age_syep])

        for i, hh in enumerate(
            [
                hh_pass_syep,
            ]
        ):
            result = cls.__call__(hh)
            assert result, f"SummerYouthEmploymentProgram test {i} failed"
        for i, hh in enumerate(
            [
                hh_fail_residence_syep,
                hh_fail_age_syep,
            ]
        ):
            result = cls.__call__(hh)
            assert not result, f"SummerYouthEmploymentProgram test {i} failed"
