import random

from users.users import Household, Person

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

    self_is_employed = random.choice([True, False])

    if self_is_employed:
        primary = Person.default_employed()
        primary["works_outside_home"] = random.choice([True, False])
        primary["work_income"] = random.randint(0, 50000)
        primary["work_hours_per_week"] = random.randint(0, 80)
    else:
        primary = Person.default_unemployed()
    
    if not primary["works_outside_home"]:
        primary["looking_for_work"] = random.choice([True, False])
    
    primary["relation"] = "self"
    primary["lives_in_temp_housing"] = random.choice([True, False])
    primary["receives_hra"] = random.choice([True, False])
    primary["receives_ssi"] = random.choice([True, False])
    primary["student"] = random.choice([True, False])
    primary["enrolled_in_educational_training"] = random.choice([True, False])
    primary["enrolled_in_vocational_training"] = random.choice([True, False])
    primary["attending_service_for_domestic_violence"] = random.choice([True, False])
    primary["receiving_treatment_for_substance_abuse"] = random.choice([True, False])
    primary["has_ssn"] = random.choice([True, False])
    primary["has_itin"] = random.choice([True, False])
    primary["name_is_on_lease"] = random.choice([True, False])
    primary["monthly_rent_spending"] = random.randint(0, primary.total_income() // 12)
    primary["place_of_residence"] = random.choice(["NYC", "Jersey"])
    primary["lives_in_rent_stabilized_apartment"] = random.choice([True, False])
    primary["lives_in_rent_controlled_apartment"] = random.choice([True, False])
    primary["lives_in_mitchell-lama"] = random.choice([True, False])
    primary["lives_in_limited_dividend_development"] = random.choice([True, False])
    primary["lives_in_redevelopment_company_development"] = random.choice([True, False])
    primary["lives_in_hdfc_development"] = random.choice([True, False])
    primary["lives_in_section_213_coop"] = random.choice([True, False])
    primary["lives_in_rent_regulated_hotel"] = random.choice([True, False])
    primary["lives_in_rent_regulated_single"] = random.choice([True, False])
    primary["receives_ssi"] = random.choice([True, False])
    primary["receives_snap"] = random.choice([True, False])
    primary["receives_ssdi"] = random.choice([True, False])
    primary["receives_va_disability"] = random.choice([True, False])
    primary["receives_disability_medicaid"] = random.choice([True, False])
    primary["has_received_ssi_or_ssdi"] = random.choice([True, False])


    members.append(primary)

    spouse_exists = random.choice([True, False])

    if spouse_exists:
        spouse_is_employed = random.choice([True, False])

        if spouse_is_employed:
            spouse = Person.default_employed()
            spouse["works_outside_home"] = random.choice([True, False])
            spouse["work_income"] = random.randint(0, 50000)
            spouse["work_hours_per_week"] = random.randint(0, 80)
        else:
            spouse = Person.default_unemployed()
        
        if not spouse["works_outside_home"]:
            spouse["looking_for_work"] = random.choice([True, False])

        spouse["relation"] = "spouse"
        spouse["lives_in_temp_housing"] = random.choice([True, False])
        spouse["receives_hra"] = random.choice([True, False])
        spouse["receives_ssi"] = random.choice([True, False])
        spouse["receives_snap"] = random.choice([True, False])
        spouse["student"] = random.choice([True, False])
        spouse["enrolled_in_educational_training"] = random.choice([True, False])
        spouse["enrolled_in_vocational_training"] = random.choice([True, False])
        spouse["attending_service_for_domestic_violence"] = random.choice([True, False])
        spouse["receiving_treatment_for_substance_abuse"] = random.choice([True, False])
        spouse["has_ssn"] = random.choice([True, False])
        spouse["has_itin"] = random.choice([True, False])

        members.append(spouse)

        filing_jointly = random.choice([True, False])

        spouse["filing_jointly"] = filing_jointly
        primary["filing_jointly"] = filing_jointly

    n_children = random.randint(0, 2)

    for _ in range(n_children):
        child = Person.default_child()

        child["relation"] = "child"
        child["in_foster_care"] = random.choice([True, False])
        child["age"] = random.randint(1, 17)
        child["has_paid_caregiver"] = random.choice([True, False])
        child["duration_more_than_half_prev_year"] = random.choice([True, False])
        child["provides_over_half_of_own_financial_support"] = random.choice([True, False])

        raw_school_level = child["age"] - 5 + random.choice([-1, 0, 1])
        
        if raw_school_level == 0:
            child["current_school_level"] = "k"
        elif raw_school_level == -1:
            child["current_school_level"] = "pk"
        elif raw_school_level > 12 or raw_school_level < -1:
            child["current_school_level"] = None
        else:
            child["current_school_level"] = raw_school_level

        members.append(child)

    n_adult_dependents = random.randint(0, 2)

    for _ in range(n_adult_dependents):
        adult_dependent = Person.default_adult_dependent()

        adult_dependent["relation"] = "other_family"
        adult_dependent["age"] = random.randint(50, 95)
        adult_dependent["has_paid_caregiver"] = random.choice([True, False])
        adult_dependent["duration_more_than_half_prev_year"] = random.choice([True, False])

        members.append(adult_dependent)

    return Household(members)

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
            
            elif hh.user()["looking_for_work"]:
                self_works = True

            spouse_works = False

            if hh.spouse() is None:
                spouse_works = True
            
            elif hh.spouse()["works_outside_home"]:
                spouse_works = True
            
            elif hh.spouse()["looking_for_work"]:
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
            if hh.marriage_work_income() > 0:
                return True
            return False

        def _r4(hh) -> bool:
            user_not_at_home = False

            if hh.user()["works_outside_home"]:
                user_not_at_home = True
            elif hh.user()["looking_for_work"]:
                user_not_at_home = True
            elif hh.user()["student"]:
                user_not_at_home = True
            elif hh.user()["disabled"]:
                user_not_at_home = True

            spouse_not_at_home = False

            if hh.spouse() is None:
                spouse_not_at_home = True
            elif hh.spouse()["works_outside_home"]:
                spouse_not_at_home = True
            elif hh.spouse()["student"]:
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

        temp_housing = hh.user()["lives_in_temp_housing"]
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
            elif member["student"]:
                return True
            elif member["enrolled_in_educational_training"]:
                return True
            elif member["enrolled_in_vocational_training"]:
                return True
            elif member["looking_for_work"]:
                return True
            elif member["lives_in_temp_housing"]:
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
            elif hh.user()["lives_in_rent_stabilized_apartment"]:
                return True
            elif hh.user()["lives_in_rent_controlled_apartment"]:
                return True
            elif hh.user()["lives_in_mitchell-lama"]:
                return True
            elif hh.user()["lives_in_limited_dividend_development"]:
                return True
            elif hh.user()["lives_in_redevelopment_company_development"]:
                return True
            elif hh.user()["lives_in_hdfc_development"]:
                return True
            elif hh.user()["lives_in_section_213_coop"]:
                return True
            elif hh.user()["lives_in_rent_regulated_hotel"]:
                return True
            elif hh.user()["lives_in_rent_regulated_single"]:
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
                if m["relation"] in ["child", "stepchild", "foster_child", "grandchild"]:
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
                if m["relation"] in ["child", "stepchild", "foster_child", "grandchild"]:
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
            if hh.marriage_investment_income() < 11000:
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

class Section8HousingChoiceVoucherProgram(BaseBenefitsProgram):
    """
    Eligibility for the Section 8/HCV programs is primarily based on household income and family size.
    """

    @staticmethod
    def __call__(hh) -> bool:
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
        owners = hh.property_owners()  # Returns all members where "is_property_owner" = True
        if not owners:
            # If there are no identified owners, the exemption cannot apply
            return False

        # Helper function to determine if two people are spouses or siblings
        def are_spouses_or_siblings(p1, p2):
            rel1, rel2 = p1.get("relation", ""), p2.get("relation", "")
            # Basic logic: If either reports "spouse" or "sibling", treat them accordingly
            return ("spouse" in [rel1, rel2]) or ("sibling" in [rel1, rel2])

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
        combined_income = hh.owners_total_income()  # Sums work + investment incomes of owners
        if combined_income > 58399:
            return False

        # 5. Ownership duration (>=12 months) or previously had SCHE
        for o in owners:
            if not o["had_previous_sche"] and o["months_owned_property"] < 12:
                return False

        return True

class SeniorCitizenRentIncreaseExemption(BaseBenefitsProgram):
    """
    Senior Citizen Rent Increase Exemption (SCRIE)
    
    To be eligible, a person must answer "yes" to ALL:
      1. Age >= 62
      2. Name on lease (or share certificate if in Mitchell-Lama)
      3. Combined household income <= $50,000
      4. Monthly rent > 1/3 of monthly income
      5. Lives in NYC in one of:
         - rent stabilized apt
         - rent controlled apt
         - rent regulated hotel
         - single room occupancy (SRO)
         - Mitchell-Lama dev
         - limited dividend dev
         - redevelopment company dev
         - HDFC dev
    """

    @staticmethod
    def __call__(hh) -> bool:
        user = hh.user()  # Typically the 'self' member (the main applicant)

        # 1. Are you 62 or older?
        if user.get("age", 0) < 62:
            return False

        # 2. Is your name on the lease (or share certificate if in Mitchell-Lama)?
        #    We'll check either user["name_is_on_lease"] or if they live in Mitchell-Lama => also check "name_is_on_lease".
        #    If not in Mitchell-Lama, we just check "name_is_on_lease" = True.
        
        in_mitchell_lama = user.get("lives_in_mitchell_lama", False)
        if not user.get("name_is_on_lease", False):
            # If you're in Mitchell-Lama, you might also require share certificate logic if it's different, 
            # but for simplicity, let's just rely on name_is_on_lease here.
            return False

        # 3. Is your combined household income <= 50,000?
        if hh.hh_total_income() > 50000:
            return False

        # 4. Do you spend more than one-third of your monthly income on rent?
        monthly_income = hh.hh_total_income() / 12.0
        monthly_rent = hh.hh_monthly_rent_spending()
        # If monthly_rent > (1/3 of monthly_income), condition is satisfied.
        if monthly_rent <= (monthly_income / 3.0):
            return False

        # 5. Must live in NYC in one of:
        #    - rent stabilized apt
        #    - rent controlled apt
        #    - rent regulated hotel
        #    - single room occupancy
        #    - Mitchell-Lama dev
        #    - limited dividend dev
        #    - redevelopment company dev
        #    - HDFC dev
        # Also confirm place_of_residence = "NYC" if you need that check.
        
        # By default, let's assume "place_of_residence" is stored in the user or members:
        if user.get("place_of_residence", "") != "NYC":
            return False

        # Check that at least one of these booleans is True
        # The user could have multiple set to True, or only one. 
        # In practice, you might only have one set to True.
        if not (
            user["lives_in_rent_stabilized_apartment"]
            or user["lives_in_rent_controlled_apartment"]
            or user["lives_in_rent_regulated_hotel"]
            or user["lives_in_rent_regulated_single"]
            or user["lives_in_mitchell_lama"]
            or user["lives_in_limited_dividend_development"]
            or user["lives_in_redevelopment_company_development"]
            or user["lives_in_hdfc_development"]
        ):
            return False

        return True
