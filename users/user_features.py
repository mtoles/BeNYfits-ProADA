import numpy as np
from names import get_full_name
from schema import And

from user_features2 import *
# from user_features3 import *


class name(BasePersonAttr):
    schema = And(str, len)
    random = lambda: get_full_name()
    default = "DefaultName"
    nl_fn = lambda n, x: f"Name: {n}"
    always_include = True


### DEMOGRAPHICS ###
class age(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 100)
    default = 20
    nl_fn = lambda n, x: f"{n} is {x} years old."


class relation(BasePersonAttr):
    schema = And(
        lambda x: x
        in (
            "self",
            "spouse",
            "child",
            "stepchild",
            "grandchild",
            "foster_child",
            "adopted_child",
            "sibling",
            "niece_nephew",
            "other_family",
            "other_non_family",
        ),
    )
    random = lambda: np.random.choice(
        [
            "spouse",
            "child",
            "stepchild",
            "grandchild",
            "foster_child",
            "adopted_child",
            "sibling_niece_nephew",
            "other_family",
            "other_non_family",
        ]
    )
    default = "self"
    nl_fn = lambda n, x: f"You are {n}" if x == "self" else f"{n} is your {x}"

    always_include = True


class disabled(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: f"{n} is disabled." if x else f"{n} is not disabled."


class has_ssn(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = True
    nl_fn = lambda n, x: (
        f"{n} has a social security number (SSN)."
        if x
        else f"{n} does not have a social security number (SSN)."
    )


class has_atin(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an adoption taxpayer ID number (ATIN)."
        if x
        else f"{n} does not have an adoption taxpayer ID number (ATIN)."
    )


class has_itin(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an individual taxpayer ID number (ITIN)."
        if x
        else f"{n} does not have an individual taxpayer ID number (ITIN)."
    )


class can_care_for_self(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = True
    nl_fn = lambda n, x: (
        f"{n} can care for themselves." if x else f"{n} cannot care for themselves."
    )


class place_of_residence(BasePersonAttr):
    schema = And(str, len)
    random = lambda: np.random.choice(["NYC", "Jersey"])
    default = "NYC"
    nl_fn = lambda n, x: f"{n} lives in {x}."


# Training Info
class enrolled_in_educational_training(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is enrolled in educational training."
        if x
        else f"{n} is not enrolled in educational training."
    )


class enrolled_in_vocational_training(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is enrolled in vocational training."
        if x
        else f"{n} is not enrolled in vocational training."
    )


# Financial Info
class work_income(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 100000)
    default = 0
    nl_fn = lambda n, x: f"{n} makes {x} per year working."


class investment_income(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 100000)
    default = 0
    nl_fn = lambda n, x: f"{n} makes {x} per year from investments."


class provides_over_half_of_own_financial_support(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = True
    nl_fn = lambda n, x: (
        f"{n} provides over half of their own financial support."
        if x
        else f"{n} does not provide over half of their own financial support."
    )


class receives_hra(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Health Reimbursement Arrangement (HRA)."
        if x
        else f"{n} does not receive Health Reimbursement Arrangement (HRA)."
    )


class receives_ssi(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Supplemental Security Income (SSI Code A)."
        if x
        else f"{n} does not receive Supplemental Security Income (SSI Code A)."
    )


class receives_snap(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Supplemental Nutrition Assistance Program (SNAP)."
        if x
        else f"{n} does not receive Supplemental Nutrition Assistance Program (SNAP)."
    )


class receives_ssdi(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Social Security Disability Insurance (SSDI)."
        if x
        else f"{n} does not receive Social Security Disability Insurance (SSDI)."
    )


class receives_va_disability(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Veterans Affairs (VA) disability pension or compensation."
        if x
        else f"{n} does not receive Veterans Affairs (VA) disability pension or compensation."
    )


class has_received_ssi_or_ssdi(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has received Supplemental Security Income (SSI) or Social Security Disability Insurance (SSDI) in the past."
        if x
        else f"{n} has not received Supplemental Security Income (SSI) or Social Security Disability Insurance (SSDI) in the past."
    )


class receives_disability_medicaid(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Medicaid due to disability."
        if x
        else f"{n} does not receive Medicaid due to disability."
    )


# School Info
class student(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: f"{n} is a student." if x else f"{n} is not a student."


class current_school_level(BasePersonAttr):
    schema = And(
        lambda x: x in ("pk", "k", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, college, None)
    )
    random = lambda: np.random.choice(
        ["pk", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, college, None]
    )
    default = None
    nl_fn = lambda n, x: (
        f"{n} is in {GRADE_DICT[x]}." if x else f"{n} is not in school."
    )


# Work Info
class works_outside_home(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} works outside the home." if x else f"{n} does not work outside the home."
    )


class looking_for_work(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is looking for work." if x else f"{n} is not looking for work."
    )


class work_hours_per_week(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 60)
    default = 0
    nl_fn = lambda n, x: f"{n} works {x} hours per week."


class days_looking_for_work(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 365)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} has been looking for work for {x} days."
        if x
        else f"{n} is not looking for work."
    )


# Family Info
class in_foster_care(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is in foster care." if x else f"{n} is not in foster care."
    )


class attending_service_for_domestic_violence(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is attending a service for domestic violence."
        if x
        else f"{n} is not attending a service for domestic violence."
    )


class has_paid_caregiver(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a paid caregiver." if x else f"{n} does not have a paid caregiver."
    )


# Housing Info
class lives_in_temp_housing(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in temporary housing."
        if x
        else f"{n} does not live in temporary housing."
    )


class name_is_on_lease(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is on the household lease."
        if x
        else f"{n} is not on the household lease."
    )


class monthly_rent_spending(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 10000)
    default = 0
    nl_fn = lambda n, x: f"{n} spends {x} per month on rent."


class lives_in_rent_stabilized_apartment(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a rent stabilized apartment."
        if x
        else f"{n} does not live in a rent stabilized apartment."
    )


class lives_in_rent_controlled_apartment(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a rent controlled apartment."
        if x
        else f"{n} does not live in a rent controlled apartment."
    )


class lives_in_mitchell_lama(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a Mitchell-Lama development."
        if x
        else f"{n} does not live in a Mitchell-Lama development."
    )


class lives_in_limited_dividend_development(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a limited dividend development."
        if x
        else f"{n} does not live in a limited dividend development."
    )


class lives_in_redevelopment_company_development(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a redevelopment company development."
        if x
        else f"{n} does not live in a redevelopment company development."
    )


class lives_in_hdfc_development(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a Housing Development Fund Corporation (HDFC) development."
        if x
        else f"{n} does not live in a Housing Development Fund Corporation (HDFC) development."
    )


class lives_in_section_213_coop(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a Section 213 coop."
        if x
        else f"{n} does not live in a Section 213 coop."
    )


class lives_in_rent_regulated_hotel(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a rent regulated hotel."
        if x
        else f"{n} does not live in a rent regulated hotel."
    )


class lives_in_rent_regulated_single(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a rent regulated single room occupancy (SRO)."
        if x
        else f"{n} does not live in a rent regulated single room occupancy (SRO)."
    )


# Relation Info
class duration_more_than_half_prev_year(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = True
    nl_fn = lambda n, x: (
        f"{n} lived with you more than half of the previous year."
        if x
        else f"{n} did not live with you more than half of the previous year."
    )


class lived_together_last_6_months(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = True
    nl_fn = lambda n, x: (
        f"{n} lived with you for the last 6 months."
        if x
        else f"{n} did not live with you for the last 6 months."
    )


class filing_jointly(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s tax filing status is married, filing jointly."
        if x
        else f"{n}'s tax filing status is single"
    )


class dependent(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is your dependent." if x else f"{n} is not your dependent."
    )


# Miscellaneous
class receiving_treatment_for_substance_abuse(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is receiving treatment for substance abuse."
        if x
        else f"{n} is not receiving treatment for substance abuse."
    )

class housing_type(BasePersonAttr):
    schema = And(str, lambda x: x in [
        "house",
        "condo",
        "cooperative apartment",
        "manufactured home",
        "farmhouse",
        "mixed use property",
        "homeless",
        "DHS shelter",
        "HRA shelter"
    ])
    random = lambda: np.random.choice([
        "house",
        "condo",
        "cooperative apartment",
        "manufactured home",
        "farmhouse",
        "mixed use property",
        "homeless"
        "DHS shelter"
        "HRA shelter"
    ])
    default = "house"
    nl_fn = lambda n, x: f"{n} owns a {x}."
class is_property_owner(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a property owner." if x else f"{n} is not a property owner."
    )

class primary_residence(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = True
    nl_fn = lambda n, x: (
        f"{n}'s home is their primary residence." if x else f"{n}'s home is not their primary residence."
    )

class months_owned_property(BasePersonAttr):
    schema = And(int, lambda v: v >= 0)
    random = lambda: np.random.randint(0, 240)  # e.g., up to 20 years
    default = 0
    nl_fn = lambda n, x: f"{n} has owned the property for {x} months."

class had_previous_sche(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} previously received SCHE on another property." 
        if x else f"{n} has not previously received SCHE on another property."
    )

### Rattandeep should have some unpushed work between 
# here
# and 
# senior citizen rent increase exemption

### New vars for Pre-K for all
class toilet_trained(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is toilet trained." 
        if x else f"{n} is not toilet trained."
    )

### New vars for Disabled Homeowners' Exemption
# I think these were already covered above

### New Vars for Veterans' Property Tax Exemption
class propery_owner_widow(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a widow of the property owner." 
        if x else f"{n} is not a widow of the property owner."
    )

class conflict_veteran(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} served in the US armed forces in the Vietnam War." 
        if x else f"{n} is not a conflict veteran."
    )

### HEAP

class heat_shut_off(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} heating system is shut off or in danger of being shut off." 
        if x else f"{n} heating system is not shut off or in danger of being shut off."
    )

class out_of_fuel(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is out of fuel." 
        if x else f"{n} is not out of fuel."
    )

class heating_bill_in_name(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a heating bill in their name." 
        if x else f"{n} does not have a heating bill in their name."
    )

class receives_temporary_assistance(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives New York OTDA Temporary Assistance." 
        if x else f"{n} does not receive New York OTDA Temporary Assistance."
    )

### NYS Unemployment Insurance

class lost_job(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lost their last job through no fault of their own." 
        if x else f"{n} did not lose their last job through no fault of their own."
    )

class months_since_worked(BasePersonAttr):
    schema = And(int, lambda v: v >= 0)
    random = lambda: np.random.randint(0, 240)  # e.g., up to 20 years
    default = 0
    nl_fn = lambda n, x: f"{n} has been unemployed for {x} months."

class can_work_immediately(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} can work immediately." 
        if x else f"{n} cannot work immediately."
    )

class authorized_to_work_in_us(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is authorized to work in the US." 
        if x else f"{n} is not authorized to work in the US."
    )

class was_authorized_to_work_when_job_lost(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} was authorized to work in the US when they lost their last job." 
        if x else f"{n} was not authorized to work in the US when they lost their last job.")
    
### Special Supplemental Nutrition Program for Women, Infants, and Children	

class months_pregnant(BasePersonAttr):
    schema = And(int, lambda v: v >= 0)
    random = lambda: np.random.randint(0, 9)
    default = 0
    nl_fn = lambda n, x: f"{n} is {x} months pregnant." if x else f"{n} is not pregnant."

class breastfeeding(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} breastfeeds a baby." 
        if x else f"{n} is not breastfeeding a baby."
    )

### NYCHA Resident Economic Empowerment and Sustainability	

class nycha_resident(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a NYCHA resident." 
        if x else f"{n} is not a NYCHA resident."
    )

### Learn & Earn	
class selective_service(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is registered for selective service." 
        if x else f"{n} is not registered for selective service."
    )

class is_eligible_for_selective_service(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is eligible for selective service." 
        if x else f"{n} is not eligible for selective service."
    )

class receives_cash_assistance(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} qualifies for and receives cash assistance." 
        if x else f"{n} does not qualify for and receive cash assistance."
    )

class is_runaway(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a runaway." 
        if x else f"{n} is not a runaway."
    )

class foster_age_out(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has aged out of foster care." 
        if x else f"{n} has not aged out of foster care or was never in it."
    )

### Family Planning Benefit Program	

class citizenship(BasePersonAttr):
    schema = And(
        lambda x: x
        in (
            "citizen_or_national",
            "lawful_resident",
            "unlawful_resident",
        ),
    )
    random = lambda: np.random.choice(
        [
            "citizen_or_national",
            "lawful_resident",
            "unlawful_resident",
        ]
    )
    default = "self"
    nl_fn = lambda n, x: f"{n} is a {x}."

### continued in user_features2