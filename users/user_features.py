import numpy as np
from names import get_full_name
from schema import And
from enum import Enum


GRADE_DICT = {
    "pk": "preschool",
    "k": "kindergarten",
    1: "1st grade",
    2: "2nd grade",
    3: "3rd grade",
    4: "4th grade",
    5: "5th grade",
    6: "6th grade",
    7: "7th grade",
    8: "8th grade",
    9: "9th grade",
    10: "10th grade",
    11: "11th grade",
    12: "12th grade",
    "college": "college",
}


class PersonAttributeMeta(type):
    """
    Metaclass for attributes that persons can have
    TODO: also used as a persons factory
    """

    registry = {}

    def __new__(cls, name, bases, attrs):
        name = (
            "lives_in_mitchell-lama" if name == "lives_in_mitchell_lama" else name
        )  # underscore problem in precomputed data. TODO: Fix
        attrs["name"] = name.lower()
        new_p_attr = super().__new__(cls, name, bases, attrs)
        if name != "BasePersonAttr":
            # check for duplicates
            assert name not in cls.registry, f"Person attribute {name} already exists"
            # class atributes
            # assert type(attrs["name"]) == str
            assert type(attrs["schema"]) == And
            assert callable(attrs["random"])
            assert callable(attrs["nl_fn"])
            # add to registry
            cls.registry[name] = new_p_attr
        return new_p_attr


class BasePersonAttr(metaclass=PersonAttributeMeta):
    # always include this attribute in the synthetic user profile even if not
    # retreived with RAG
    always_include = False  # whether to always include this attribute in RAG output

    def conform(cls, hh, person_idx, original_value):
        return original_value

    pass


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


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


class sex(BasePersonAttr):
    schema = And(lambda x: x in [y.value for y in Sex])
    random = lambda: np.random.choice(list(Sex)).value
    default = Sex.FEMALE.value
    nl_fn = lambda n, x: f"{n} is {x}."


class RelationType(Enum):
    SELF = "self"
    SPOUSE = "spouse"
    CHILD = "child"
    STEPCHILD = "stepchild"
    GRANDCHILD = "grandchild"
    FOSTER_CHILD = "foster child"
    ADOPTED_CHILD = "adopted child"
    SIBLING = "sibling"
    NIECE_NEPHEW = "niece_nephew"
    OTHER_FAMILY = "other_family"
    OTHER_NON_FAMILY = "other_non_family"


class relation(BasePersonAttr):
    schema = And(lambda x: x in RelationType._value2member_map_)
    random = lambda: np.random.choice(list(RelationType)).value
    default = RelationType.SELF.value
    nl_fn = lambda n, x: (
        f"You are {n}" if x == RelationType.SELF.value else f"{n} is your {x}"
    )

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

    def conform(cls, hh, person_idx, original_value):
        if (
            hh.members[person_idx]["citizenship"]
            != Citizenship.CITIZEN_OR_NATIONAL.value
        ):
            return False
        return original_value


class has_atin(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an adoption taxpayer ID number (ATIN)."
        if x
        else f"{n} does not have an adoption taxpayer ID number (ATIN)."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["relation"] != RelationType.ADOPTED_CHILD.value:
            return False
        return original_value


class has_itin(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an individual taxpayer ID number (ITIN)."
        if x
        else f"{n} does not have an individual taxpayer ID number (ITIN)."
    )

    def conform(cls, hh, person_idx, original_value):
        if (
            hh.members[person_idx]["citizenship"]
            != Citizenship.CITIZEN_OR_NATIONAL.value
        ):
            return False
        if hh.members[person_idx]["has_ssn"] == True:
            return False
        return original_value


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

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class enrolled_in_vocational_training(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is enrolled in vocational training."
        if x
        else f"{n} is not enrolled in vocational training."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


# Financial Info
class work_income(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 100000)
    default = 0
    nl_fn = lambda n, x: f"{n} makes {x} per year working."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


class investment_income(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 100000)
    default = 0
    nl_fn = lambda n, x: f"{n} makes {x} per year from investments."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


class provides_over_half_of_own_financial_support(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = True
    nl_fn = lambda n, x: (
        f"{n} provides over half of their own financial support."
        if x
        else f"{n} does not provide over half of their own financial support."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class receives_hra(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Health Reimbursement Arrangement (HRA)."
        if x
        else f"{n} does not receive Health Reimbursement Arrangement (HRA)."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        if hh.members[person_idx]["monthly_rent_spending"] <= 0:
            return False
        return original_value


class receives_ssi(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Supplemental Security Income (SSI Code A)."
        if x
        else f"{n} does not receive Supplemental Security Income (SSI Code A)."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class receives_snap(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Supplemental Nutrition Assistance Program (SNAP)."
        if x
        else f"{n} does not receive Supplemental Nutrition Assistance Program (SNAP)."
    )
    hh_level = True

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class receives_ssdi(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Social Security Disability Insurance (SSDI)."
        if x
        else f"{n} does not receive Social Security Disability Insurance (SSDI)."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class receives_va_disability(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Veterans Affairs (VA) disability pension or compensation."
        if x
        else f"{n} does not receive Veterans Affairs (VA) disability pension or compensation."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 17:
            return False
        return original_value


class has_received_ssi_or_ssdi(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has received Supplemental Security Income (SSI) or Social Security Disability Insurance (SSDI) in the past."
        if x
        else f"{n} has not received Supplemental Security Income (SSI) or Social Security Disability Insurance (SSDI) in the past."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class receives_disability_medicaid(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Medicaid due to disability."
        if x
        else f"{n} does not receive Medicaid due to disability."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


# School Info
class student(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: f"{n} is a student." if x else f"{n} is not a student."


class current_school_level(BasePersonAttr):
    schema = And(
        lambda x: x
        in ("pk", "k", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "college", None)
    )
    random = lambda: np.random.choice(
        ["pk", "k", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "college", None]
    )
    default = None
    nl_fn = lambda n, x: (
        f"{n} is in {GRADE_DICT[x]}." if x else f"{n} is not in school."
    )

    def conform(cls, hh, person_idx, original_value):
        if original_value is None:
            return None
        elif hh.members[person_idx]["age"] < 4:
            return None
        elif hh.members[person_idx]["age"] == 4:
            return "pk"
        elif hh.members[person_idx]["age"] == 5:
            return "k"
        elif hh.members[person_idx]["age"] == 6:
            return 1
        elif hh.members[person_idx]["age"] == 7:
            return 2
        elif hh.members[person_idx]["age"] == 8:
            return 3
        elif hh.members[person_idx]["age"] == 9:
            return 4
        elif hh.members[person_idx]["age"] == 10:
            return 5
        elif hh.members[person_idx]["age"] == 11:
            return 6
        elif hh.members[person_idx]["age"] == 12:
            return 7
        elif hh.members[person_idx]["age"] == 13:
            return 8
        elif hh.members[person_idx]["age"] == 14:
            return 9
        elif hh.members[person_idx]["age"] == 15:
            return 10
        elif hh.members[person_idx]["age"] == 16:
            return 11
        elif hh.members[person_idx]["age"] == 17:
            return 12
        elif hh.members[person_idx]["age"] >= 18:
            return "college"
        return original_value


# Work Info
class works_outside_home(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} works outside the home." if x else f"{n} does not work outside the home."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class looking_for_work(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is looking for work." if x else f"{n} is not looking for work."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class work_hours_per_week(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 60)
    default = 0
    nl_fn = lambda n, x: f"{n} works {x} hours per week."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


class days_looking_for_work(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 365)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} has been looking for work for {x} days."
        if x
        else f"{n} is not looking for work."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


# Family Info
class in_foster_care(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is in foster care." if x else f"{n} is not in foster care."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 18:
            return False
        return original_value


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
# class lives_in_temp_housing(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in temporary housing."
#         if x
#         else f"{n} does not live in temporary housing."
#     )


class name_is_on_lease(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is on the household lease."
        if x
        else f"{n} is not on the household lease."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class monthly_rent_spending(BasePersonAttr):
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 10000)
    default = 0
    nl_fn = lambda n, x: f"{n} spends {x} per month on rent."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


# class lives_in_rent_stabilized_apartment(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in a rent stabilized apartment."
#         if x
#         else f"{n} does not live in a rent stabilized apartment."
#     )


# class lives_in_rent_controlled_apartment(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in a rent controlled apartment."
#         if x
#         else f"{n} does not live in a rent controlled apartment."
#     )


# class lives_in_mitchell_lama(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in a Mitchell-Lama development."
#         if x
#         else f"{n} does not live in a Mitchell-Lama development."
#     )


# class lives_in_limited_dividend_development(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in a limited dividend development."
#         if x
#         else f"{n} does not live in a limited dividend development."
#     )


# class lives_in_redevelopment_company_development(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in a redevelopment company development."
#         if x
#         else f"{n} does not live in a redevelopment company development."
#     )


# class lives_in_hdfc_development(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in a Housing Development Fund Corporation (HDFC) development."
#         if x
#         else f"{n} does not live in a Housing Development Fund Corporation (HDFC) development."
#     )


# class lives_in_section_213_coop(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in a Section 213 coop."
#         if x
#         else f"{n} does not live in a Section 213 coop."
#     )


# class lives_in_rent_regulated_hotel(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in a rent regulated hotel."
#         if x
#         else f"{n} does not live in a rent regulated hotel."
#     )


# class lives_in_rent_regulated_single(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} lives in a rent regulated single room occupancy (SRO)."
#         if x
#         else f"{n} does not live in a rent regulated single room occupancy (SRO)."
#     )


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

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["relation"] != RelationType.SPOUSE.value:
            return False
        else:
            # set same as user
            hh.members[person_idx]["filing_jointly"] = hh.members[0]["filing_jointly"]
        return original_value


class dependent(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is your dependent." if x else f"{n} is not your dependent."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["relation"] != RelationType.SELF.value:
            return False
        return original_value


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

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class HousingType(Enum):
    HOUSE = "house"
    CONDO = "condo"
    COOPERATIVE_APARTMENT = "cooperative apartment"
    MANUFACTURED_HOME = "manufactured home"
    FARMHOUSE = "farmhouse"
    MIXED_USE_PROPERTY = "mixed use property"
    HOMELESS = "homeless"
    DHS_SHELTER = "DHS shelter"
    HRA_SHELTER = "HRA shelter"
    TEMPORARY_HOUSING = "temporary housing"
    RENT_STABILIZED_APARTMENT = "rent stabilized apartment"
    RENT_CONTROLLED_APARTMENT = "rent controlled apartment"
    MITCHELL_LAMA_DEVELOPMENT = "mitchell-lama development"
    LIMITED_DIVIDEND_DEVELOPMENT = "limited dividend development"
    REDEVELOPMENT_COMPANY_DEVELOPMENT = "redevelopment company development"
    HDFC_DEVELOPMENT = "Housing Development Fund Corporation (HDFC) development"
    SECTION_213_COOP = "Section 213 coop"
    RENT_REGULATED_HOTEL = "rent regulated hotel"
    RENT_REGULATED_SINGLE_ROOM_OCCUPANCY = "rent regulated single room occupancy (SRO)"


class housing_type(BasePersonAttr):
    schema = And(str, lambda x: x in [y.value for y in HousingType])
    random = lambda: np.random.choice(list(HousingType)).value
    default = HousingType.HOUSE.value
    nl_fn = lambda n, x: (
        f"{n} lives in a {x}." if x != "homeless" else f"{n} is homeless."
    )

    def conform(cls, hh, person_idx, original_value):
        return hh.members[0][cls.__name__]

    hh_level = True


class is_property_owner(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a property owner." if x else f"{n} is not a property owner."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class primary_residence(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = True
    nl_fn = lambda n, x: (
        f"{n}'s home is their primary residence."
        if x
        else f"{n}'s home is not their primary residence."
    )

    def conform(cls, hh, person_idx, original_value):
        return hh.members[0][cls.__name__]

    hh_level = True


class months_owned_property(BasePersonAttr):
    schema = And(int, lambda v: v >= 0)
    random = lambda: np.random.randint(0, 240)  # e.g., up to 20 years
    default = 0
    nl_fn = lambda n, x: (
        f"{n} has owned the house they live in for {x} months."
        if x
        else f"{n} has never owned the house they live in."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        if not hh.members[person_idx]["is_property_owner"]:
            return 0
        return original_value


class had_previous_sche(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} previously received SCHE on another property."
        if x
        else f"{n} has not previously received SCHE on another property."
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
        f"{n} is toilet trained." if x else f"{n} is not toilet trained."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] > 5:
            return True
        return original_value


### New vars for Disabled Homeowners' Exemption
# I think these were already covered above


### New Vars for Veterans' Property Tax Exemption
class propery_owner_widow(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a widow of the property owner."
        if x
        else f"{n} is not a widow of the property owner."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class conflict_veteran(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} served in the US armed forces in conflict in Iraq."
        if x
        else f"{n} is not a conflict veteran."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


### HEAP


class heat_shut_off(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} heating system is shut off or in danger of being shut off."
        if x
        else f"{n} heating system is not shut off or in danger of being shut off."
    )

    # requires not homeless


class out_of_fuel(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (f"{n} is out of fuel." if x else f"{n} is not out of fuel.")

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]

    # requires not homeless


class heating_bill_in_name(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a heating bill in their name."
        if x
        else f"{n} does not have a heating bill in their name."
    )

    # requires not homeless

    def conform(cls, hh, person_idx, original_value):
        if hh.members[0]["housing_type"] == HousingType.HOMELESS.value:
            return False
        return original_value


class receives_temporary_assistance(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives New York OTDA Temporary Assistance."
        if x
        else f"{n} does not receive New York OTDA Temporary Assistance."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


### NYS Unemployment Insurance


class lost_job(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lost their last job through no fault of their own."
        if x
        else f"{n} did not lose their last job through no fault of their own."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class months_since_worked(BasePersonAttr):
    schema = And(int, lambda v: v >= 0)
    random = lambda: np.random.randint(0, 240)  # e.g., up to 20 years
    default = 0
    nl_fn = lambda n, x: f"{n} has been unemployed for {x} months."

    # if under 16, set to age * 12

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return hh.members[person_idx]["age"] * 12
        return original_value


class can_work_immediately(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} can work immediately." if x else f"{n} cannot work immediately."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class authorized_to_work_in_us(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is authorized to work in the US."
        if x
        else f"{n} is not authorized to work in the US."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["has_ssn"]:
            return True
        return original_value


class was_authorized_to_work_when_job_lost(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} was authorized to work in the US when they lost their last job."
        if x
        else f"{n} was not authorized to work in the US when they lost their last job."
    )


### Special Supplemental Nutrition Program for Women, Infants, and Children


class months_pregnant(BasePersonAttr):
    schema = And(int, lambda v: v >= 0)
    random = lambda: np.random.randint(0, 9)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} is {x} months pregnant." if x else f"{n} is not pregnant."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        if hh.members[person_idx]["sex"] == Sex.MALE.value:
            return 0
        return original_value


class breastfeeding(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} breastfeeds a baby." if x else f"{n} is not breastfeeding a baby."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        if hh.members[person_idx]["sex"] == Sex.MALE.value:
            return False
        return original_value


### NYCHA Resident Economic Empowerment and Sustainability


class nycha_resident(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a NYCHA resident." if x else f"{n} is not a NYCHA resident."
    )

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


### Learn & Earn
class selective_service(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is registered for selective service."
        if x
        else f"{n} is not registered for selective service."
    )
    male_only = True


class is_eligible_for_selective_service(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is eligible for selective service."
        if x
        else f"{n} is not eligible for selective service."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 18:
            return False
        if (
            hh.members[person_idx]["citizenship"]
            != Citizenship.CITIZEN_OR_NATIONAL.value
        ):
            return False
        else:
            if hh.members[person_idx]["sex"] == Sex.MALE.value:
                return True
            else:
                return False


class receives_cash_assistance(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} qualifies for and receives cash assistance."
        if x
        else f"{n} does not qualify for and receive cash assistance."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class is_runaway(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (f"{n} is a runaway." if x else f"{n} is not a runaway.")

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] >= 18:
            return False
        return original_value


class foster_age_out(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has aged out of foster care."
        if x
        else f"{n} has not aged out of foster care or was never in it."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] >= 18:
            return False
        return original_value


### Family Planning Benefit Program


class Citizenship(Enum):
    CITIZEN_OR_NATIONAL = "citizen_or_national"
    LAWFUL_RESIDENT = "lawful_resident"
    UNLAWFUL_RESIDENT = "unlawful_resident"


class citizenship(BasePersonAttr):
    schema = And(str, lambda x: x in [c.value for c in Citizenship])
    random = lambda: np.random.choice(list(Citizenship)).value
    default = "self"
    nl_fn = lambda n, x: f"{n} is a {x}."


class responsible_for_day_to_day(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is responsible all their children's day-to-day life."
        if x
        else f"{n} is not responsible all any child's day-to-day life."
    )

    def conform(cls, hh, person_idx, original_value):
        for p in hh.members:
            if p["relation"] in [
                RelationType.CHILD.value,
                RelationType.ADOPTED_CHILD.value,
                RelationType.STEPCHILD.value,
                RelationType.GRANDCHILD.value,
            ]:
                return original_value
        return False

    # parent only


### Adult Protective Services


class hiv_aids(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has been diagnosed with HIV or AIDS."
        if x
        else f"{n} has not been diagnosed with HIV or AIDS."
    )


class can_manage_self(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} can manage their own resources, carry out daily activities, and protect themself from dangerous situations without help from others."
        if x
        else f"{n} cannot manage their own resources, carry out daily activities, and protect themself from dangerous situations without help from others."
    )


class has_family_to_help(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has people to help them manage their own resources, carry out daily activities, and protect themself from dangerous situations without help from others. "
        if x
        else f"{n} does not have people to help them manage their own resources, carry out daily activities, and protect themself from dangerous situations without help from others."
    )


### Access-A-Ride Paratransit Service


class can_access_subway_or_bus(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} can use accessible buses or subways for some or all of their trips."
        if x
        else f"{n} cannot use accessible buses or subways for some or all of their trips."
    )


class recovering_from_surgery(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is recovering from surgery."
        if x
        else f"{n} is not recovering from surgery."
    )
    # "long term condition" is covered by disabled


### CUNY Fatherhood Academy

from enum import Enum


class EducationLevel(Enum):
    HIGH_SCHOOL_DIPLOMA = "high school diploma"
    HSE_DIPLOMA = "HSE diploma"
    GED = "GED"
    NO_HIGH_SCHOOL_EQUIVALENT = "no high school diploma equivalent"


class high_school_equivalent(BasePersonAttr):
    schema = And(lambda x: x in [e.value for e in EducationLevel])
    random = lambda: np.random.choice([e.value for e in EducationLevel])
    default = EducationLevel.HIGH_SCHOOL_DIPLOMA.value
    nl_fn = lambda n, x: f"{n}'s education level is: {x}."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return EducationLevel.NO_HIGH_SCHOOL_EQUIVALENT.value
        return original_value


class college(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (f"{n} is in college." if x else f"{n} is not in college.")

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


### Newborn Home Visiting Program


class acs(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} gets help from Administration for Children's Services (ACS)."
        if x
        else f"{n} does not get help from Administration for Children's Services (ACS)."
    )


### Children and Youth with Special Health Care Needs


class chronic_health_condition(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a chronic health condition."
        if x
        else f"{n} does not have a chronic health condition."
    )


class developmental_condition(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a serious developmental condition that interferes with social functions."
        if x
        else f"{n} does not have a developmental condition."
    )


class emotional_behavioral_condition(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an serious emotional or behavioral condition that interferes with social functions."
        if x
        else f"{n} does not have an emotional or behavioral condition."
    )
    # for "need extra health care assitance" use can_care_for_self


### Outpatient Treatment Services
class mental_health_condition(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a serious mental health condition that interferes with social functions."
        if x
        else f"{n} does not have a mental health condition."
    )


class difficulty_in_regular_classroom(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has serious difficulty in a regular classroom."
        if x
        else f"{n} does not have difficulty in a regular classroom."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] > 18:
            return False
        return original_value


### Child Health Plus and Children's Medicaid


class health_insurance(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has health insurance" if x else f"{n} does not have health insurance."
    )


### Family Assessment Program


class struggles_to_relate(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} struggles to relate to their family."
        if x
        else f"{n} does not struggle to relate to their family."
    )


### NYCHA Public Housing


class emancipated_minor(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (f"{n} is emancipated." if x else f"{n} is not emancipated.")

    # contingent on being a minor

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] >= 18:
            return False
        return original_value


### Accelerated Study in Associate Programs


class accepted_to_cuny(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has completed admission requirements and is accepted to CUNY."
        if x
        else f"{n} has not completed admission requirements and is not accepted to CUNY."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class eligible_for_instate_tuition(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is eligible for in-state tuition."
        if x
        else f"{n} is not eligible for in-state tuition."
    )


class proficient_in_math(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is proficient in math." if x else f"{n} is not proficient in math."
    )


class proficient_in_english_reading_and_writing(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is proficient in English reading and writing."
        if x
        else f"{n} is not proficient in English reading and writing."
    )


class college_credits(BasePersonAttr):
    schema = And(int)
    random = lambda: np.random.randint(0, 200)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} has {x} college credits."
        if x
        else f"{n} does not have any college credits."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


class gpa(BasePersonAttr):
    schema = And(float)
    random = lambda: np.random.uniform(0.0, 4.0)
    default = 0.0
    nl_fn = lambda n, x: (f"{n} has a {x} GPA." if x else f"{n} does not have a GPA.")

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 12:
            return 0.0
        return original_value


### CUNY Start

# use: accepted_to_cuny


### Advance & Earn
class work_authorization(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is authorized to work in NYC and the US."
        if x
        else f"{n} is not authorized to work in NYC or the US."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["has_ssn"]:
            return True
        return original_value


class involved_in_justice_system(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is involved in the justice system."
        if x
        else f"{n} is not involved in the justice system."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 13:
            return False
        return original_value


### NYC YouthHealth

# None

### NYC Ladders for Leaders


class work_or_volunteer_experience(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has work or volunteer experience."
        if x
        else f"{n} does not have work or volunteer experience."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] > 0:
            return True
        if hh.members[person_idx]["age"] < 13:
            return False
        return original_value


### Jobs Plus


class lives_in_jobs_plus_neighborhood(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a Jobs Plus neighborhood."
        if x
        else f"{n} does not live in a Jobs Plus neighborhood."
    )

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


### Career and Technical Education
# None

### Veterans Affairs Supported Housing


class va_healthcare(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is eligible for VA healthcare."
        if x
        else f"{n} is not eligible for VA healthcare."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


### Cooling Assistance Benefit


class heat_exacerbated_condition(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a heat-exacerbated condition."
        if x
        else f"{n} does not have a heat-exacerbated condition."
    )


class ac(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an air conditioning unit."
        if x
        else f"{n} does not have an air conditioning unit."
    )

    # same for all members

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


class got_heap_ac(BasePersonAttr):
    schema = And(int)
    random = lambda: np.random.randint(0, 10)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} received a HEAP air conditioning unit {x} years ago."
        if x
        else f"{n} did not receive a HEAP air conditioning unit."
    )

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


class heat_included_in_rent(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has heat included in their rent."
        if x
        else f"{n} does not have heat included in their rent."
    )

    # same for all members

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


### NYC Care


class qualify_for_health_insurance(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} qualifies for a health care plan available in New York State"
        if x
        else f"{n} does not qualify for a health care plan available in New York State."
    )


### We Speak NYC


class english_language_learner(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is an English language learner."
        if x
        else f"{n} is not an English language learner."
    )


### Homebase


class at_risk_of_homelessness(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is at risk of homelessness."
        if x
        else f"{n} is not at risk of homelessness."
    )

    # contingent on not being homeless

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["housing_type"] == HousingType.HOMELESS.value:
            return True
        return hh.members[person_idx][cls.__name__]


class transitional_job(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s job is from a transitional jobs program."
        if x
        else f"{n}'s job is not from a transitional jobs program."
    )

    # contingent on having a job

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class federal_work_study(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s job is from a federal work study job"
        if x
        else f"{n}'s job is not a federal work study job."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class scholarship(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is compensated by a qualified scholarship program."
        if x
        else "is not compensated by a qualified scholarship program."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class government_job(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} works for a government agency"
        if x
        else f"{n} does not work for a government agency."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class is_therapist(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a physical therpaist licensed in New York State."
        if x
        else f"{n} is not a physical therapist licensed in New York State."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 18:
            return False
        return original_value


class contractor(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is an independent contractor."
        if x
        else f"{n} is not an independent contractor."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class wep(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is in the Work Experience Program."
        if x
        else f"{n} is not in the Work Experience Program."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class collective_bargaining(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is subject to a collective bargaining agreement waiving safe and sick leave."
        if x
        else f"{n} is not subject to a collective bargaining agreement waiving safe and sick leave."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


### COVID-19 Funeral Assistance


class covid_funeral_expenses(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = 0
    nl_fn = lambda n, x: (
        f"{n} incurred funeral expenses due to a covid death."
        if x
        else f"{n} did not incur funeral expenses due to a covid death."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


### Child Care Vouchers

# None

### NYC Financial Empowerment Centers

# None

### Family Homelessness and Eviction Prevention Supplement


class evicted_months_ago(BasePersonAttr):
    schema = And(int)
    random = lambda: np.random.randint(0, 24)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} was evicted {x} months ago." if x else f"{n} has never been evicted."
    )


class currently_being_evicted(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = 0
    nl_fn = lambda n, x: (
        f"{n} is currently being evicted."
        if x
        else f"{n} is not currently being evicted."
    )

    # contingent on not homeless

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


### NYS Paid Family Leave


class employer_opt_in(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s private employer has opted in to paid family leave."
        if x
        else f"{n}'s employer has not opted in to paid family leave."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class consecutive_work_weeks(BasePersonAttr):
    schema = And(int)
    random = lambda: np.random.randint(0, 52)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} has worked {x} consecutive weeks at their current employer."
        if x
        else f"{n} has not worked any consecutive weeks at their current employer."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return 0
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


class nonconsecutive_work_weeks(BasePersonAttr):
    schema = And(int)
    random = lambda: np.random.randint(0, 52)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} has worked {x} nonconsecutive weeks at their current employer."
        if x
        else f"{n} has not worked any nonconsecutive weeks at their current employer."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["work_income"] == 0:
            return 0
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


### Family Type Homes for Adults


class developmental_mental_day_treatment(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} attends a developmental mental day treatment program."
        if x
        else f"{n} does not attend a developmental mental day treatment program."
    )


class years_sober(BasePersonAttr):
    schema = And(int)
    random = lambda: np.random.randint(-1, 16)
    default = 10

    def get_string(n, x):
        if x == 0:
            return f"{n} is not sober."
        if x > 0:
            return f"{n} has been sober for {x} years."
        if x < 0:
            return f"{n} does not have a history of substance abuse"

    nl_fn = get_string

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return -1
        return original_value


class medication_treatment_non_compliance(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has been non-compliant with medication and treatment."
        if x
        else f"{n} has always been compliant with medication and treatment."
    )


class arson(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a history of arson."
        if x
        else f"{n} does not have a history of arson."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class verbal_abuse(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a history of verbal abuse."
        if x
        else f"{n} does not have a history of verbal abuse."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class imprisonment(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a history of imprisonment."
        if x
        else f"{n} does not have a history of imprisonment."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


### HomeFirst Down Payment Assistance


class first_time_home_buyer(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a first-time home buyer."
        if x
        else f"{n} is not a first-time home buyer."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class honorable_service(BasePersonAttr):
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has honorable military service and was discharged with a DD-214."
        if x
        else f"{n} does not have honorable military service."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value
