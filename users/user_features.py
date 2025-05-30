import numpy as np
from names import get_full_name
from schema import And
from enum import Enum
import random

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
    "none": "none",
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
            assert callable(attrs["demographic"])
            assert callable(attrs["nl_fn"])
            # add to registry
            cls.registry[name] = new_p_attr
        return new_p_attr

    @staticmethod
    def attribute_distribution():
        attr_names = PersonAttributeMeta.registry.keys()
        attributes_dict = {
            attr: PersonAttributeMeta.registry[attr].distribution
            for attr in attr_names
            if hasattr(PersonAttributeMeta.registry[attr], "distribution")
        }
        return attributes_dict


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
    demographic = lambda: get_full_name()
    default = "DefaultName"
    nl_fn = lambda n, x: f"Name: {n}"
    always_include = True


def sample_from_distribution(dist):
    total_weight = sum(weight for (_, weight) in dist)
    r = random.uniform(0, total_weight)

    cumulative = 0
    for low_high, weight in dist:
        low, high = low_high
        if cumulative + weight >= r:
            if isinstance(low, int) and isinstance(high, int):
                return random.randint(low, high)
            else:
                return random.uniform(low, high)
        cumulative += weight

    last_range, _ = dist[-1]
    low, high = last_range
    if isinstance(low, int) and isinstance(high, int):
        return random.randint(low, high)
    else:
        return random.uniform(low, high)


def sample_categorical(dist):
    total_weight = sum(weight for (_, weight) in dist)
    r = random.uniform(0, total_weight)

    cumulative = 0
    for label, weight in dist:
        if cumulative + weight >= r:
            return label
        cumulative += weight

    return dist[-1][0]


def yes_no_to_bool_map(yes_or_no_string):
    return yes_or_no_string.lower() == "yes"


### DEMOGRAPHICS ###
class age(BasePersonAttr):
    distribution = [
        ((0, 4), 5.4),
        ((5, 9), 5.4),
        ((10, 14), 5.6),
        ((15, 19), 5.7),
        ((20, 24), 7.0),
        ((25, 29), 9.0),
        ((30, 34), 8.8),
        ((35, 39), 7.4),
        ((40, 44), 6.5),
        ((45, 49), 6.0),
        ((50, 54), 6.2),
        ((55, 59), 6.2),
        ((60, 64), 5.8),
        ((65, 69), 4.8),
        ((70, 74), 3.9),
        ((75, 79), 2.6),
        ((80, 84), 1.8),
        ((85, 100), 1.9),
    ]
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 80)
    demographic = lambda: sample_from_distribution(age.distribution)
    default = 20
    nl_fn = lambda n, x: f"{n} is {x} years old."


class SexEnum(Enum):
    MALE = "male"
    FEMALE = "female"


class sex(BasePersonAttr):
    distribution = [("male", 47.5), ("female", 52.5)]
    schema = And(lambda x: x in [y.value for y in SexEnum])
    random = lambda: np.random.choice(list(SexEnum)).value
    demographic = lambda: sample_categorical(sex.distribution)
    default = SexEnum.FEMALE.value
    nl_fn = lambda n, x: f"{n} is {x}."


class RelationEnum(Enum):
    SELF = "self"
    SPOUSE = "spouse"
    CHILD = "child"
    STEPCHILD = "stepchild"
    GRANDCHILD = "grandchild"
    FOSTER_CHILD = "foster child"
    ADOPTED_CHILD = "adopted child"
    SIBLING = "sibling"
    NIECE_NEPHEW = "niece or nephew"
    OTHER_FAMILY = "cousin"
    OTHER_NON_FAMILY = "friend"


class relation(BasePersonAttr):
    distribution = [
        (RelationEnum.SELF.value, 0),
        (RelationEnum.SPOUSE.value, 13.8),
        (RelationEnum.CHILD.value, 27.6),
        (RelationEnum.STEPCHILD.value, 2.075),
        (RelationEnum.GRANDCHILD.value, 2.5),
        (RelationEnum.FOSTER_CHILD.value, 2.65),
        (RelationEnum.ADOPTED_CHILD.value, 13.8),
        (RelationEnum.SIBLING.value, 2.075),
        (RelationEnum.NIECE_NEPHEW.value, 2.075),
        (RelationEnum.OTHER_FAMILY.value, 2.075),
        (RelationEnum.OTHER_NON_FAMILY.value, 2.65),
    ]
    schema = And(lambda x: x in RelationEnum._value2member_map_)
    random = lambda: np.random.choice(list(RelationEnum)).value
    demographic = lambda: sample_categorical(relation.distribution)
    default = RelationEnum.SELF.value
    nl_fn = lambda n, x: (
        f"You are {n}" if x == RelationEnum.SELF.value else f"{n} is your {x}"
    )

    always_include = True


class disabled(BasePersonAttr):
    distribution = [("Yes", 11), ("No", 89)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(disabled.distribution))
    default = False
    nl_fn = lambda n, x: f"{n} is disabled." if x else f"{n} is not disabled."


class has_ssn(BasePersonAttr):
    distribution = [("Yes", 96.83), ("No", 3.17)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(has_ssn.distribution))
    default = True
    nl_fn = lambda n, x: (
        f"{n} has a social security number (SSN)."
        if x
        else f"{n} does not have a social security number (SSN)."
    )

    def conform(cls, hh, person_idx, original_value):
        if (
            hh.members[person_idx]["citizenship"]
            != CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        ):
            return False
        return original_value


class has_atin(BasePersonAttr):
    distribution = [("Yes", 4.0), ("No", 96.0)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(has_atin.distribution))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an adoption taxpayer ID number (ATIN)."
        if x
        else f"{n} does not have an adoption taxpayer ID number (ATIN)."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["relation"] != RelationEnum.ADOPTED_CHILD.value:
            return False
        return original_value


class has_itin(BasePersonAttr):
    distribution = [("Yes", 0.74), ("No", 99.26)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(has_itin.distribution))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an individual taxpayer ID number (ITIN)."
        if x
        else f"{n} does not have an individual taxpayer ID number (ITIN)."
    )

    def conform(cls, hh, person_idx, original_value):
        if (
            hh.members[person_idx]["citizenship"]
            != CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        ):
            return False
        if hh.members[person_idx]["has_ssn"] == True:
            return False
        return original_value


class can_care_for_self(BasePersonAttr):
    # distribution = [("Yes", 3.69), ("No", 96.31)]
    distribution = [("Yes", 96.31), ("No", 3.69)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(can_care_for_self.distribution)
    )
    default = True
    nl_fn = lambda n, x: (
        f"{n} can care for themselves." if x else f"{n} cannot care for themselves."
    )

    def conform(cls, hh, person_idx, original_value):
        if not hh.members[person_idx]["disabled"]:
            return False
        return original_value


class PlaceOfResidenceEnum(Enum):
    NYC = "New York City"
    Jersey = "Jersey"


class place_of_residence(BasePersonAttr):
    distribution = [("New York City", 47.1), ("Jersey", 52.9)]
    schema = And(lambda x: x in [y.value for y in PlaceOfResidenceEnum])
    random = lambda: np.random.choice([x.value for x in PlaceOfResidenceEnum])
    demographic = lambda: sample_categorical(place_of_residence.distribution)
    default = PlaceOfResidenceEnum.NYC.value
    nl_fn = lambda n, x: f"{n} lives in {x}."

    def conform(cls, hh, person_idx, original_value):
        return hh.members[0]["place_of_residence"]


# Training Info
class enrolled_in_educational_training(BasePersonAttr):
    distribution = [("Yes", 53.8), ("No", 56.2)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(enrolled_in_educational_training.distribution)
    )
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
    distribution = [("Yes", 10.0), ("No", 90.0)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(enrolled_in_vocational_training.distribution)
    )
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
class annual_work_income(BasePersonAttr):
    distribution = [
        ((0, 9999), 5.3),
        ((10000, 14999), 3.5),
        ((15000, 24999), 6.4),
        ((25000, 34999), 6.8),
        ((35000, 49999), 10.3),
        ((50000, 74999), 16.1),
        ((75000, 99999), 12.7),
        ((100000, 149999), 17.4),
        ((150000, 199999), 9.1),
        ((200000, 400000), 12.4),
    ]

    schema = And(int, lambda n: n >= 0)

    def random():
        if np.random.choice([True, False]):
            return np.random.randint(0, 100000)
        else:
            return 0

    def demographic():
        return sample_from_distribution(annual_work_income.distribution)

    default = 0
    nl_fn = lambda n, x: (
        f"{n} makes {x} per year working." if x else f"{n} does not work."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


class annual_investment_income(BasePersonAttr):
    distribution = [
        ((0, 34999), 19),
        ((35000, 52999), 44),
        ((53000, 99999), 66),
        ((100000, 250000), 88),
    ]

    schema = And(int, lambda n: n >= 0)

    # random = lambda: np.random.randint(0, 100000)
    def random():
        if np.random.choice([True, False]):
            return np.random.randint(0, 100000)
        else:
            return 0

    def demographic():
        return sample_from_distribution(annual_investment_income.distribution)

    default = 0
    nl_fn = lambda n, x: f"{n} makes {x} per year from investments."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


class provides_over_half_of_own_financial_support(BasePersonAttr):
    distribution = [("Yes", 0.12), ("No", 99.88)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(provides_over_half_of_own_financial_support.distribution)
    )
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
    distribution = [("Yes", 3.05), ("No", 96.95)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_hra.distribution)
    )
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
    distribution = [("Yes", 2.94), ("No", 97.06)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_ssi.distribution)
    )
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
    distribution = [("Yes", 11.48), ("No", 88.52)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_snap.distribution)
    )
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
    distribution = [("Yes", 2.69), ("No", 97.31)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_ssdi.distribution)
    )
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
    distribution = [("Yes", 0.65), ("No", 99.35)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_va_disability.distribution)
    )
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
    distribution = [("Yes", 4.29), ("No", 95.71)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(has_received_ssi_or_ssdi.distribution)
    )
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
    distribution = [("Yes", 57.4), ("No", 42.6)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_disability_medicaid.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Medicaid due to disability."
        if x
        else f"{n} does not receive Medicaid due to disability."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        if not hh.members[person_idx]["receives_medicaid"]:
            return False
        return original_value


# School Info
# class student(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: f"{n} is a student." if x else f"{n} is not a student."


class GradeLevelEnum(Enum):
    NONE = "none"
    PK = "pk"
    K = "k"
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    ELEVEN = 11
    TWELVE = 12
    COLLEGE = "college"


GRADE_DICT = {
    GradeLevelEnum.NONE.value: "not in school",
    GradeLevelEnum.PK.value: "in preschool (PK)",
    GradeLevelEnum.K.value: "in kindergarten",
    GradeLevelEnum.ONE.value: "in 1st grade",
    GradeLevelEnum.TWO.value: "in 2nd grade",
    GradeLevelEnum.THREE.value: "in 3rd grade",
    GradeLevelEnum.FOUR.value: "in 4th grade",
    GradeLevelEnum.FIVE.value: "in 5th grade",
    GradeLevelEnum.SIX.value: "in 6th grade",
    GradeLevelEnum.SEVEN.value: "in 7th grade",
    GradeLevelEnum.EIGHT.value: "in 8th grade",
    GradeLevelEnum.NINE.value: "in 9th grade",
    GradeLevelEnum.TEN.value: "in 10th grade",
    GradeLevelEnum.ELEVEN.value: "in 11th grade",
    GradeLevelEnum.TWELVE.value: "in 12th grade",
    GradeLevelEnum.COLLEGE.value: "in college",
}


class current_school_level(BasePersonAttr):
    # Rattan - Mixed distribution with string and int type
    distribution = [
        ("none", 0),
        ("pk", 6.4),
        ("k", 5.2),
        (1, 4.8375),
        (2, 4.8375),
        (3, 4.8375),
        (4, 4.8375),
        (5, 4.8375),
        (6, 4.8375),
        (7, 4.8375),
        (8, 4.8375),
        (9, 4.9),
        (10, 4.9),
        (11, 4.9),
        (12, 4.9),
        ("college", 30.1),
    ]

    def randomize():
        r = np.random.choice([x.value for x in GradeLevelEnum])
        if r.isdigit():
            r = int(r)
        return r

    schema = And(lambda x: x in [x.value for x in GradeLevelEnum])
    random = randomize
    demographic = lambda: sample_categorical(current_school_level.distribution)
    default = GradeLevelEnum.NONE.value
    nl_fn = lambda n, x: (f"{n} is {GRADE_DICT[x]}." if x else f"{n} is not in school.")

    def conform(cls, hh, person_idx, original_value):
        if original_value == GradeLevelEnum.NONE.value:
            return GradeLevelEnum.NONE.value
        elif hh.members[person_idx]["age"] < 4:
            return GradeLevelEnum.NONE.value
        elif hh.members[person_idx]["age"] == 4:
            return GradeLevelEnum.PK.value
        elif hh.members[person_idx]["age"] == 5:
            return GradeLevelEnum.K.value
        elif hh.members[person_idx]["age"] == 6:
            return GradeLevelEnum.ONE.value
        elif hh.members[person_idx]["age"] == 7:
            return GradeLevelEnum.TWO.value
        elif hh.members[person_idx]["age"] == 8:
            return GradeLevelEnum.THREE.value
        elif hh.members[person_idx]["age"] == 9:
            return GradeLevelEnum.FOUR.value
        elif hh.members[person_idx]["age"] == 10:
            return GradeLevelEnum.FIVE.value
        elif hh.members[person_idx]["age"] == 11:
            return GradeLevelEnum.SIX.value
        elif hh.members[person_idx]["age"] == 12:
            return GradeLevelEnum.SEVEN.value
        elif hh.members[person_idx]["age"] == 13:
            return GradeLevelEnum.EIGHT.value
        elif hh.members[person_idx]["age"] == 14:
            return GradeLevelEnum.NINE.value
        elif hh.members[person_idx]["age"] == 15:
            return GradeLevelEnum.TEN.value
        elif hh.members[person_idx]["age"] == 16:
            return GradeLevelEnum.ELEVEN.value
        elif hh.members[person_idx]["age"] == 17:
            return GradeLevelEnum.TWELVE.value
        elif hh.members[person_idx]["age"] >= 18:
            return GradeLevelEnum.COLLEGE.value
        return original_value


# Work Info
class works_outside_home(BasePersonAttr):
    distribution = [("Yes", 87.8), ("No", 12.2)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(works_outside_home.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} works outside the home." if x else f"{n} does not work outside the home."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


# use days_looking_for_work != 0
# class looking_for_work(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} is looking for work." if x else f"{n} is not looking for work."
#     )

#     def conform(cls, hh, person_idx, original_value):
#         if hh.members[person_idx]["age"] < 16:
#             return False
#         return original_value


class work_hours_per_week(BasePersonAttr):
    distribution = [
        ((0, 20), 8.3),
        ((21, 30), 7.5),
        ((31, 40), 57.2),
        ((41, 50), 16),
        ((51, 60), 7.6),
        ((61, 80), 3.4),
    ]
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 60)
    demographic = lambda: sample_from_distribution(work_hours_per_week.distribution)
    default = 0
    nl_fn = lambda n, x: f"{n} works {x} hours per week."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


class days_looking_for_work(BasePersonAttr):
    distribution = [
        ((0, 4), 31.2),
        ((5, 14), 28.9),
        ((15, 26), 17.5),
        ((27, 1000), 22.4),
    ]
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 365)
    demographic = lambda: sample_from_distribution(days_looking_for_work.distribution)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} has been looking for work for {x} days."
        if x
        else f"{n} is not looking for work."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        if hh.members[person_idx]["work_hours_per_week"] > 0:
            return 0
        if hh.members[person_idx]["annual_work_income"] > 0:
            return 0
        return original_value


# Family Info
class in_foster_care(BasePersonAttr):
    distribution = [("Yes", 3.3), ("No", 96.7)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(in_foster_care.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is in foster care." if x else f"{n} is not in foster care."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 18:
            return False
        return original_value


class attending_service_for_domestic_violence(BasePersonAttr):
    distribution = [("Yes", 0.48), ("No", 99.52)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(attending_service_for_domestic_violence.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is attending a service for domestic violence."
        if x
        else f"{n} is not attending a service for domestic violence."
    )


class has_paid_caregiver(BasePersonAttr):
    distribution = [("Yes", 16), ("No", 84)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(has_paid_caregiver.distribution)
    )
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
    distribution = [("Yes", 62.5), ("No", 37.5)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(name_is_on_lease.distribution)
    )
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
    distribution = [
        ((0, 1099), 25.2),
        ((1100, 1649), 24.8),
        ((1650, 2399), 23.9),
        ((2400, 10000000), 26.1),
    ]
    schema = And(int, lambda n: n >= 0)
    random = lambda: np.random.randint(0, 10000)
    demographic = lambda: sample_from_distribution(monthly_rent_spending.distribution)
    default = 0
    nl_fn = lambda n, x: f"{n} spends {x} per month on rent."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        if hh.members[person_idx]["housing_type"] in [
            HousingEnum.HOMELESS.value,
            HousingEnum.DHS_SHELTER.value,
            HousingEnum.HRA_SHELTER.value,
        ]:
            return 0
        if hh.members[person_idx]["is_property_owner"]:
            return 0
        if person_idx == 0:
            return 1000
        return original_value


class lived_together_last_6_months(BasePersonAttr):
    distribution = [("Yes", 36), ("No", 64)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(lived_together_last_6_months.distribution)
    )
    default = True
    nl_fn = lambda n, x: (
        f"{n} lived with you for the last 6 months."
        if x
        else f"{n} did not live with you for the last 6 months."
    )


class filing_jointly(BasePersonAttr):
    distribution = [("Yes", 30.7), ("No", 69.3)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(filing_jointly.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s tax filing status is married, filing jointly."
        if x
        else f"{n}'s tax filing status is single"
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["relation"] != RelationEnum.SPOUSE.value:
            return False
        else:
            # set same as user
            hh.members[person_idx]["filing_jointly"] = hh.members[0]["filing_jointly"]
        return original_value


class dependent(BasePersonAttr):
    distribution = [("Yes", 26.8), ("No", 73.2)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(dependent.distribution))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is your dependent." if x else f"{n} is not your dependent."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["relation"] != RelationEnum.SELF.value:
            return False
        return original_value


# Miscellaneous
class receiving_treatment_for_substance_abuse(BasePersonAttr):
    distribution = [("Yes", 12.2), ("No", 87.8)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receiving_treatment_for_substance_abuse.distribution)
    )
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


class HousingEnum(Enum):
    HOUSE_2B = "2 bedroom, 1 family house"
    HOUSE_4B = "4 bedroom, 1 family house"
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
    NYCHA_DEVELOPMENT = "NYCHA development"
    SECTION_8 = "Section 8"


class housing_type(BasePersonAttr):
    distribution = [
        (HousingEnum.HOUSE_2B.value, 17.9),
        (HousingEnum.HOUSE_4B.value, 17.9),
        (HousingEnum.CONDO.value, 2.65),
        (HousingEnum.COOPERATIVE_APARTMENT.value, 3.21),
        (HousingEnum.MANUFACTURED_HOME.value, 2.36),
        (HousingEnum.FARMHOUSE.value, 2.36),
        (HousingEnum.MIXED_USE_PROPERTY.value, 3.2),
        (HousingEnum.HOMELESS.value, 4.74),
        (HousingEnum.DHS_SHELTER.value, 2.26),
        (HousingEnum.HRA_SHELTER.value, 2.26),
        (HousingEnum.TEMPORARY_HOUSING.value, 10.51),
        (HousingEnum.RENT_STABILIZED_APARTMENT.value, 23.2),
        (HousingEnum.RENT_CONTROLLED_APARTMENT.value, 0.18),
        (HousingEnum.MITCHELL_LAMA_DEVELOPMENT.value, 1.06),
        (HousingEnum.LIMITED_DIVIDEND_DEVELOPMENT.value, 0.34),
        (HousingEnum.REDEVELOPMENT_COMPANY_DEVELOPMENT.value, 0.34),
        (HousingEnum.HDFC_DEVELOPMENT.value, 0.03),
        (HousingEnum.SECTION_213_COOP.value, 1.36),
        (HousingEnum.RENT_REGULATED_HOTEL.value, 0.18),
        (HousingEnum.RENT_REGULATED_SINGLE_ROOM_OCCUPANCY.value, 0.18),
        (HousingEnum.NYCHA_DEVELOPMENT.value, 1.89),
        (HousingEnum.SECTION_8.value, 1.89),
    ]

    schema = And(str, lambda x: x in [y.value for y in HousingEnum])
    random = lambda: np.random.choice(list(HousingEnum)).value
    demographic = lambda: sample_categorical(housing_type.distribution)
    default = HousingEnum.HOUSE_2B.value
    nl_fn = lambda n, x: (
        f"{n} lives in a {x}. It is not any other type of government housing or development."
        if x != HousingEnum.HOMELESS.value
        else f"{n} is homeless."
    )

    def conform(cls, hh, person_idx, original_value):
        return hh.members[0][cls.__name__]

    hh_level = True


class is_property_owner(BasePersonAttr):
    distribution = [("Yes", 65.6), ("No", 34.4)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(is_property_owner.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a property owner." if x else f"{n} is not a property owner."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        if hh.members[person_idx]["housing_type"] in [
            HousingEnum.COOPERATIVE_APARTMENT.value,
            HousingEnum.MIXED_USE_PROPERTY.value,
            HousingEnum.HOMELESS.value,
            HousingEnum.DHS_SHELTER.value,
            HousingEnum.HRA_SHELTER.value,
            HousingEnum.TEMPORARY_HOUSING.value,
            HousingEnum.RENT_STABILIZED_APARTMENT.value,
            HousingEnum.RENT_CONTROLLED_APARTMENT.value,
            HousingEnum.MITCHELL_LAMA_DEVELOPMENT.value,
            HousingEnum.LIMITED_DIVIDEND_DEVELOPMENT.value,
            HousingEnum.REDEVELOPMENT_COMPANY_DEVELOPMENT.value,
            HousingEnum.HDFC_DEVELOPMENT.value,
            HousingEnum.SECTION_213_COOP.value,
            HousingEnum.RENT_REGULATED_HOTEL.value,
            HousingEnum.RENT_REGULATED_SINGLE_ROOM_OCCUPANCY.value,
            HousingEnum.NYCHA_DEVELOPMENT.value,
            HousingEnum.SECTION_8.value,
        ]:
            return False
        return original_value


class primary_residence(BasePersonAttr):
    distribution = [("Yes", 65.7), ("No", 34.3)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(primary_residence.distribution)
    )
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
    demographic = lambda: np.random.randint(0, 286)  # e.g., up to 20 years
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
        else:
            return hh.members[0][cls.__name__]


class had_previous_sche(BasePersonAttr):
    distribution = [("Yes", 29.1), ("No", 70.9)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(had_previous_sche.distribution)
    )
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


# ### New vars for Pre-K for all
# class toilet_trained(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} is toilet trained." if x else f"{n} is not toilet trained."
#     )

#     def conform(cls, hh, person_idx, original_value):
#         if hh.members[person_idx]["age"] > 5:
#             return True
#         return original_value


### New vars for Disabled Homeowners' Exemption
# I think these were already covered above


### New Vars for Veterans' Property Tax Exemption
class propery_owner_widow(BasePersonAttr):
    distribution = [("Yes", 3.38), ("No", 96.62)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(propery_owner_widow.distribution)
    )
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
    distribution = [("Yes", 3.32), ("No", 96.68)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(conflict_veteran.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} served in the US armed forces in conflict in Iraq."
        if x
        else f"{n} is not a conflict veteran and did not serve in any conflict."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


### HEAP


class electricity_shut_off(BasePersonAttr):
    distribution = [("Yes", 3), ("No", 97)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(electricity_shut_off.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s electricity is shut off or in danger of being shut off."
        if x
        else f"{n}'s electricity system is not shut off or in danger of being shut off."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["housing_type"] == HousingEnum.HOMELESS.value:
            return False
        return original_value


class heat_shut_off(BasePersonAttr):
    distribution = [("Yes", 3), ("No", 97)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(heat_shut_off.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s heating system is shut off or in danger of being shut off."
        if x
        else f"{n}'s heating system is not shut off or in danger of being shut off."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["housing_type"] == HousingEnum.HOMELESS.value:
            return False
        return original_value

    # requires not homeless


class out_of_fuel(BasePersonAttr):
    distribution = [("Yes", 3), ("No", 97)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(out_of_fuel.distribution)
    )
    default = False
    nl_fn = lambda n, x: (f"{n} is out of fuel." if x else f"{n} is not out of fuel.")

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]

    # requires not homeless


class heating_electrical_bill_in_name(BasePersonAttr):
    distribution = [("Yes", 79), ("No", 21)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(heating_electrical_bill_in_name.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a heating and electrical bill in their name."
        if x
        else f"{n} does not have a heating or electrical bill in their name."
    )

    # requires not homeless

    def conform(cls, hh, person_idx, original_value):
        if hh.members[0]["housing_type"] == HousingEnum.HOMELESS.value:
            return False
        return original_value


class available_financial_resources(BasePersonAttr):
    schema = And(float)
    random = lambda: float(np.random.randint(0, 10000))

    def demographic():
        # Distribution
        # 0   --> 0
        # 30  --> 3000
        # 70  --> 8000
        # 100 --> 10000
        r = np.random.uniform(0, 100)
        if r <= 30:
            return 0.0 + ((r - 0.0)/(30.0 - 0.0)) * (3000.0 - 0.0)
        elif r <= 70:
            return 3000.0 + ((r - 30.0)/(70.0 - 30.0)) * (8000.0 - 3000.0)
        else:
            return 8000.0 + ((r - 70.0)/(100.0 - 70.0)) * (10000.0 - 8000.0)
    
    default = 0.0
    nl_fn = lambda n, x: f"{n}'s household has {x} in available financial resources."


class receives_temporary_assistance(BasePersonAttr):
    distribution = [("Yes", 8.9), ("No", 91.1)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_temporary_assistance.distribution)
    )
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
    distribution = [("Yes", 4.1), ("No", 95.9)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(lost_job.distribution))
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
    schema = And(int, lambda v: v >= -1)

    # random = lambda: np.random.randint(-1, 240)  # e.g., up to 20 years
    def random():
        r = np.random.randint(3)
        if r == 0:
            return -1
        elif r == 1:
            return 0
        else:
            return np.random.randint(1, 240)

    def demographic():
        # r = np.random.randint(3)
        # if r == 0:
        #     return -1
        # elif r == 1:
        #     return 0
        # else:
        #     return np.random.randint(1, 240)
        r = np.random.uniform(0, 100)
        if r < 3.7:  # currently unemployed
            return np.random.randint(12)
        else:
            return 0

    default = 0

    def nl_fn(n, x):
        if x == 0:
            return f"{n} is currently working"
        if x == -1:
            return f"{n} has never worked"
        return f"{n} has been unemployed for {x} months"

    # nl_fn = lambda n, x: f"{n} has been unemployed for {x} months."

    # if under 16, set to age * 12

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return hh.members[person_idx]["age"] * 12
        return original_value


class work_experience(BasePersonAttr):
    distribution = [("Yes", 62.18), ("No", 27.82)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(work_experience.distribution)
    )
    default = False
    nl_fn = lambda n, x: f"{n} has {x} years of work experience."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        if hh.members[person_idx]["months_since_worked"] > 0:
            return True
        if hh.members[person_idx]["annual_work_income"] > 0:
            return True
        return original_value


class can_work_immediately(BasePersonAttr):
    distribution = [("Yes", 5.2), ("No", 94.8)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(can_work_immediately.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} can work immediately." if x else f"{n} cannot work immediately."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class authorized_to_work_in_us(BasePersonAttr):
    distribution = [("Yes", 96.7), ("No", 3.3)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(authorized_to_work_in_us.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is authorized to work in the US and NYC."
        if x
        else f"{n} is not authorized to work in the US and NYC."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["has_ssn"]:
            return True
        return original_value


class was_authorized_to_work_when_job_lost(BasePersonAttr):
    distribution = [("Yes", 96.7), ("No", 3.3)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(was_authorized_to_work_when_job_lost.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} was authorized to work in the US when they lost their last job."
        if x
        else f"{n} was not authorized to work in the US when they lost their last job."
    )


### Special Supplemental Nutrition Program for Women, Infants, and Children


class is_parent(BasePersonAttr):
    distribution = [("Yes", 63.6), ("No", 36.4)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(is_parent.distribution))
    default = False
    nl_fn = lambda n, x: (f"{n} is a parent." if x else f"{n} is not a parent.")

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["relation"] in [
            RelationEnum.SELF.value,
            RelationEnum.SPOUSE.value,
        ]:
            for m in hh.members:
                if m["relation"] in [
                    RelationEnum.ADOPTED_CHILD,
                    RelationEnum.CHILD,
                    RelationEnum.FOSTER_CHILD,
                    RelationEnum.GRANDCHILD,
                    RelationEnum.STEPCHILD,
                ]:
                    return True
        return original_value


class months_pregnant(BasePersonAttr):
    distribution = [
        (0, 96),
        (1, 0.5),
        (2, 0.5),
        (3, 0.5),
        (4, 0.5),
        (5, 0.5),
        (6, 0.5),
        (7, 0.5),
        (8, 0.5),
        (9, 0.5),
    ]

    schema = And(int, lambda v: v >= 0)

    # random = lambda: np.random.randint(0, 9)
    def random():
        if np.random.choice([True, False]):
            return np.random.randint(1, 9)
        else:
            return 0

    def demographic():
        return sample_categorical(months_pregnant.distribution)

    default = 0
    nl_fn = lambda n, x: (
        f"{n} is {x} months pregnant." if x else f"{n} is not pregnant."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return 0
        if hh.members[person_idx]["sex"] == SexEnum.MALE.value:
            return 0
        return original_value


class breastfeeding(BasePersonAttr):
    distribution = [("Yes", 2.9), ("No", 97.1)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(breastfeeding.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} breastfeeds a baby." if x else f"{n} is not breastfeeding a baby."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return False
        if hh.members[person_idx]["sex"] == SexEnum.MALE.value:
            return False
        return original_value


### NYCHA Resident Economic Empowerment and Sustainability


# class nycha_resident(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} is a NYCHA resident." if x else f"{n} is not a NYCHA resident."
#     )

#     def conform(cls, hh, person_idx, original_value):
#         return hh.members[person_idx][cls.__name__]


### Learn & Earn
class selective_service(BasePersonAttr):
    distribution = [("Yes", 84), ("No", 16)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(selective_service.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is registered for selective service."
        if x
        else f"{n} is not registered for selective service."
    )
    male_only = True


class is_eligible_for_selective_service(BasePersonAttr):
    distribution = [("Yes", 5.27), ("No", 94.73)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(is_eligible_for_selective_service.distribution)
    )
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
            != CitizenshipEnum.CITIZEN_OR_NATIONAL.value
        ):
            return False
        else:
            if hh.members[person_idx]["sex"] == SexEnum.MALE.value:
                return True
            else:
                return False


class receives_cash_assistance(BasePersonAttr):
    distribution = [("Yes", 9.3), ("No", 80.7)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_cash_assistance.distribution)
    )
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
    distribution = [("Yes", 6.93), ("No", 93.07)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(is_runaway.distribution)
    )
    default = False
    nl_fn = lambda n, x: (f"{n} is a runaway." if x else f"{n} is not a runaway.")

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] >= 18:
            return False
        return original_value


class foster_age_out(BasePersonAttr):
    distribution = [("Yes", 6.04), ("No", 93.96)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(foster_age_out.distribution)
    )
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


class CitizenshipEnum(Enum):
    CITIZEN_OR_NATIONAL = "citizen_or_national"
    LAWFUL_RESIDENT = "lawful_resident"
    UNLAWFUL_RESIDENT = "unlawful_resident"


class citizenship(BasePersonAttr):
    distribution = [
        ("citizen_or_national", 73.7),
        ("lawful_resident", 23),
        ("unlawful_resident", 3.3),
    ]
    schema = And(str, lambda x: x in [c.value for c in CitizenshipEnum])
    random = lambda: np.random.choice(list(CitizenshipEnum)).value
    demographic = lambda: sample_categorical(citizenship.distribution)
    default = CitizenshipEnum.CITIZEN_OR_NATIONAL.value
    nl_fn = lambda n, x: f"{n} is a {x}."


class responsible_for_day_to_day(BasePersonAttr):
    distribution = [("Yes", 8.1), ("No", 91.9)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(responsible_for_day_to_day.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is responsible all their children's day-to-day life."
        if x
        else f"{n} is not responsible all any child's day-to-day life."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 18:
            return False
        for p in hh.members:
            if p["relation"] in [
                RelationEnum.CHILD.value,
                RelationEnum.ADOPTED_CHILD.value,
                RelationEnum.STEPCHILD.value,
                RelationEnum.GRANDCHILD.value,
            ]:
                return original_value
        return False


### Adult Protective Services


class hiv_aids(BasePersonAttr):
    distribution = [("Yes", 0.36), ("No", 99.64)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(hiv_aids.distribution))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has been diagnosed with HIV or AIDS."
        if x
        else f"{n} has not been diagnosed with HIV or AIDS."
    )


class can_manage_self(BasePersonAttr):
    distribution = [("Yes", 53.4), ("No", 46.6)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(can_manage_self.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} can manage their own resources, carry out daily activities, and protect themself from dangerous situations without help from others."
        if x
        else f"{n} cannot manage their own resources, carry out daily activities, and protect themself from dangerous situations without help from others."
    )


class has_family_to_help(BasePersonAttr):
    distribution = [("Yes", 58.7), ("No", 41.3)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(has_family_to_help.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has people to help them manage their own resources, carry out daily activities, and protect themself from dangerous situations without help from others. "
        if x
        else f"{n} does not have people to help them manage their own resources, carry out daily activities, and protect themself from dangerous situations without help from others."
    )


### Access-A-Ride Paratransit Service


class can_access_subway_or_bus(BasePersonAttr):
    distribution = [("Yes", 1.1), ("No", 98.9)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(can_access_subway_or_bus.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} can use accessible buses or subways for some or all of their trips."
        if x
        else f"{n} cannot use accessible buses or subways for some or all of their trips."
    )


class recovering_from_surgery(BasePersonAttr):
    distribution = [("Yes", 11.3), ("No", 88.7)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(recovering_from_surgery.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is recovering from surgery."
        if x
        else f"{n} is not recovering from surgery."
    )
    # "long term condition" is covered by disabled


### CUNY Fatherhood Academy

from enum import Enum


class EducationLevelEnum(Enum):
    HIGH_SCHOOL_DIPLOMA = "high school diploma"
    HSE_DIPLOMA = "HSE diploma"
    GED = "GED"
    NO_HIGH_SCHOOL_EQUIVALENT = "no high school diploma equivalent"


class high_school_equivalent(BasePersonAttr):
    distribution = [
        ("high school diploma", 78.11),
        ("HSE diploma", 2.295),
        ("GED", 2.295),
        ("no high school diploma equivalent", 17.3),
    ]

    schema = And(lambda x: x in [e.value for e in EducationLevelEnum])
    random = lambda: np.random.choice([e.value for e in EducationLevelEnum])
    demographic = lambda: sample_categorical(high_school_equivalent.distribution)
    default = EducationLevelEnum.HIGH_SCHOOL_DIPLOMA.value
    nl_fn = lambda n, x: f"{n}'s education level is: {x}."

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] < 16:
            return EducationLevelEnum.NO_HIGH_SCHOOL_EQUIVALENT.value
        return original_value


# class college(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (f"{n} is in college." if x else f"{n} is not in college.")

#     def conform(cls, hh, person_idx, original_value):
#         if hh.members[person_idx]["age"] < 16:
#             return False
#         if not hh.members[person_idx]["student"]:
#             return False
#         return original_value


### Newborn Home Visiting Program


class acs(BasePersonAttr):
    distribution = [("Yes", 89), ("No", 11)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(acs.distribution))
    default = False
    nl_fn = lambda n, x: (
        f"{n} gets help from Administration for Children's Services (ACS)."
        if x
        else f"{n} does not get help from Administration for Children's Services (ACS)."
    )


### Children and Youth with Special Health Care Needs


class chronic_health_condition(BasePersonAttr):
    distribution = [("Yes", 40), ("No", 60)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(chronic_health_condition.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a chronic health condition."
        if x
        else f"{n} does not have a chronic health condition."
    )


class developmental_condition(BasePersonAttr):
    distribution = [("Yes", 8.56), ("No", 91.44)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(developmental_condition.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a serious developmental condition that interferes with social functions."
        if x
        else f"{n} does not have a developmental condition."
    )


class emotional_behavioral_condition(BasePersonAttr):
    distribution = [("Yes", 18), ("No", 82)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(emotional_behavioral_condition.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an serious emotional or behavioral condition that interferes with social functions."
        if x
        else f"{n} does not have an emotional or behavioral condition and is not at risk."
    )
    # for "need extra health care assitance" use can_care_for_self


### Outpatient Treatment Services
class mental_health_condition(BasePersonAttr):
    distribution = [("Yes", 23.08), ("No", 76.92)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(mental_health_condition.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a serious mental health condition that interferes with social functions."
        if x
        else f"{n} does not have a mental health condition."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] > 18:
            return False
        return original_value


### Child Health Plus and Children's Medicaid


class health_insurance(BasePersonAttr):
    distribution = [("Yes", 92), ("No", 8)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(health_insurance.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has health insurance" if x else f"{n} is not covered by health insurance."
    )


### Family Assessment Program


class struggles_to_relate(BasePersonAttr):
    distribution = [("Yes", 40), ("No", 60)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(struggles_to_relate.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} struggles to relate to their family."
        if x
        else f"{n} does not struggle to relate to their family."
    )


### NYCHA Public Housing


class emancipated_minor(BasePersonAttr):
    distribution = [("Yes", 9), ("No", 91)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(emancipated_minor.distribution)
    )
    default = False
    nl_fn = lambda n, x: (f"{n} is emancipated." if x else f"{n} is not emancipated.")

    # contingent on being a minor

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["age"] >= 18:
            return False
        return original_value


### Accelerated Study in Associate Programs


class accepted_to_cuny(BasePersonAttr):
    distribution = [("Yes", 64), ("No", 36)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(accepted_to_cuny.distribution)
    )
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
    distribution = [("Yes", 50), ("No", 50)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(eligible_for_instate_tuition.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is eligible for in-state tuition."
        if x
        else f"{n} is not eligible for in-state tuition."
    )


class proficient_in_math(BasePersonAttr):
    distribution = [("Yes", 28), ("No", 72)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(proficient_in_math.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is proficient in math." if x else f"{n} is not proficient in math."
    )


class proficient_in_english_reading_and_writing(BasePersonAttr):
    distribution = [("Yes", 33), ("No", 67)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(proficient_in_english_reading_and_writing.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is proficient in English reading and writing."
        if x
        else f"{n} is not proficient in English reading and writing."
    )


class college_credits(BasePersonAttr):
    schema = And(int)

    # random = lambda: np.random.randint(0, 200)
    def random():
        if np.random.choice([True, False]):
            return np.random.randint(1, 200)
        else:
            return 0

    def demographic():
        categories = ['less_than_hs', 'hs', 'some_college', 'associate', 'bachelor', 'advanced']
        probabilities = [0.09, 0.28, 0.15, 0.10, 0.23, 0.15]

        choice = np.random.choice(categories, p=probabilities)

        if choice in ('less_than_hs', 'hs'):
            return 0
        elif choice == 'some_college':
            return np.random.randint(0, 51)
        else:
            return 200

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
    demographic = lambda: np.clip(np.random.normal(3.15, 1.0), 0.0, 4.0)
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
    distribution = [("Yes", 44.3), ("No", 55.7)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(work_authorization.distribution)
    )
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
    distribution = [("Yes", 2.1), ("No", 97.9)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(involved_in_justice_system.distribution)
    )
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
    distribution = [("Yes", 25), ("No", 75)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(work_or_volunteer_experience.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has work or volunteer experience."
        if x
        else f"{n} does not have work or volunteer experience."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] > 0:
            return True
        if hh.members[person_idx]["age"] < 13:
            return False
        return original_value


### Jobs Plus


class lives_in_jobs_plus_neighborhood(BasePersonAttr):
    distribution = [("Yes", 10.10), ("No", 89.9)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(lives_in_jobs_plus_neighborhood.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} lives in a Jobs Plus neighborhood."
        if x
        else f"{n} does not live in a Jobs Plus neighborhood."
    )

    def conform(cls, hh, person_idx, original_value):
        if (
            hh.members[person_idx]["place_of_residence"]
            != PlaceOfResidenceEnum.NYC.value
        ):
            return False
        return hh.members[person_idx][cls.__name__]


### Career and Technical Education
# None

### Veterans Affairs Supported Housing


class va_healthcare(BasePersonAttr):
    distribution = [("Yes", 39), ("No", 61)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(va_healthcare.distribution)
    )
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
    distribution = [("Yes", 11), ("No", 89)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(heat_exacerbated_condition.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has a heat-exacerbated condition."
        if x
        else f"{n} does not have a heat-exacerbated condition."
    )


class ac(BasePersonAttr):
    distribution = [("Yes", 89), ("No", 11)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(ac.distribution))
    default = False
    nl_fn = lambda n, x: (
        f"{n} has an air conditioning unit."
        if x
        else f"{n} does not have an air conditioning unit or the unit is over 5 years old."
    )

    # same for all members

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


class got_heap_ac(BasePersonAttr):
    schema = And(int)
    random = lambda: np.random.randint(0, 10)
    demographic = lambda: np.random.randint(0, 10)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} received a HEAP air conditioning unit {x} years ago."
        if x
        else f"{n} did not receive a HEAP air conditioning unit."
    )

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


class heat_included_in_rent(BasePersonAttr):
    distribution = [("Yes", 34), ("No", 66)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(heat_included_in_rent.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has heat included in their rent."
        if x
        else f"{n} does not have heat included in their rent."
    )


### NYC Care


class qualify_for_health_insurance(BasePersonAttr):
    distribution = [("Yes", 0.016), ("No", 99.984)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(qualify_for_health_insurance.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} qualifies for a health care plan available in New York State"
        if x
        else f"{n} does not qualify for a health care plan available in New York State."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["health_insurance"] == True:
            return True
        return original_value


### We Speak NYC


# class english_language_learner(BasePersonAttr):
#     schema = And(bool)
#     random = lambda: bool(np.random.choice([True, False]))
#     default = False
#     nl_fn = lambda n, x: (
#         f"{n} is an English language learner."
#         if x
#         else f"{n} is not an English language learner."
#     )
# use proficient_in_english_reading_and_writing


### Homebase


class at_risk_of_homelessness(BasePersonAttr):
    distribution = [("Yes", 3.31), ("No", 96.69)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(at_risk_of_homelessness.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is at risk of homelessness."
        if x
        else f"{n} is not at risk of homelessness."
    )

    # contingent on not being homeless

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["housing_type"] == HousingEnum.HOMELESS.value:
            return True
        return hh.members[person_idx][cls.__name__]


class transitional_job(BasePersonAttr):
    distribution = [("Yes", 0.33), ("No", 99.67)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(transitional_job.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s job is from a transitional jobs program."
        if x
        else f"{n}'s job is not from a transitional jobs program."
    )

    # contingent on having a job

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class federal_work_study(BasePersonAttr):
    distribution = [("Yes", 0.2), ("No", 99.8)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(federal_work_study.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s job is from a federal work study job"
        if x
        else f"{n}'s job is not a federal work study job."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class scholarship(BasePersonAttr):
    distribution = [("Yes", 25), ("No", 75)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(scholarship.distribution)
    )
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
    distribution = [("Yes", 3.8), ("No", 96.2)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(government_job.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} works for a government agency"
        if x
        else f"{n} does not work for a government agency."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class is_therapist(BasePersonAttr):
    distribution = [("Yes", 0.12), ("No", 99.88)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(is_therapist.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is a physical therpaist licensed in New York State."
        if x
        else f"{n} is not a physical therapist licensed in New York State."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 18:
            return False
        return original_value


class contractor(BasePersonAttr):
    distribution = [("Yes", 6.9), ("No", 93.1)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(contractor.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is an independent contractor."
        if x
        else f"{n} is not an independent contractor."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class wep(BasePersonAttr):
    distribution = [("Yes", 3), ("No", 97)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(wep.distribution))
    default = False
    nl_fn = lambda n, x: (
        f"{n} is in the Work Experience Program."
        if x
        else f"{n} is not in the Work Experience Program."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class collective_bargaining(BasePersonAttr):
    distribution = [("Yes", 10.10), ("No", 89.9)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(collective_bargaining.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is subject to a collective bargaining agreement waiving safe and sick leave."
        if x
        else f"{n} is not subject to a collective bargaining agreement waiving safe and sick leave."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


### COVID-19 Funeral Assistance


class covid_funeral_expenses(BasePersonAttr):
    distribution = [("Yes", 1.13), ("No", 98.87)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(covid_funeral_expenses.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} incurred funeral expenses due to a covid death on or after January 20, 2020, and the death was attributed to COVID-19 on the death certificate."
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

    def random():
        if np.random.choice([True, False]):
            return 0
        else:
            return np.random.randint(1, 24)

    def demographic():
        if np.random.random() < 0.95:
            return 0
        else:
            return np.random.randint(1, 25)

    default = 0
    nl_fn = lambda n, x: (
        f"{n} was evicted {x} months ago." if x else f"{n} has never been evicted."
    )


class currently_being_evicted(BasePersonAttr):
    distribution = [("Yes", 5), ("No", 95)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(currently_being_evicted.distribution)
    )
    default = False
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
    distribution = [("Yes", 14), ("No", 86)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(employer_opt_in.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n}'s private employer has opted in to paid family leave."
        if x
        else f"{n}'s employer has not opted in to paid family leave."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return False
        if hh.members[person_idx]["age"] < 16:
            return False
        return original_value


class consecutive_work_weeks(BasePersonAttr):
    distribution = [
        ((0, 13), 4),
        ((14, 26), 5.2),
        ((27, 39), 3.7),
        ((40, 47), 4.9),
        ((48, 49), 1.6),
        ((50, 52), 80.6),
    ]
    schema = And(int)
    random = lambda: np.random.randint(0, 52)
    demographic = lambda: sample_from_distribution(consecutive_work_weeks.distribution)
    default = 0
    nl_fn = lambda n, x: (
        f"{n} has worked {x} consecutive weeks at their current employer."
        if x
        else f"{n} has not worked any consecutive weeks at their current employer."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return 0
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


class nonconsecutive_work_days(BasePersonAttr):
    distribution = [
        ((0, 0), 36.2),
        ((1, 21), 51.4),
        ((22, 35), 1),
        ((36, 91), 3.1),
        ((92, 182), 2.4),
        ((183, 273), 3.3),
        ((274, 365), 2.5),
    ]
    schema = And(int)
    random = lambda: np.random.randint(0, 365)
    demographic = lambda: sample_from_distribution(
        nonconsecutive_work_days.distribution
    )
    default = 0
    nl_fn = lambda n, x: (
        f"{n} has worked {x} nonconsecutive weeks at their current employer."
        if x
        else f"{n} has not worked any nonconsecutive weeks at their current employer."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["annual_work_income"] == 0:
            return 0
        if hh.members[person_idx]["age"] < 16:
            return 0
        return original_value


### Family Type Homes for Adults


class developmental_mental_day_treatment(BasePersonAttr):
    distribution = [("Yes", 34.19), ("No", 65.81)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(developmental_mental_day_treatment.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} attends a developmental mental day treatment program."
        if x
        else f"{n} does not attend a developmental mental day treatment program."
    )


class years_sober(BasePersonAttr):
    distribution = [((0, 1), 27), ((2, 5), 24), ((6, 10), 13), ((10, 16), 36)]
    schema = And(int)

    def random():
        r = np.random.randint(3)
        if r == 0:
            return 0
        elif r == 1:
            return -1
        else:
            return np.random.randint(1, 16)

    def demographic():
        return sample_from_distribution(years_sober.distribution)

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
    distribution = [("Yes", 12.3), ("No", 87.7)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(medication_treatment_non_compliance.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} has been non-compliant with medication and treatment."
        if x
        else f"{n} has always been compliant with medication and treatment."
    )


class arson(BasePersonAttr):
    distribution = [("Yes", 0.0109), ("No", 99.9891)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(arson.distribution))
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
    distribution = [("Yes", 0.147), ("No", 99.853)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(verbal_abuse.distribution)
    )
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
    distribution = [("Yes", 2.7), ("No", 97.3)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(imprisonment.distribution)
    )
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
    distribution = [("Yes", 24), ("No", 76)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(first_time_home_buyer.distribution)
    )
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
    distribution = [("Yes", 3.32), ("No", 96.68)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(honorable_service.distribution)
    )
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


class receives_medicaid(BasePersonAttr):
    distribution = [("Yes", 48.8), ("No", 51.2)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_medicaid.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Medicaid." if x else f"{n} does not receive Medicaid."
    )


class eligible_for_medicaid(BasePersonAttr):
    distribution = [("Yes", 22), ("No", 88)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(eligible_for_medicaid.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is eligible for Medicaid." if x else f"{n} is not eligible for Medicaid."
    )

    def conform(cls, hh, person_idx, original_value):
        if hh.members[person_idx]["receives_medicaid"]:
            return True
        return original_value


class receives_fpha(BasePersonAttr):
    distribution = [("Yes", 2.8), ("No", 97.2)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_fpha.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Federal Public Housing Assistance (FPHA)."
        if x
        else f"{n} does not receive Federal Public Housing Assistance (FPHA)."
    )

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


class receives_vpsb(BasePersonAttr):
    distribution = [("Yes", 0.35), ("No", 99.65)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(receives_vpsb.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} receives Veterans Pension and Survivor Benefits (VPSB)."
        if x
        else f"{n} does not receive Veterals Pension and Survivor Benefits (VPSB)."
    )


class eligible_for_hra_shelter(BasePersonAttr):
    distribution = [("Yes", 0.527), ("No", 99.473)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(eligible_for_hra_shelter.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is eligible for Health Reimbursement Arrangement (HRA) shelter."
        if x
        else f"{n} is not eligible for Health Reimbursement Arrangement (HRA) shelter."
    )

    def conform(cls, hh, person_idx, original_value):
        return hh.members[person_idx][cls.__name__]


class wheelchair(BasePersonAttr):
    distribution = [("Yes", 8.5), ("No", 91.5)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(
        sample_categorical(wheelchair.distribution)
    )
    default = False
    nl_fn = lambda n, x: (
        f"{n} is wheelchair bound." if x else f"{n} does not use a wheelchair."
    )

    def conform(cls, hh, person_idx, original_value):
        if not hh.members[person_idx]["disabled"]:
            return False
        return original_value


class bedridden(BasePersonAttr):
    distribution = [("Yes", 1.1), ("No", 98.9)]
    schema = And(bool)
    random = lambda: bool(np.random.choice([True, False]))
    demographic = lambda: yes_no_to_bool_map(sample_categorical(bedridden.distribution))
    default = False
    nl_fn = lambda n, x: (f"{n} is bedridden." if x else f"{n} is not bedridden.")

    def conform(cls, hh, person_idx, original_value):
        if not hh.members[person_idx]["disabled"]:
            return False
        return original_value


if __name__ == "__main__":
    # hh = get_random_household_input()
    print(sample_from_distribution(years_sober.distribution))
