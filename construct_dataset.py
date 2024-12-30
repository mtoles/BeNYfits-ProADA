from analysis.dataset_constructor import DatasetConstructor
from users.benefits_programs import ChildAndDependentCareTaxCredit, EarlyHeadStartPrograms, InfantToddlerPrograms, ComprehensiveAfterSchool, InfantToddlerPrograms, ChildTaxCredit, DisabilityRentIncreaseExemption, EarnedIncomeTaxCredit, HeadStart, get_random_household_input

for _ in range(1000):
    for class_name in [ChildAndDependentCareTaxCredit]:
        count = 0
        while True:
            hh = get_random_household_input()
            count += 1

            print(DatasetConstructor._trace_execution(class_name.__call__, hh))

            if class_name.__call__(hh) == True:
                print("Done", count)
                break