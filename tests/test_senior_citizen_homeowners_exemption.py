import unittest
from unittest.mock import MagicMock
from users.benefits_programs import SeniorCitizenHomeownersExemption

# If your program class is located elsewhere, make sure to import it properly:
# from path.to.your.module import SeniorCitizenHomeownersExemption

class TestSeniorCitizenHomeownersExemption(unittest.TestCase):
    def setUp(self):
        # Instantiate the program we want to test
        self.program = SeniorCitizenHomeownersExemption()

    def test_ineligible_housing_type(self):
        """
        Test that the program returns False for an invalid housing type.
        """
        hh_mock = MagicMock()
        # Mock household to return an invalid housing type
        hh_mock.get_housing_type.return_value = "random_non_valid"

        # No owners
        hh_mock.property_owners.return_value = []

        result = self.program(hh_mock)
        self.assertFalse(result, "Should be False if the household's housing type is invalid.")

    def test_no_property_owners(self):
        """
        Test that the program returns False if there are no property owners.
        """
        hh_mock = MagicMock()
        hh_mock.get_housing_type.return_value = "one_family_home"
        hh_mock.property_owners.return_value = []  # no owners
        result = self.program(hh_mock)
        self.assertFalse(result, "Should be False if no members own the property.")

    def test_single_owner_under_65(self):
        """
        Single owner who is under 65 => ineligible.
        """
        hh_mock = MagicMock()
        hh_mock.get_housing_type.return_value = "one_family_home"

        # Mock a single owner < 65
        person_mock = MagicMock()
        person_mock.get.side_effect = lambda key, default=None: {
            "age": 60,               # < 65
            "primary_residence": True,
            "had_previous_sche": False,
            "months_owned_property": 24
        }.get(key, default)

        hh_mock.property_owners.return_value = [person_mock]
        # Combined income check
        hh_mock.owners_total_income.return_value = 0

        result = self.program(hh_mock)
        self.assertFalse(result, "Should be False if the single owner is under 65.")

    def test_single_owner_age_65(self):
        """
        Single owner who is exactly 65 => eligible assuming other conditions are met.
        """
        hh_mock = MagicMock()
        hh_mock.get_housing_type.return_value = "one_family_home"

        # Mock a single owner = 65
        person_mock = MagicMock()
        person_mock.get.side_effect = lambda key, default=None: {
            "age": 65,
            "primary_residence": True,
            "had_previous_sche": False,
            "months_owned_property": 24
        }.get(key, default)

        hh_mock.property_owners.return_value = [person_mock]
        # Combined income within allowed limit
        hh_mock.owners_total_income.return_value = 58000

        result = self.program(hh_mock)
        self.assertTrue(result, "Should be True if single owner is 65+ with valid conditions.")

    def test_two_owners_spouses_only_one_65(self):
        """
        Two owners, are spouses or siblings; only one needs to be 65+.
        """
        hh_mock = MagicMock()
        hh_mock.get_housing_type.return_value = "two_family_home"

        # Mock two owners
        owner1 = MagicMock()
        owner2 = MagicMock()

        def owner1_get(key, default=None):
            data = {
                "age": 70,  # 70 => meets 65+ requirement
                "relation": "spouse", 
                "primary_residence": True,
                "months_owned_property": 24,
                "had_previous_sche": False
            }
            return data.get(key, default)

        def owner2_get(key, default=None):
            data = {
                "age": 60,
                "relation": "spouse",
                "primary_residence": True,
                "months_owned_property": 24,
                "had_previous_sche": False
            }
            return data.get(key, default)

        owner1.get.side_effect = owner1_get
        owner2.get.side_effect = owner2_get

        hh_mock.property_owners.return_value = [owner1, owner2]
        # Combined income
        hh_mock.owners_total_income.return_value = 50000

        # The program checks if they're spouses/siblings (which we simulate by storing "spouse" in the relation).
        result = self.program(hh_mock)
        self.assertTrue(result, "Should be True if at least one spouse is 65+.")

    def test_two_owners_not_spouses_both_must_be_65(self):
        """
        Two owners, but not spouses or siblings => both must be 65+.
        Here, one is under 65 => ineligible.
        """
        hh_mock = MagicMock()
        hh_mock.get_housing_type.return_value = "one_family_home"

        owner1 = MagicMock()
        owner2 = MagicMock()
        
        owner1.get.side_effect = lambda key, default=None: {
            "age": 70,
            "relation": "other_family",  # Not spouse or sibling
            "primary_residence": True,
            "months_owned_property": 24,
            "had_previous_sche": False
        }.get(key, default)
        
        owner2.get.side_effect = lambda key, default=None: {
            "age": 60,
            "relation": "other_family",  # Not spouse or sibling
            "primary_residence": True,
            "months_owned_property": 24,
            "had_previous_sche": False
        }.get(key, default)

        hh_mock.property_owners.return_value = [owner1, owner2]
        hh_mock.owners_total_income.return_value = 50000

        result = self.program(hh_mock)
        self.assertFalse(result, "Should be False if two non-spouse owners are not both 65+.")

    def test_more_than_two_owners_all_must_be_65(self):
        """
        If 3 or more owners, all must be 65 or older.
        """
        hh_mock = MagicMock()
        hh_mock.get_housing_type.return_value = "three_family_home"

        # Mock three owners
        owner_ages = [65, 70, 63]  # One is 63 => ineligible
        owners_mocks = []
        for age in owner_ages:
            o = MagicMock()
            o.get.side_effect = lambda k, d=None, age=age: {
                "age": age,
                "relation": "other_family",
                "primary_residence": True,
                "months_owned_property": 24,
                "had_previous_sche": False
            }.get(k, d)
            owners_mocks.append(o)

        hh_mock.property_owners.return_value = owners_mocks
        hh_mock.owners_total_income.return_value = 40000

        result = self.program(hh_mock)
        self.assertFalse(result, "Should be False if any one of the 3+ owners is under 65.")

    def test_combined_income_exceeds_limit(self):
        """
        Combined income over $58,399 => ineligible.
        """
        hh_mock = MagicMock()
        hh_mock.get_housing_type.return_value = "one_family_home"

        # Single eligible owner but combined income is too high
        owner = MagicMock()
        owner.get.side_effect = lambda k, d=None: {
            "age": 70,
            "relation": "self",
            "primary_residence": True,
            "months_owned_property": 24,
            "had_previous_sche": False
        }.get(k, d)

        hh_mock.property_owners.return_value = [owner]
        # Over the limit
        hh_mock.owners_total_income.return_value = 60000

        result = self.program(hh_mock)
        self.assertFalse(result, "Should be False if combined income exceeds $58,399.")

    def test_ownership_duration_requirement(self):
        """
        Each owner must own for >=12 months unless they had SCHE before.
        """
        hh_mock = MagicMock()
        hh_mock.get_housing_type.return_value = "coop"

        # Two owners, both 65+, incomes within range
        # But one hasn't owned for >=12 months and didn't have SCHE previously => fails
        owner1 = MagicMock()
        owner2 = MagicMock()

        owner1.get.side_effect = lambda k, d=None: {
            "age": 70,
            "relation": "spouse",
            "primary_residence": True,
            "months_owned_property": 10,   # < 12 months
            "had_previous_sche": False
        }.get(k, d)

        owner2.get.side_effect = lambda k, d=None: {
            "age": 66,
            "relation": "spouse",
            "primary_residence": True,
            "months_owned_property": 24,
            "had_previous_sche": False
        }.get(k, d)

        hh_mock.property_owners.return_value = [owner1, owner2]
        hh_mock.owners_total_income.return_value = 40000

        result = self.program(hh_mock)
        self.assertFalse(result, "Should be False if any owner has fewer than 12 months of ownership and no prior SCHE.")

    def test_valid_sche_all_criteria_met(self):
        """
        A scenario where all requirements are satisfied => eligible.
        """
        hh_mock = MagicMock()
        hh_mock.get_housing_type.return_value = "two_family_home"

        # Two sibling owners, at least one is 65, all have valid ownership durations, 
        # combined income under limit, both live at the property.
        owner1 = MagicMock()
        owner2 = MagicMock()

        owner1.get.side_effect = lambda k, d=None: {
            "age": 70,
            "relation": "sibling",
            "primary_residence": True,
            "months_owned_property": 24,
            "had_previous_sche": False
        }.get(k, d)

        owner2.get.side_effect = lambda k, d=None: {
            "age": 64,
            "relation": "sibling",
            "primary_residence": True,
            "months_owned_property": 24,
            "had_previous_sche": False
        }.get(k, d)

        hh_mock.property_owners.return_value = [owner1, owner2]
        hh_mock.owners_total_income.return_value = 40000

        result = self.program(hh_mock)
        self.assertTrue(result, "Should be True if siblings own the property, at least one is 65+, and other conditions are met.")


# If you want to run this as a script:
if __name__ == '__main__':
    unittest.main()
