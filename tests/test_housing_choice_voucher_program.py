import unittest
from unittest.mock import MagicMock
from users.benefits_programs import Section8HousingChoiceVoucherProgram

class TestSection8HousingChoiceVoucherProgram(unittest.TestCase):
    def setUp(self):
        # Import the class here or at top if available.
        # from your_module import Section8HousingChoiceVoucherProgram
        self.program = Section8HousingChoiceVoucherProgram()

    def test_household_one_member_within_limit(self):
        # 1-member household, income limit = 54350
        hh = MagicMock()
        hh.num_members.return_value = 1
        hh.hh_total_income.return_value = 50000
        self.assertTrue(self.program(hh))

    def test_household_one_member_exceeds_limit(self):
        # 1-member household, income > 54350
        hh = MagicMock()
        hh.num_members.return_value = 1
        hh.hh_total_income.return_value = 60000
        self.assertFalse(self.program(hh))

    def test_household_two_members_within_limit(self):
        # 2-member household, income limit = 62150
        hh = MagicMock()
        hh.num_members.return_value = 2
        hh.hh_total_income.return_value = 60000
        self.assertTrue(self.program(hh))

    def test_household_two_members_exceeds_limit(self):
        # 2-member household, income > 62150
        hh = MagicMock()
        hh.num_members.return_value = 2
        hh.hh_total_income.return_value = 70000
        self.assertFalse(self.program(hh))

    def test_household_eight_members_within_limit(self):
        # 8-member household, income limit = 102500
        hh = MagicMock()
        hh.num_members.return_value = 8
        hh.hh_total_income.return_value = 100000
        self.assertTrue(self.program(hh))

    def test_household_eight_members_exceeds_limit(self):
        # 8-member household, income > 102500
        hh = MagicMock()
        hh.num_members.return_value = 8
        hh.hh_total_income.return_value = 110000
        self.assertFalse(self.program(hh))

    def test_household_more_than_eight_members_within_limit(self):
        # 9-member household:
        # Base for 8 members = 102500
        # Difference between 7 and 8 member limit: 102500 - 96300 = 6300
        # For 9 members: limit = 102500 + (1 * 6300) = 108800
        hh = MagicMock()
        hh.num_members.return_value = 9
        hh.hh_total_income.return_value = 108000
        self.assertTrue(self.program(hh))

    def test_household_more_than_eight_members_exceeds_limit(self):
        # 9-member household:
        # limit = 108800 (from calculation above)
        hh = MagicMock()
        hh.num_members.return_value = 9
        hh.hh_total_income.return_value = 120000
        self.assertFalse(self.program(hh))

if __name__ == '__main__':
    unittest.main()
