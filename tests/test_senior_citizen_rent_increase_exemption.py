import unittest
from unittest.mock import MagicMock
from users.benefits_programs import SeniorCitizenRentIncreaseExemption

class TestSeniorCitizenRentIncreaseExemption(unittest.TestCase):
    def setUp(self):
        """Instantiate the SCRIE program."""
        self.scrie = SeniorCitizenRentIncreaseExemption()

    def _make_user_mock(self, age=62, name_on_lease=True, place_of_residence="NYC",
                        rent_stab=False, rent_ctrl=False, rent_hotel=False, rent_sro=False,
                        mitchell_lama=False, limited_div=False, redevel=False, hdfc=False):
        """
        Utility to create a MagicMock 'user' with relevant attributes. 
        This helps keep tests concise.
        """
        user_mock = MagicMock()
        
        def user_get_side_effect(key, default=None):
            data = {
                "age": age,
                "name_is_on_lease": name_on_lease,
                "place_of_residence": place_of_residence,
                "lives_in_rent_stabilized_apartment": rent_stab,
                "lives_in_rent_controlled_apartment": rent_ctrl,
                "lives_in_rent_regulated_hotel": rent_hotel,
                "lives_in_rent_regulated_single": rent_sro,
                "lives_in_mitchell_lama": mitchell_lama,
                "lives_in_limited_dividend_development": limited_div,
                "lives_in_redevelopment_company_development": redevel,
                "lives_in_hdfc_development": hdfc,
            }
            return data.get(key, default)
        
        user_mock.get.side_effect = user_get_side_effect
        return user_mock

    def _make_household_mock(self, user_mock, hh_income, hh_rent):
        """
        Utility to create a MagicMock 'household' with total income and monthly rent
        plus the user mock.
        """
        hh_mock = MagicMock()
        hh_mock.user.return_value = user_mock
        hh_mock.hh_total_income.return_value = hh_income
        hh_mock.hh_monthly_rent_spending.return_value = hh_rent
        return hh_mock

    # ---------------------------
    # Test: All criteria met
    # ---------------------------
    def test_eligible_all_criteria_met(self):
        """
        A scenario where:
         - User is >= 62
         - Name on lease
         - Combined HH income <= 50,000
         - Rent > 1/3 of monthly income
         - Lives in NYC in an eligible property type
        => Should be True
        """
        user_mock = self._make_user_mock(
            age=70, 
            name_on_lease=True, 
            place_of_residence="NYC", 
            rent_stab=True  # rent stabilized apt
        )
        hh_mock = self._make_household_mock(
            user_mock=user_mock,
            hh_income=48000,      # monthly ~ 4000
            hh_rent=1800          # 1800 > 1/3 of 4000 => True
        )

        result = self.scrie(hh_mock)
        self.assertTrue(result, "Expected True because all SCRIE criteria are satisfied.")

    # ---------------------------
    # Test: Ineligible due to age < 62
    # ---------------------------
    def test_ineligible_due_to_age(self):
        """
        User is under 62 => fails immediately, even if everything else meets criteria.
        """
        user_mock = self._make_user_mock(
            age=60,      # under 62
            name_on_lease=True, 
            place_of_residence="NYC", 
            rent_stab=True
        )
        hh_mock = self._make_household_mock(
            user_mock=user_mock,
            hh_income=40000,
            hh_rent=2000
        )

        result = self.scrie(hh_mock)
        self.assertFalse(result, "Expected False because the applicant is under 62.")

    # ---------------------------
    # Test: Ineligible due to not on lease
    # ---------------------------
    def test_ineligible_due_to_not_on_lease(self):
        """
        User is >= 62 but 'name_is_on_lease' is False => ineligible.
        """
        user_mock = self._make_user_mock(
            age=65,
            name_on_lease=False, 
            place_of_residence="NYC",
            rent_stab=True
        )
        hh_mock = self._make_household_mock(
            user_mock=user_mock,
            hh_income=30000,
            hh_rent=1500
        )

        result = self.scrie(hh_mock)
        self.assertFalse(result, "Expected False because user isn't on lease.")

    # ---------------------------
    # Test: Ineligible due to high income
    # ---------------------------
    def test_ineligible_due_to_income(self):
        """
        Household income is > 50,000 => ineligible.
        """
        user_mock = self._make_user_mock(
            age=70,
            name_on_lease=True,
            place_of_residence="NYC",
            rent_stab=True
        )
        hh_mock = self._make_household_mock(
            user_mock=user_mock,
            hh_income=51000,  # just above limit
            hh_rent=2000
        )

        result = self.scrie(hh_mock)
        self.assertFalse(result, "Expected False because HH income is > 50,000.")

    # ---------------------------
    # Test: Ineligible due to rent fraction (rent <= 1/3 of monthly income)
    # ---------------------------
    def test_ineligible_due_to_rent_fraction(self):
        """
        Must spend more than 1/3 of monthly income on rent. 
        If user only spends exactly or less => ineligible.
        """
        user_mock = self._make_user_mock(
            age=70,
            name_on_lease=True,
            place_of_residence="NYC",
            rent_stab=True
        )
        # HH income = 48000 => monthly ~ 4000 => 1/3 of 4000 = 1333.33
        # If rent <= 1333 => fail
        hh_mock = self._make_household_mock(
            user_mock=user_mock,
            hh_income=48000,  
            hh_rent=1300       # 1300 < 1/3 of 4000
        )

        result = self.scrie(hh_mock)
        self.assertFalse(result, "Expected False because rent is not > 1/3 of monthly income.")

    # ---------------------------
    # Test: Ineligible due to not living in NYC
    # ---------------------------
    def test_ineligible_due_to_not_in_nyc(self):
        """
        If place_of_residence != "NYC", ineligible regardless of other conditions.
        """
        user_mock = self._make_user_mock(
            age=70,
            name_on_lease=True,
            place_of_residence="Jersey",  # Not NYC
            rent_stab=True
        )
        hh_mock = self._make_household_mock(
            user_mock=user_mock,
            hh_income=30000,
            hh_rent=1600
        )

        result = self.scrie(hh_mock)
        self.assertFalse(result, "Expected False because user does not live in NYC.")

    # ---------------------------
    # Test: Ineligible due to no eligible housing type
    # ---------------------------
    def test_ineligible_due_to_housing_type(self):
        """
        Must live in rent stabilized, rent controlled, rent regulated hotel/SRO, 
        Mitchell-Lama, limited dividend, redevelopment, or HDFC. 
        If none are True => ineligible.
        """
        user_mock = self._make_user_mock(
            age=70,
            name_on_lease=True,
            place_of_residence="NYC",
            rent_stab=False,
            rent_ctrl=False,
            rent_hotel=False,
            rent_sro=False,
            mitchell_lama=False,
            limited_div=False,
            redevel=False,
            hdfc=False
        )
        hh_mock = self._make_household_mock(
            user_mock=user_mock,
            hh_income=30000,
            hh_rent=1600
        )

        result = self.scrie(hh_mock)
        self.assertFalse(result, "Expected False because the user is not in an eligible housing type.")

    # ---------------------------
    # Test: Edge case: rent == 1/3 of monthly income
    # ---------------------------
    def test_ineligible_rent_exactly_one_third(self):
        """
        The requirement is strictly more than one-third.
        If rent = exactly 1/3, that fails the > condition.
        """
        user_mock = self._make_user_mock(
            age=70,
            name_on_lease=True,
            place_of_residence="NYC",
            rent_stab=True
        )
        # HH income = 48000 => monthly = 4000 => 1/3 = 1333
        hh_mock = self._make_household_mock(
            user_mock=user_mock,
            hh_income=48000,
            hh_rent=1333  # exactly 1/3
        )

        result = self.scrie(hh_mock)
        self.assertFalse(result, "Expected False because rent is not strictly > 1/3 of monthly income.")

    # ---------------------------
    # Test: Different eligible housing type (Mitchell-Lama)
    # ---------------------------
    def test_eligible_mitchell_lama(self):
        """
        Another scenario: user is in Mitchell-Lama, 66 years old, 
        lease in their name, in NYC, HH income < 50k, rent fraction > 1/3
        => should pass.
        """
        user_mock = self._make_user_mock(
            age=66,
            name_on_lease=True,
            place_of_residence="NYC",
            mitchell_lama=True
        )
        # Income = 30000 => monthly=2500 => 1/3=833 => rent=900 => 900>833
        hh_mock = self._make_household_mock(
            user_mock=user_mock,
            hh_income=30000,
            hh_rent=900
        )

        result = self.scrie(hh_mock)
        self.assertTrue(result, "Expected True for Mitchell-Lama occupant meeting all conditions.")


# If you want to run these tests directly:
if __name__ == '__main__':
    unittest.main()
