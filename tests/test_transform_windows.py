import unittest
from pysweat.transformation.windows import subtract_n_minutes

class WindowsTransformTest(unittest.TestCase):
    def test_subtract_5_minutes(self):
        """Should default to 5 minutes"""
        self.assertEqual(subtract_n_minutes(423), 123)

    def test_subtract_5_minutes_negative_result(self):
        """Should not return negative values, but default to a minimum of 0"""
        self.assertEqual(subtract_n_minutes(99), 0)

    def test_subtract_5_minutes_minimum_value(self):
        """Should return the provided minimum value in case of negative result"""
        self.assertEqual(subtract_n_minutes(99, minimum_value=50), 50)

    def test_subtract_1_minute(self):
        """Should return result for a provided amount of minutes to be subtracted"""
        self.assertEqual(subtract_n_minutes(423, 1), 363)
