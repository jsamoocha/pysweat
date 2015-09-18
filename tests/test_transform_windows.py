import unittest
from pysweat.transformation.windows import subtract_n_minutes

class WindowsTransformTest(unittest.TestCase):
    def test_subtract_5_minutes(self):
        self.assertEqual(subtract_n_minutes(423), 123)

    def test_subtract_5_minutes_negative_result(self):
        self.assertEqual(subtract_n_minutes(99), 0)

    def test_subtract_5_minutes_minimum_value(self):
        self.assertEqual(subtract_n_minutes(99, minimum_value=50), 50)

    def test_subtract_1_minute(self):
        self.assertEqual(subtract_n_minutes(423, 1), 363)
