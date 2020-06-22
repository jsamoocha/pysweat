import unittest
import pandas as pd
from pysweat.transformation.general import delta_constant


class GeneralTransformationTest(unittest.TestCase):
    def test_delta_constant(self):
        """Should return x - constant by default"""
        test_df = pd.DataFrame({'x': [1, 2, 3]})

        transform_result_df = delta_constant(test_df, 2, 'mean')

        self.assertCountEqual(transform_result_df.columns, ['x', 'd_x_mean'])
        self.assertEqual(list(transform_result_df.d_x_mean.values), [-1, 0, 1])