import unittest
import numpy as np
import pandas as pd
import pysweat.transformation.streams as streams

class StreamTransformationTest(unittest.TestCase):
    def test_smooth(self):
        """Should smooth stream using centered moving average, for default window size and column name"""
        test_df = pd.DataFrame({'x': [1, 2, 3, 4]})

        transform_result = streams.smooth(test_df)

        self.assertEqual(list(transform_result.columns.values), ['x', 'x_smooth'])
        self.assertTrue(np.isnan(transform_result.x_smooth[0]))
        self.assertEqual(transform_result.x_smooth[1], 2.0)
        self.assertEqual(transform_result.x_smooth[2], 3.0)
        self.assertTrue(np.isnan(transform_result.x_smooth[3]))

    def test_smooth_alternative_window_size(self):
        """Should smooth stream with given window size"""
        test_df = pd.DataFrame({'x': [1, 2, 3]})

        transform_result = streams.smooth(test_df, window_size=1)

        self.assertEqual(list(transform_result.x_smooth.values), [1, 2, 3])

    def test_smooth_alternative_column_name(self):
        """Should smooth stream of given column name"""
        test_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

        transform_result = streams.smooth(test_df, smooth_colname='y')

        self.assertEqual(list(transform_result.columns.values), ['x', 'y', 'y_smooth'])
