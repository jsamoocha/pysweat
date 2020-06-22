import unittest

import arrow
import numpy as np
import pandas as pd
import pysweat.transformation.streams as streams
from pysweat.transformation.similarities import cosine_similarity


class StreamTransformationTest(unittest.TestCase):
    def test_smooth(self):
        """Should smooth stream using moving average, for default window size and column name"""
        test_df = pd.DataFrame({'x': [1, 2, 3, 4]})

        transform_result = streams.smooth(test_df)

        self.assertEqual(list(transform_result.columns.values), ['x', 'x_smooth'])
        self.assertTrue(np.isnan(transform_result.x_smooth[0]))
        self.assertTrue(np.isnan(transform_result.x_smooth[1]))
        self.assertEqual(transform_result.x_smooth[2], 2.0)
        self.assertEqual(transform_result.x_smooth[3], 3.0)

    def test_smooth_multiple_columns(self):
        """Should smooth stream using moving average, for default window size and column name"""
        test_df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [11, 12, 13, 14]})

        transform_result = streams.smooth(test_df, smooth_colnames=['x', 'y'])

        self.assertCountEqual(['x', 'x_smooth', 'y', 'y_smooth'], transform_result.columns)
        self.assertEqual(transform_result.x_smooth[2], 2.0)
        self.assertEqual(transform_result.y_smooth[3], 13.0)

        transform_result = streams.smooth(test_df, smooth_colnames=['y'])

        self.assertCountEqual(['x', 'y', 'y_smooth'], transform_result.columns)

    def test_smooth_time_based_index(self):
        """Should smooth stream using moving average, with dynamic window size depending on the index"""
        test_df = pd.DataFrame({'x': [1, 2, 3, 4]}, index=[1, 4, 5, 7])
        transform_result = streams.smooth(test_df, window_size=2, use_index=True)

        self.assertEqual(transform_result.x_smooth[1], 1.0)
        self.assertEqual(transform_result.x_smooth[4], 2.0)
        self.assertEqual(transform_result.x_smooth[5], 2.5)
        self.assertEqual(transform_result.x_smooth[7], 4.0)

    def test_smooth_alternative_window_size(self):
        """Should smooth stream with given window size"""
        test_df = pd.DataFrame({'x': [1, 2, 3]})

        transform_result = streams.smooth(test_df, window_size=1)

        self.assertEqual(list(transform_result.x_smooth.values), [1, 2, 3])

    def test_derivative(self):
        """Should compute derivative for all columns by default"""
        test_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 7]})

        transform_result = streams.derivative(test_df)

        self.assertCountEqual(['x', 'y', 'dx_dt', 'dy_dt'], transform_result.columns)
        self.assertTrue(np.isnan(transform_result.dx_dt[0]))
        self.assertEqual(transform_result.dx_dt[1], 1.0)
        self.assertEqual(transform_result.dy_dt[2], 2.0)

    def test_derivative_multiple_column_names(self):
        """Should compute derivative of given column name"""
        test_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 6, 8]})

        transform_result = streams.derivative(test_df, derivative_colnames=['x', 'y'])

        self.assertCountEqual(['x', 'y', 'dx_dt', 'dy_dt'], transform_result.columns)
        self.assertTrue(np.isnan(transform_result.dy_dt[0]))
        self.assertEqual(transform_result.dx_dt[1], 1.0)
        self.assertEqual(transform_result.dy_dt[2], 2.0)

        transform_result = streams.derivative(test_df, derivative_colnames=['y'])

        self.assertCountEqual(['x', 'y', 'dy_dt'], transform_result.columns)

    def test_derivative_nonlinear_index(self):
        """Should take into account dt in index when computing derivative"""
        test_df = pd.DataFrame({'x': [1, 2, 3]}, index=[1, 2, 4])

        transform_result = streams.derivative(test_df)

        self.assertEqual(list(transform_result.columns.values), ['x', 'dx_dt'])
        self.assertTrue(np.isnan(transform_result.dx_dt[1]))
        self.assertEqual(transform_result.dx_dt[2], 1.0)
        self.assertEqual(transform_result.dx_dt[4], 0.5)

    def test_rolling_similarity(self):
        """Should compute similarity between the sequence of vectors defined by the given columns, and given
        similarity function"""
        test_df = pd.DataFrame({'dx_dt': [1, 1, 1], 'dy_dt': [1, 1, -1]})

        transform_result = streams.rolling_similarity(test_df, cosine_similarity, 'dx_dt', 'dy_dt')

        self.assertEqual(list(transform_result.columns.values), ['dx_dt', 'dy_dt', 'cosine_similarity_dx_dt_dy_dt'])
        self.assertTrue(np.isnan(transform_result.cosine_similarity_dx_dt_dy_dt[0]))
        self.assertAlmostEqual(transform_result.cosine_similarity_dx_dt_dy_dt[1], 1.0, 9)
        self.assertAlmostEqual(transform_result.cosine_similarity_dx_dt_dy_dt[2], 0, 9)
