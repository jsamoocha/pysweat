import unittest
import pandas as pd
import numpy as np
import pysweat.transformation.gps as gps


class GPSTransformationTest(unittest.TestCase):
    def test_latlong_to_x_y(self):
        """Should convert latlng column to separate x and y columns, using Equirectangular projection"""
        latlong_df = pd.DataFrame({'latlng': [[52.1, 5.3], [52.2, 5.4], [52.3, 5.5]]}, index=[1, 2, 3])

        transform_result = gps.lat_long_to_x_y(latlong_df)

        self.assertEqual(list(transform_result.columns.values), ['latlng', 'x', 'y'])
        np.testing.assert_almost_equal(list(transform_result.x.values), [0.057, 0.058, 0.059], 3)
        np.testing.assert_almost_equal(list(transform_result.y.values), [0.909, 0.911, 0.913], 3)

    def test_latlong_to_x_y_alternative_latlong_col_name(self):
        """Should use alternative column name for lat-long"""
        latlong_df = pd.DataFrame({'alt_latlng': [[52.1, 5.3], [52.2, 5.4], [52.3, 5.5]]}, index=[1, 2, 3])

        transform_result = gps.lat_long_to_x_y(latlong_df, lat_long_colname='alt_latlng')

        self.assertEqual(list(transform_result.columns.values), ['alt_latlng', 'x', 'y'])
        np.testing.assert_almost_equal(list(transform_result.x.values), [0.057, 0.058, 0.059], 3)
        np.testing.assert_almost_equal(list(transform_result.y.values), [0.909, 0.911, 0.913], 3)

    def test_latlong_to_x_y_latlong_column_unavailable(self):
        """Should raise exception if lat long column is unavailable"""
        foobar_df = pd.DataFrame({'foo': [1, 2, 3]}, index=[1, 2, 3])

        self.assertRaises(KeyError, gps.lat_long_to_x_y, foobar_df)
