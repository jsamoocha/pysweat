import unittest
import pandas as pd
from mock import patch
from pymongo import MongoClient
from pysweat.features.activities import ActivityFeatures


class ActivityFeaturesTest(unittest.TestCase):
    def test_turns_per_km(self):
        lat_long_stream_df = pd.DataFrame(
            {'latlng': [[52.1, 5.3], [52.2, 5.4], [58, 5.5], [52.4, 5.4], [52.5, 5.3]]},
            index=[1, 2, 3, 4, 5])

        total_turns_result = ActivityFeatures.sum_of_turns(lat_long_stream_df)
        self.assertAlmostEqual(0.12, total_turns_result, places=2)

    def test_turns_per_km_no_turns(self):
        lat_long_stream_df = pd.DataFrame(
            {'latlng': [[52.1, 5.3], [52.2, 5.3], [52.3, 5.3], [52.4, 5.3], [52.5, 5.3]]},
            index=[1, 2, 3, 4, 5])

        total_turns_result = ActivityFeatures.sum_of_turns(lat_long_stream_df)
        self.assertAlmostEqual(0.0, total_turns_result, places=2)

    def test_max_value_maintained_for_n_minutes(self):
        """Should return the maximum heartrate that was maintained for at least 5 minutes by default"""
        hr_stream_df = pd.DataFrame(
            {'heartrate': [100, 115, 120, 100, 110]},
            index=[180, 360, 540, 720, 900]
        )
        max_value_result = ActivityFeatures.max_value_maintained_for_n_minutes(hr_stream_df)

        self.assertEqual(115, max_value_result)

    def test_max_value_maintained_for_n_minutes_alternative_measure_and_window(self):
        """Should return the maximum value of the given measure that was maintained for the given number of
        minutes"""
        power_stream_df = pd.DataFrame(
            {'power': [100, 115, 120, 100, 110]},
            index=[180, 360, 540, 720, 900]
        )
        max_value_result = ActivityFeatures.max_value_maintained_for_n_minutes(power_stream_df, window_size=8)

        self.assertEqual(100, max_value_result)
