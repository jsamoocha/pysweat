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


    @patch('pysweat.features.activities.load_stream')
    def test_max_value_maintained_for_n_minutes(self, load_stream_mock):
        """Should return the maximum heartrate that was maintained for at least 5 minutes by default"""
        load_stream_mock.return_value = pd.DataFrame(
            {'heartrate': [100, 115, 120, 100, 110]},
            index=[180, 360, 540, 720, 900]
        )
        activity_df = pd.DataFrame(
            {'strava_id': [1, 2], 'distance': [1, 2]}
        )
        activity_features = ActivityFeatures(MongoClient())

        features_result = activity_features.max_value_maintained_for_n_minutes(activity_df)

        self.assertItemsEqual(list(features_result.columns.values),
                              ['strava_id', 'distance', 'max_heartrate_5_minutes'])
        self.assertEqual(len(features_result), 2)
        self.assertEqual(features_result.max_heartrate_5_minutes[0], 115)

    @patch('pysweat.features.activities.load_stream')
    def test_max_value_maintained_for_n_minutes_alternative_measure_and_window(self, load_stream_mock):
        """Should return the maximum value of the given measure that was maintained for the given number of
        minutes"""
        load_stream_mock.return_value = pd.DataFrame(
            {'power': [100, 115, 120, 100, 110]},
            index=[180, 360, 540, 720, 900]
        )
        activity_df = pd.DataFrame(
            {'strava_id': [1, 2], 'distance': [1, 2]}
        )
        activity_features = ActivityFeatures(MongoClient())

        features_result = activity_features.max_value_maintained_for_n_minutes(activity_df,
                                                                               measurement='power',
                                                                               window_size=8)

        self.assertItemsEqual(list(features_result.columns.values),
                              ['strava_id', 'distance', 'max_power_8_minutes'])
        self.assertEqual(len(features_result), 2)
        self.assertEqual(features_result.max_power_8_minutes[0], 100)
