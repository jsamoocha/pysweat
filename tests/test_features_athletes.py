import unittest
import pandas as pd
import numpy as np
from mock import patch
from math import sqrt
from pymongo import MongoClient
from pysweat.features.athletes import AthleteFeatures


class AthletesFeaturesTest(unittest.TestCase):
    @patch('pysweat.features.athletes.load_activities')
    def test_summary_stats_defaults(self, load_activities_mock):
        """Should return summary stats for average_speed measurement of rides"""
        load_activities_mock.return_value = pd.DataFrame({
            'athlete_id': [1, 2, 1],
            'average_speed': [25, 25, 27],
            'unused_measurement': [1, 1, 1]
        })
        athlete_df = pd.DataFrame({
            'athlete_id':[1, 2],
            'name': ['foo', 'bar']
        })
        athlete_features = AthleteFeatures(MongoClient())

        features_results = athlete_features.summary_stats(athlete_df)

        self.assertEqual(len(features_results), 2)
        self.assertItemsEqual(features_results.columns, ['athlete_id', 'name', 'ride_average_speed_mean',
                                                         'ride_average_speed_std', 'ride_count'])
        self.assertAlmostEqual(features_results.ride_average_speed_mean[0], 26, 9)
        self.assertAlmostEqual(features_results.ride_average_speed_mean[1], 25, 9)
        self.assertAlmostEqual(features_results.ride_average_speed_std[0], sqrt(2), 9)  # unbiased std
        self.assertTrue(np.isnan(features_results.ride_average_speed_std[1]))
        self.assertEqual(features_results.ride_count[0], 2)
        self.assertEqual(features_results.ride_count[1], 1)

    @patch('pysweat.features.athletes.load_activities')
    def test_summary_stats_custom_type_and_measurement(self, load_activities_mock):
        """Should return summary stats for given measurement of given activity type"""
        load_activities_mock.return_value = pd.DataFrame({
            'athlete_id': [1, 2, 1],
            'average_speed': [10, 13, 12],
            'heart_rate': [130, 140, 150]
        })
        athlete_df = pd.DataFrame({
            'athlete_id':[1, 2],
            'name': ['foo', 'bar']
        })
        mongo = MongoClient()
        athlete_features = AthleteFeatures(mongo)

        features_results = athlete_features.summary_stats(athlete_df, activity_type='Run',
                                                          activity_measurement='heart_rate')

        load_activities_mock.assert_called_with(mongo, type='Run')
        self.assertEqual(len(features_results), 2)
        self.assertItemsEqual(features_results.columns, ['athlete_id', 'name', 'run_heart_rate_mean',
                                                         'run_heart_rate_std', 'run_count'])
        self.assertAlmostEqual(features_results.run_heart_rate_mean[0], 140, 9)
        self.assertAlmostEqual(features_results.run_heart_rate_mean[1], 140, 9)
        self.assertAlmostEqual(features_results.run_heart_rate_std[0], sqrt(200), 9)  # unbiased std
        self.assertTrue(np.isnan(features_results.run_heart_rate_std[1]))
        self.assertEqual(features_results.run_count[0], 2)
        self.assertEqual(features_results.run_count[1], 1)
