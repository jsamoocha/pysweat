import unittest
import pandas as pd
import numpy as np
from math import sqrt
from pysweat.features.athletes import summary_stats


class AthletesFeaturesTest(unittest.TestCase):
    def test_summary_stats_defaults(self):
        """Should return summary stats for average_speed measurement of provided activities"""
        athlete_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['foo', 'bar']
        })
        activity_df = pd.DataFrame({
            'athlete_id': [1, 2, 1],
            'average_speed': [25, 25, 27],
            'unused_measurement': [1, 1, 1],
            'type': ['Ride', 'Ride', 'Ride']
        })

        features_results = summary_stats(athlete_df, activity_df)

        self.assertEqual(len(features_results), 2)
        self.assertCountEqual(features_results.columns, ['id', 'name', 'ride_average_speed_mean',
                                                         'ride_average_speed_std', 'ride_count'])
        self.assertAlmostEqual(features_results.ride_average_speed_mean[0], 26, 9)
        self.assertAlmostEqual(features_results.ride_average_speed_mean[1], 25, 9)
        self.assertAlmostEqual(features_results.ride_average_speed_std[0], sqrt(2), 9)  # unbiased std
        self.assertTrue(np.isnan(features_results.ride_average_speed_std[1]))
        self.assertEqual(features_results.ride_count[0], 2)
        self.assertEqual(features_results.ride_count[1], 1)

    def test_summary_stats_custom_type_and_measurement(self):
        """Should return summary stats for given measurement of provided activities"""
        athlete_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['foo', 'bar']
        })
        activity_df = pd.DataFrame({
            'athlete_id': [1, 2, 1],
            'average_speed': [10, 13, 12],
            'heart_rate': [130, 140, 150],
            'type': ['Run', 'Run', 'Run']
        })

        features_results = summary_stats(athlete_df, activity_df, activity_measurement='heart_rate')

        self.assertEqual(len(features_results), 2)
        self.assertCountEqual(features_results.columns, ['id', 'name', 'run_heart_rate_mean',
                                                         'run_heart_rate_std', 'run_count'])
        self.assertAlmostEqual(features_results.run_heart_rate_mean[0], 140, 9)
        self.assertAlmostEqual(features_results.run_heart_rate_mean[1], 140, 9)
        self.assertAlmostEqual(features_results.run_heart_rate_std[0], sqrt(200), 9)  # unbiased std
        self.assertTrue(np.isnan(features_results.run_heart_rate_std[1]))
        self.assertEqual(features_results.run_count[0], 2)
        self.assertEqual(features_results.run_count[1], 1)

    def test_summary_stats_athlete_no_activities_for_type(self):
        """Should return nan for summary stats if athlete has no activities of the given type"""
        athlete_df = pd.DataFrame({
            'id': [1, 3],
            'name': ['foo', 'baz']
        })
        activity_df = pd.DataFrame({
            'athlete_id': [1, 2, 1],
            'average_speed': [10, 13, 12],
            'heart_rate': [130, 140, 150],
            'type': ['Run', 'Run', 'Run']
        })

        features_results = summary_stats(athlete_df, activity_df)

        self.assertEqual(len(features_results), 2)
        self.assertAlmostEqual(features_results.run_average_speed_mean[0], 11, 9)
        self.assertTrue(np.isnan(features_results.run_average_speed_mean[1]))
        self.assertTrue(np.isnan(features_results.run_count[1]))

    def test_summary_stats_inconsistent_activities_provided(self):
        """Should raise ValueError in case of activities with multiple types"""
        athlete_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['foo', 'bar']
        })
        activity_df = pd.DataFrame({
            'athlete_id': [1, 2, 1],
            'average_speed': [25, 25, 12],
            'unused_measurement': [1, 1, 1],
            'type': ['Ride', 'Ride', 'Run']  # Mixture of rides and runs
        })

        self.assertRaises(ValueError, summary_stats, athlete_df, activity_df)
