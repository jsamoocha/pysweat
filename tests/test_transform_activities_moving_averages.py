import unittest

import numpy as np
import pandas as pd

from pysweat.transformation.activities import weighted_average, compute_moving_averages
from pysweat.transformation.general import get_observations_without_feature
from pysweat.transformation.windows import select_activity_window


test_activities = pd.DataFrame().from_dict({
    'start_date_local': [np.datetime64(ts) for ts in pd.date_range(end='2015-05-01', periods=3).tolist()],
    'test_var': [1, 2, 3.5],
    'distance': [1, 1, 2],
    'average_speed': [18, 22, 12],
    'average_speed_28': [18, np.NaN, np.NaN]
})


class ActivityMovingAverageTransformationTest(unittest.TestCase):
    mock_athletes = [
        {'id': 123},
        {'id': 456}
    ]

    def setUp(self):
        self.test_activities = pd.DataFrame.copy(test_activities)

    def test_select_window_end_ts_and_window_size_within_data(self):
        """Should return dataframe with complete window"""
        selected_activities = select_activity_window(self.test_activities, self.test_activities.start_date_local[2], 2)

        self.assertEqual(2, len(selected_activities))
        self.assertEqual(2, selected_activities.test_var.values[0])
        self.assertEqual(3.5, selected_activities.test_var.values[1])

    def test_select_window_end_ts_after_last_activity_window_size_within_data(self):
        """Should return last activity"""
        selected_activities = select_activity_window(self.test_activities, pd.tslib.Timestamp('2015-05-02'), 2)

        self.assertEqual(1, len(selected_activities))
        self.assertEqual(3.5, selected_activities.test_var.values[0])

    def test_select_window_end_ts_after_last_activity_window_size_outside_data(self):
        """Should return empty"""
        selected_activities = select_activity_window(self.test_activities, pd.tslib.Timestamp('2015-05-05'), 2)

        self.assertEqual(0, len(selected_activities))

    def test_select_window_end_ts_before_last_activity_window_size_outside_data(self):
        """Should return first activity"""
        selected_activities = select_activity_window(self.test_activities, pd.tslib.Timestamp('2015-04-29'), 2)

        self.assertEqual(1, len(selected_activities))
        self.assertEqual(1, selected_activities.test_var.values[0])

    def test_weighted_average(self):
        """Should return average speed weighted by distance"""
        self.assertEqual(16, weighted_average(self.test_activities, feature='average_speed', weight_feature='distance'))

    def test_get_activities_without_feature_all_activities_with_feature(self):
        """Should return all-false boolean index if all activities have the feature"""
        self.assertEqual([False, False, False],
                         list(get_observations_without_feature(self.test_activities, 'average_speed')))

    def test_get_activities_without_feature_no_activities_with_feature(self):
        """Should return all-true boolean index if no activities have the feature"""
        self.assertEqual([True, True, True],
                         list(get_observations_without_feature(self.test_activities, 'non_existing_feature')))

    def test_get_activities_without_feature_first_activity_has_feature(self):
        """Should return all-true boolean index except for first activity that has the feature"""
        self.assertEqual([False, True, True],
                         list(get_observations_without_feature(self.test_activities, 'average_speed_28')))

    def test_compute_moving_averages_retains_original_data(self):
        """Should compute moving average for given feature retaining existing features and observations"""
        self.assertEqual(3, len(compute_moving_averages(self.test_activities, feature_name='test_var', window_days=2)))
        self.assertEqual(len(test_activities.columns) + 1,
                         len(compute_moving_averages(
                                 self.test_activities, feature_name='test_var', window_days=2).columns))

    def test_compute_moving_averages_no_new_column_for_existing_moving_averages(self):
        """Should not add new column if one or more moving averages were computed for the given feature"""
        self.assertEqual(len(test_activities.columns),
                         len(compute_moving_averages(
                                 self.test_activities, feature_name='average_speed', window_days=28).columns))

    def test_compute_moving_averages_adds_column_for_given_feature(self):
        """Should create new column with name [original_feature_name]_[window_size] as name"""
        self.assertIn('test_var_3',
                      compute_moving_averages(self.test_activities, feature_name='test_var', window_days=3).columns)

    def test_compute_moving_averages_computes_moving_averages(self):
        """Should compute moving averages for given feature and window"""
        self.assertEqual([1, 1.5, 3],
                         list(compute_moving_averages(self.test_activities,
                                                      feature_name='test_var',
                                                      window_days=2).test_var_2))
