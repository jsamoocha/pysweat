import unittest

import numpy as np
import pandas as pd

from pysweat.transformation.activities import weighted_average
from pysweat.transformation.general import get_observations_without_feature
from pysweat.transformation.windows import select_activity_window


# def compute_moving_averages(activity_df, to_be_computed):
#     activity_df.loc[to_be_computed, 'avg_speed_28'] = [
#         weighted_average(select_activity_window(activity_df, before, 28))
#         for before in activity_df.start_date_local[to_be_computed]
#         ]
#     return activity_df


class ActivityMovingAverageTransformationTest(unittest.TestCase):
    test_activities = pd.DataFrame().from_dict({
        'start_date_local': [np.datetime64(ts) for ts in pd.date_range(end='2015-05-01', periods=3).tolist()],
        'test_var': [1, 2, 3],
        'distance': [1, 1, 2],
        'average_speed': [18, 22, 12],
        'average_speed_28': [18, np.NaN, np.NaN]
    })

    mock_athletes = [
        {'id': 123},
        {'id': 456}
    ]

    def test_select_window_end_ts_and_window_size_within_data(self):
        """Should return dataframe with complete window"""
        selected_activities = select_activity_window(self.test_activities, self.test_activities.start_date_local[2], 2)

        self.assertEqual(2, len(selected_activities))
        self.assertEqual(2, selected_activities.test_var.values[0])
        self.assertEqual(3, selected_activities.test_var.values[1])

    def test_select_window_end_ts_after_last_activity_window_size_within_data(self):
        """Should return last activity"""
        selected_activities = select_activity_window(self.test_activities, pd.tslib.Timestamp('2015-05-02'), 2)

        self.assertEqual(1, len(selected_activities))
        self.assertEqual(3, selected_activities.test_var.values[0])

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

    # def test_compute_moving_averages(self):
