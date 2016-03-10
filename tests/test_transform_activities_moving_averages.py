import unittest

import numpy as np
import pandas as pd

from pysweat.transformation.windows import select_activity_window


def weighted_average(activities):
    return sum(activities.distance * activities.average_speed) / sum(activities.distance)


class ActivityMovingAverageTransformationTest(unittest.TestCase):
    test_activities = pd.DataFrame().from_dict({
        'start_date_local': [np.datetime64(ts) for ts in pd.date_range(end='2015-05-01', periods=3).tolist()],
        'test_var': [1, 2, 3],
        'distance': [1, 1, 2],
        'average_speed': [18, 22, 12]
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
        self.assertEqual(16, weighted_average(self.test_activities))