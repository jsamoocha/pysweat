import unittest

import pandas as pd

from pysweat.features.activities import ActivityFeatures, _moving_sum_filter


class ActivityFeaturesTest(unittest.TestCase):
    def test_moving_sum_filter_fixed_window_size(self):
        test_series = pd.Series([1, 0, 1, 1, 2, 1, 0, 0], index=[1, 2, 5, 8, 9, 11, 12, 13])

        filtered_series = _moving_sum_filter(test_series)  # default threshold of 0 - does not filter
        self.assertListEqual([1, 0, 1, 1, 2, 1, 0, 0], filtered_series.values.tolist())

        filtered_series = _moving_sum_filter(test_series, threshold=2)  # default window size of 3
        self.assertListEqual([0, 0, 1, 1, 2, 1, 0, 0], filtered_series.values.tolist())

        filtered_series = _moving_sum_filter(test_series, threshold=2, window_size=2)
        self.assertListEqual([0, 0, 0, 1, 2, 1, 0, 0], filtered_series.values.tolist())

    def test_moving_sum_filter_time_based_index(self):
        test_series = pd.Series([1, 0, 1, 1, 2, 1, 0, 0], index=[1, 2, 5, 8, 9, 11, 12, 13])

        filtered_series = _moving_sum_filter(test_series, use_index=True)  # default threshold of 0 - does not filter
        self.assertListEqual([1, 0, 1, 1, 2, 1, 0, 0], filtered_series.values.tolist())

        filtered_series = _moving_sum_filter(test_series, threshold=2, use_index=True)  # default window size of 3
        self.assertListEqual([0, 0, 0, 1, 2, 1, 0, 0], filtered_series.values.tolist())

        filtered_series = _moving_sum_filter(test_series, threshold=2, window_size=2, use_index=True)
        self.assertListEqual([0, 0, 0, 1, 2, 0, 0, 0], filtered_series.values.tolist())

    def test_sum_of_turns(self):
        lat_long_stream_df = pd.DataFrame(
            {'latlng': [[52.1, 5.3], [52.2, 5.4], [58, 5.5], [52.4, 5.4], [52.5, 5.3]]},
            index=[1, 2, 3, 4, 5])

        total_turns_result = ActivityFeatures.sum_of_turns(lat_long_stream_df)
        self.assertAlmostEqual(0.12, total_turns_result, places=2)

    def test_sum_of_turns_given_window_size(self):
        """With window size of 1 sec (in this case 1 observation), no smoothing is applied, i.e. more severe turns"""
        lat_long_stream_df = pd.DataFrame(
            {'latlng': [[52.1, 5.3], [52.2, 5.4], [58, 5.5], [52.4, 5.4], [52.5, 5.3]]},
            index=[1, 2, 3, 4, 5])

        total_turns_result = ActivityFeatures.sum_of_turns(lat_long_stream_df, window_size=1)
        self.assertAlmostEqual(2.0, total_turns_result, places=1)

    def test_sum_of_turns_noise_threshold(self):
        """Should only count turns more severe than noise threshold as 'real' turns"""
        # ~ 45-degree turn, i.e. deviation ~ 0.25
        lat_long_stream_df = pd.DataFrame(
            {'latlng': [[5.0, 5.3], [5.1, 5.3], [5.2, 5.3], [5.3, 5.4], [5.4, 5.5], [5.5, 5.6], [5.6, 5.7]]},
            index=[1, 2, 3, 4, 5, 6, 7])

        total_turns_result = ActivityFeatures.sum_of_turns(lat_long_stream_df, window_size=1, noise_threshold=0.5)
        self.assertAlmostEqual(0.0, total_turns_result, places=2)
        total_turns_result = ActivityFeatures.sum_of_turns(lat_long_stream_df, window_size=1, noise_threshold=0.2)
        self.assertAlmostEqual(0.25, total_turns_result, places=2)

    def test_sum_of_turns_no_turns(self):
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

    def test_max_value_maintained_for_n_minutes_multiple_measurements(self):
        """Raises ValueError in case of stream df with != 1 column"""
        multi_stream_df = pd.DataFrame(
            {'heartrate': [100, 115, 120, 100, 110],
             'power': [200, 215, 220, 200, 210]},
            index=[180, 360, 540, 720, 900]
        )
        with self.assertRaises(ValueError):
            ActivityFeatures.max_value_maintained_for_n_minutes(multi_stream_df)
