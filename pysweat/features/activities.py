from __future__ import division

import logging

import arrow
import numpy as np
import pandas as pd

from pysweat.transformation.gps import lat_long_to_x_y
from pysweat.transformation.similarities import cosine_similarity, cosine_to_deviation
from pysweat.transformation.streams import smooth, derivative, rolling_similarity
from pysweat.transformation.windows import subtract_n_minutes


def _moving_sum_filter(series, window_size=3, threshold=0, use_index=False):
    if use_index:
        # assumes series has index representing seconds since start, window size is interpreted as seconds (and dynamic)
        series = series.copy()  # prevent overwriting original index
        base_dt = arrow.get('2001-01-01')  # pick arbitrary date as base for artificial DatetimeIndex
        base_index_seconds = series.index

        series.index = [pd.Timestamp(base_dt.shift(seconds=s).datetime) for s in base_index_seconds]
        sums_ascending = series.rolling(window=str(window_size) + 's').sum()

        # Reverse rolling sum
        series_rev = series[::-1]
        series_rev.index = [pd.Timestamp(base_dt.shift(seconds=-s).datetime) for s in base_index_seconds[::-1]]
        sums_descending = series_rev.rolling(window=str(window_size) + 's').sum()[::-1]
    else:
        # uses fixed window size
        sums_ascending = series.rolling(window=window_size, min_periods=1).sum()
        sums_descending = series[::-1].rolling(window=window_size, min_periods=1).sum()[::-1]

    return pd.Series(np.where(np.maximum(sums_ascending, sums_descending) > threshold, series, 0))


class ActivityFeatures(object):

    @staticmethod
    def sum_of_turns(lat_long_stream_df, window_size=3, noise_threshold=0):
        """
        Returns the total number of 180 degree turns during an activity, i.e. an activity consisting of a single lap
        on a running track would return "2". A threshold can be provided to filter out relatively small route
        deviations that are not real turns. The window size represents the number of seconds a turn takes.
        :param lat_long_stream_df: Pandas dataframe with (at least) one column called 'latlng' consisting of
        2-element lists with lat-long values and index that represents number of seconds since the start of the
        activity
        :param window_size: window size for filters, expressed in seconds
        :type window_size: int, > 0
        :param noise_threshold: the minimum amount of deviation within window_size seconds to count as a turn
        (sensible values are between 0.25 and 0.5, or 45-degree to 90-degree turns).
        :type noise_threshold: float, in the range [0, 1]
        :return: numpy scalar representing the total sum of turns in the stream, or NaN if the computation failed
        """
        turns_stream_df = (
            lat_long_to_x_y(lat_long_stream_df)
                .pipe(smooth, smooth_colnames=['x', 'y'], window_size=window_size, use_index=True)
                .pipe(derivative, derivative_colnames=['x_smooth', 'y_smooth'])
                .pipe(rolling_similarity, cosine_similarity, 'dx_smooth_dt', 'dy_smooth_dt')
                .pipe(cosine_to_deviation, 'cosine_similarity_dx_smooth_dt_dy_smooth_dt')
        )

        try:
            return _moving_sum_filter(
                turns_stream_df.deviation.fillna(0), use_index=True, window_size=window_size, threshold=noise_threshold
            ).sum()
        except ValueError:
            logging.warning('Failed to compute sum of turns, returning NaN')
            return np.nan

    @staticmethod
    def max_value_maintained_for_n_minutes(stream_df, window_size=5):
        """
        Returns the maximum value of a measurement that is maintained for at least n minutes, e.g.
        if during a low-intensity activity there was one intense interval of at least n minutes during which the
        minimum heart rate was x, then x is returned.
        :param stream_df: Pandas dataframe with exactly one column representing the measurement
        :param window_size: (integer) number of minutes for which a minimum value needs to be maintained
        :return: the maximum value of the measurement that was maintained for at least n minutes
        """
        if len(stream_df.columns) != 1:
            raise ValueError('Expecting exactly 1 measurement column in stream dataframe, got %d' %
                             len(stream_df.columns))
        min_index = stream_df.index.min()
        return np.nanmax(
            [stream_df.loc[subtract_n_minutes(second, minutes=window_size, minimum_value=min_index):second].min()
             for second in stream_df.index])
