from __future__ import division

import logging

import numpy as np

from pysweat.transformation.gps import lat_long_to_x_y
from pysweat.transformation.similarities import cosine_similarity, cosine_to_deviation
from pysweat.transformation.streams import smooth, derivative, rolling_similarity
from pysweat.transformation.windows import subtract_n_minutes


class ActivityFeatures(object):
    @staticmethod
    def sum_of_turns(lat_long_stream_df, window_size=3, noise_threshold=0):
        """
        Returns the total number of 360 degree turns during an activity, i.e. an activity consisting of a single lap
        on a running track would count to "1".
        :param window_size: window size for filters, expressed in seconds
        :type window_size: int
        :param lat_long_stream_df: Pandas dataframe with (at least) one column called 'latlng' consisting of
        2-element lists with lat-long values and index that represents number of seconds since the start of the
        activity
        :return: numpy scalar representing the total sum of turns in the stream, or NaN if the computation failed
        """
        mean_time_diff = np.diff(lat_long_stream_df.index.values).mean()
        filter_window_size = int(round(window_size / mean_time_diff))

        turns_stream_df = (
            lat_long_to_x_y(lat_long_stream_df)
                .pipe(smooth, smooth_colname='x', window_size=filter_window_size)
                .pipe(smooth, smooth_colname='y', window_size=filter_window_size)
                .pipe(derivative, derivative_colname='x_smooth')
                .pipe(derivative, derivative_colname='y_smooth')
                .pipe(rolling_similarity, cosine_similarity, 'dx_smooth_dt', 'dy_smooth_dt')
                .pipe(cosine_to_deviation, 'cosine_similarity_dx_smooth_dt_dy_smooth_dt')
        )

        def turn_filter(deviation_window):
            return (deviation_window.values[len(deviation_window) // 2]
                    if deviation_window.sum() > noise_threshold else 0)
        try:
            return np.nansum(turns_stream_df.deviation
                             .fillna(0)
                             .rolling(filter_window_size, center=True)
                             .apply(turn_filter, raw=False))
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
