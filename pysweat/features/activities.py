from __future__ import division

import logging

import numpy as np

from pysweat.transformation.gps import lat_long_to_x_y
from pysweat.transformation.similarities import cosine_similarity, cosine_to_deviation
from pysweat.transformation.streams import smooth, derivative, rolling_similarity
from pysweat.transformation.windows import subtract_n_minutes


class ActivityFeatures(object):
    @staticmethod
    def sum_of_turns(lat_long_stream_df):
        """
        Returns the total number of 360 degree turns during an activity, i.e. an activity consisting of a single lap
        on a running track would count to "1".
        :param lat_long_stream_df: Pandas dataframe with (at least) one column called 'latlng' consisting of
        2-element lists with lat-long values
        :return: numpy scalar representing the total sum of turns in the stream, or NaN if the computation failed
        """

        turns_stream_df = (
            lat_long_to_x_y(lat_long_stream_df)
                .pipe(smooth, smooth_colname='x')
                .pipe(smooth, smooth_colname='y')
                .pipe(derivative, derivative_colname='x_smooth')
                .pipe(derivative, derivative_colname='y_smooth')
                .pipe(rolling_similarity, cosine_similarity, 'dx_smooth_dt', 'dy_smooth_dt')
        )
        try:
            return np.nansum([cosine_to_deviation(cos)
                              for cos in turns_stream_df.cosine_similarity_dx_smooth_dt_dy_smooth_dt])
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
