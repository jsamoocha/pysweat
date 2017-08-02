from __future__ import division
import logging
import numpy as np
from pysweat.persistence.streams import load_stream
from pysweat.transformation.gps import lat_long_to_x_y
from pysweat.transformation.streams import smooth, derivative, rolling_similarity
from pysweat.transformation.similarities import cosine_similarity, cosine_to_deviation
from pysweat.transformation.windows import subtract_n_minutes


class ActivityFeatures(object):
    def __init__(self, database_driver=None):
        self.database_driver = database_driver

    @staticmethod
    def sum_of_turns(lat_long_stream_df):

        lat_long_stream_df = lat_long_to_x_y(lat_long_stream_df)
        lat_long_stream_df = smooth(lat_long_stream_df, smooth_colname='x')
        lat_long_stream_df = smooth(lat_long_stream_df, smooth_colname='y')
        lat_long_stream_df = derivative(lat_long_stream_df, derivative_colname='x_smooth')
        lat_long_stream_df = derivative(lat_long_stream_df, derivative_colname='y_smooth')
        lat_long_stream_df = rolling_similarity(lat_long_stream_df, cosine_similarity,
                                                'dx_smooth_dt', 'dy_smooth_dt')

        try:
            return np.nansum([cosine_to_deviation(cos)
                              for cos in lat_long_stream_df.cosine_similarity_dx_smooth_dt_dy_smooth_dt])
        except ValueError:
            logging.warning('Failed to compute route deviation, returning NaN')
            return np.nan

    def max_value_maintained_for_n_minutes(self, activity_df, measurement='heartrate', window_size=5):
        all_max_values = [float('NaN')] * len(activity_df)

        for i in range(len(activity_df)):
            stream_df = load_stream(self.database_driver, activity_df.strava_id[i], '%s' % measurement)
            if stream_df is not None:
                min_index = stream_df.index.min()
                all_max_values[i] = np.nanmax(
                    [stream_df[measurement].loc[subtract_n_minutes(second,
                                                                   minutes=window_size,
                                                                   minimum_value=min_index):second].min()
                     for second in stream_df.index])

        activity_df['max_%s_%d_minutes' % (measurement, window_size)] = all_max_values
        return activity_df
