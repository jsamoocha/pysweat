from __future__ import division
import logging
import numpy as np
from pysweat.persistence.streams import load_stream
from pysweat.transformation.gps import lat_long_to_x_y
from pysweat.transformation.streams import smooth, derivative, rolling_similarity
from pysweat.transformation.similarities import cosine_similarity, cosine_to_deviation
from pysweat.transformation.windows import subtract_n_minutes


class ActivityFeatures(object):
    def __init__(self, database_driver):
        self.database_driver = database_driver

    def __total_deviation(self, activity_id):
        stream_df = load_stream(self.database_driver, activity_id, 'latlng')

        if stream_df is not None:
            stream_df = lat_long_to_x_y(stream_df)
            stream_df = smooth(stream_df, smooth_colname='x')
            stream_df = smooth(stream_df, smooth_colname='y')
            stream_df = derivative(stream_df, derivative_colname='x_smooth')
            stream_df = derivative(stream_df, derivative_colname='y_smooth')
            stream_df = rolling_similarity(stream_df, cosine_similarity, 'dx_smooth_dt', 'dy_smooth_dt')

            try:
                return np.nansum([cosine_to_deviation(cos)
                                  for cos in stream_df.cosine_similarity_dx_smooth_dt_dy_smooth_dt])
            except ValueError:
                logging.warning('Failed to compute route deviation for activity %d, returning NaN' % activity_id)
                return np.nan
        else:
            logging.warning('No gps stream available for activity %d, returning NaN' % activity_id)
            return np.nan

    def turns_per_km(self, activity_df):
        activity_df['turns_per_km'] = [self.__total_deviation(activity_df.strava_id[i]) / activity_df.distance[i]
                                       for i in activity_df.index]
        return activity_df

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
