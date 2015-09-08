from __future__ import division
import numpy as np
from pysweat.persistence.streams import load_stream
from pysweat.transformation.gps import lat_long_to_x_y
from pysweat.transformation.streams import smooth, derivative, rolling_similarity
from pysweat.transformation.similarities import cosine_similarity, cosine_to_deviation


class ActivityFeatures(object):
    def __init__(self, database_driver):
        self.database_driver = database_driver

    def __total_deviation(self, activity_id):
        stream_df = load_stream(self.database_driver, activity_id, 'latlng')
        stream_df = lat_long_to_x_y(stream_df)
        stream_df = smooth(stream_df, smooth_colname='x')
        stream_df = smooth(stream_df, smooth_colname='y')
        stream_df = derivative(stream_df, derivative_colname='x')
        stream_df = derivative(stream_df, derivative_colname='y')
        stream_df = rolling_similarity(stream_df, cosine_similarity, 'dx_dt', 'dy_dt')
        return np.nansum([cosine_to_deviation(cos) for cos in stream_df.cosine_similarity_dx_dt_dy_dt])

    def turns_per_km(self, activity_df):
        activity_df['turns_per_km'] = [self.__total_deviation(activity_df.strava_id[i]) / activity_df.distance[i]
                                       for i in activity_df.index]
        return activity_df
