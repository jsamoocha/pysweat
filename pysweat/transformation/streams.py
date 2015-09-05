from __future__ import division
import pandas as pd
import numpy as np


def smooth(stream_df, window_size=3, smooth_colname='x'):
    stream_df[smooth_colname + '_smooth'] = pd.Series(pd.rolling_mean(stream_df[smooth_colname], center=True,
                                                                      window=window_size))
    return stream_df

def derivative(stream_df, derivative_colname='x'):
    stream_df['d' + derivative_colname + '_dt'] = pd.Series(
        np.diff(stream_df[derivative_colname]) / np.diff(stream_df.index.values), index=stream_df.index[1:])
    return stream_df

def rolling_similarity(stream_df, similarity_function, *column_names):
    vectors = zip(*[stream_df[column_name] for column_name in column_names])
    stream_df[similarity_function.__name__ + '_' + '_'.join(column_names)] = pd.Series([
        similarity_function(vectors[i - 1], vectors[i]) for i in range(1, len(vectors))
    ], index=stream_df.index[1:])
    return stream_df
