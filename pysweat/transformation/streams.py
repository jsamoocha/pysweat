from __future__ import division

import arrow
import pandas as pd
import numpy as np


def smooth(stream_df, window_size=3, smooth_colnames=None, use_index=False):
    """
    Smooths signals in dataframe with moving average filter. Uses either fix-sized window or fixed-duration window in
    case that observations are non-equally spread over time.
    :param stream_df: Pandas dataframe
    :param window_size: Window size for moving average filter, interpreted either as number of observations (default),
    or number of seconds (if use_index is true).
    :param smooth_colnames: Iterable of column names to use, by default all columns are smoothed.
    :param use_index: If True, the dataframe's index is interpreted as the number of seconds since start of the
    activity and will be used to determine the dynamic window size when applying the moving average.
    :return:
    """
    if use_index:
        tmp_df = stream_df.copy()  # prevent overwriting original index
        base_dt = arrow.get('2001-01-01')  # pick arbitrary date as base for artificial DatetimeIndex
        base_index_seconds = stream_df.index
        tmp_df.index = [pd.Timestamp(base_dt.shift(seconds=s).datetime) for s in base_index_seconds]
        window_size = str(window_size) + 's'

        return stream_df.assign(**{
            smooth_colname + '_smooth': pd.Series(tmp_df[smooth_colname].rolling(window=window_size).mean().values,
                                                  index=stream_df.index)
            for smooth_colname in smooth_colnames or stream_df.columns
        })
    else:
        return stream_df.assign(**{
            smooth_colname + '_smooth': pd.Series(stream_df[smooth_colname].rolling(window=window_size).mean())
            for smooth_colname in smooth_colnames or stream_df.columns
        })


def derivative(stream_df, derivative_colnames=None):
    return stream_df.assign(**{
        'd' + derivative_colname + '_dt': pd.Series(
            np.diff(stream_df[derivative_colname]) / np.diff(stream_df.index.values), index=stream_df.index[1:])
        for derivative_colname in derivative_colnames or stream_df.columns
    })


def rolling_similarity(stream_df, similarity_function, *column_names):
    vectors = list(zip(*[stream_df[column_name] for column_name in column_names]))
    return stream_df.assign(**{
        similarity_function.__name__ + '_' + '_'.join(column_names): pd.Series([
            similarity_function(vectors[i - 1], vectors[i]) for i in range(1, len(vectors))
        ], index=stream_df.index[1:])
    })
