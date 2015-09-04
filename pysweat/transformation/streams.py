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
