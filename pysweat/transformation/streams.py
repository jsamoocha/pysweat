import pandas as pd

def smooth(raw_df, window_size=3, smooth_colname='x'):
    raw_df[smooth_colname + '_smooth'] = pd.Series(pd.rolling_mean(raw_df[smooth_colname], center=True,
                                                                   window=window_size))
    return raw_df
