import pandas as pd


def delta_constant(observations_df, constant, constant_description, measurement='x'):
    return observations_df.assign(**{
        'd_' + measurement + '_' + constant_description: observations_df[measurement] - constant
    })


def get_observations_without_feature(observations_df, feature_name):
    return (pd.isnull(observations_df[feature_name]) if feature_name in observations_df
            else [True] * len(observations_df.index))
