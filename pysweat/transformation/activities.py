from pysweat.transformation.general import get_observations_without_feature
from pysweat.transformation.windows import select_activity_window


def weighted_average(activity_df, feature, weight_feature):
    return sum(activity_df[weight_feature] * activity_df[feature]) / sum(activity_df[weight_feature])


def compute_moving_averages(activity_df, feature_name, window_days):
    to_be_computed = get_observations_without_feature(activity_df, feature_name)
    activity_df.loc[to_be_computed, feature_name + '_' + str(window_days)] = [
        weighted_average(select_activity_window(activity_df, before, window_days), feature_name, 'distance')
        for before in activity_df.start_date_local[to_be_computed]
        ]
    return activity_df
