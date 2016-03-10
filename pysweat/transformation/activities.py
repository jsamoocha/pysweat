def weighted_average(activities_df, feature, weight_feature):
    return sum(activities_df[weight_feature] * activities_df[feature]) / sum(activities_df[weight_feature])
