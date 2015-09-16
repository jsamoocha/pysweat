import pandas as pd

def summary_stats(athlete_df, activity_df, activity_measurement='average_speed'):
    activity_type = activity_df.type[0]
    athlete_activity_stats = pd.DataFrame([(athlete_id,
                                            athlete_activities[activity_measurement].mean(),
                                            athlete_activities[activity_measurement].std(),
                                            len(athlete_activities))
                                           for athlete_id, athlete_activities in activity_df.groupby('athlete_id')],
                                          columns=['id',
                                                   activity_type.lower() + '_' + activity_measurement + '_' +
                                                   'mean',
                                                   activity_type.lower() + '_' + activity_measurement + '_' + 'std',
                                                   activity_type.lower() + '_' + 'count'])
    return pd.merge(athlete_df, athlete_activity_stats, how='left', on='id')
