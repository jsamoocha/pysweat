import pandas as pd
from pysweat.persistence.activities import load_activities


class AthleteFeatures(object):
    def __init__(self, database_driver):
        self.database_driver = database_driver

    def summary_stats(self, athlete_df, activity_type='Ride', activity_measurement='average_speed'):
        activity_df = load_activities(self.database_driver, type=activity_type)
        athlete_activity_stats = pd.DataFrame([(athlete_id,
                                                athlete_activities[activity_measurement].mean(),
                                                athlete_activities[activity_measurement].std(),
                                                len(athlete_activities))
                                               for athlete_id, athlete_activities in activity_df.groupby('athlete_id')],
                                              columns=['athlete_id',
                                                       activity_type.lower() + '_' + activity_measurement + '_' +
                                                       'mean',
                                                       activity_type.lower() + '_' + activity_measurement + '_' + 'std',
                                                       activity_type.lower() + '_' + 'count'])
        return pd.merge(athlete_df, athlete_activity_stats, how='left', on='athlete_id')
