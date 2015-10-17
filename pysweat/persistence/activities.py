from pymongo import UpdateOne
import pandas as pd
import numpy as np


def load_activities(mongo, **query):
    return pd.DataFrame(list(mongo.db.activities.find(query)))


def __should_write_field(key, value):
    if key != 'strava_id':
        try:
            return not np.isnan(value)
        except TypeError:
            return True
    else:
        return False


def save_activities(mongo, activities_df):
    mongo.db.activities.bulk_write([
        UpdateOne({'strava_id': record['strava_id']},
                  {'$set': {key: value for (key, value) in record.items()
                            if __should_write_field(key, value)}},
                  upsert=True)
        for record in activities_df.to_dict(orient='record')
    ])
