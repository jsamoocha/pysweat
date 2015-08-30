from pymongo import UpdateOne
import pandas as pd


def load_activities(mongo, **query):
    return pd.DataFrame(list(mongo.db.activities.find(query)))


def save_activities(mongo, activities_df):
    mongo.db.activities.bulk_write([
        UpdateOne({'strava_id': record['strava_id']},
                  {'$set': {key: value for (key, value) in record.items() if key != 'strava_id'}},
                  upsert=True)
        for record in activities_df.to_dict(outtype='record')
    ])
