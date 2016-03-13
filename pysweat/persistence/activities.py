from pymongo import UpdateOne
import pandas as pd
import numpy as np
import json


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


def get_activity_types(mongo):
    return mongo.db.activities.find().distinct('type')


def get_first_activity_without_feature_for_type(mongo, feature_name, activity_type='Run', athlete_id=None):
    """Returns the datetime of the first activity (in time) of the given type for which the given feature does not
    exist. If athlete_id is provided, returns only the datetime for that athlete."""
    athlete_id_filter_json = """, "athlete_id": %s""" % athlete_id
    match_expression_json = """{"$match": {"type": "%s",
                                        "suspicious": {"$exists": false},
                                        "flagged": false,
                                        "%s": {"$exists": false}
                                        %s}}""" % \
                            (activity_type, feature_name, athlete_id_filter_json if athlete_id else "")
    return mongo.db.activities.aggregate([json.loads(match_expression_json),
                                          {"$group": {"_id": "$athlete_id",
                                                      "first_date": {"$min": "$start_date_local"}}}])
