import pandas as pd


def load_athletes(mongo, **query):
    return pd.DataFrame(list(mongo.db.athletes.find(query)))


def get_athlete_ids(mongo):
    return mongo.db.athletes.find().distinct('id')
