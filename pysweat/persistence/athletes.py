import pandas as pd

def load_athletes(mongo, **query):
    return pd.DataFrame(list(mongo.db.athletes.find(query)))
