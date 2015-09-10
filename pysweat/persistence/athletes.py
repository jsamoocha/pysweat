import pandas as pd

def load_athletes(mongo):
    return pd.DataFrame(list(mongo.db.athletes.find()))
