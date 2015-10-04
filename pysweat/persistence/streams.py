import pandas as pd


def load_stream(mongo, activity_id, stream_type):
    index_stream = mongo.db.streams.find_one({'activity_id': activity_id, 'type': 'time'})
    data_stream = mongo.db.streams.find_one({'activity_id': activity_id, 'type': stream_type})
    return pd.DataFrame({stream_type: data_stream['data']},
                        index=index_stream['data']).groupby(level=0).last() \
        if (index_stream and data_stream) \
        else None
