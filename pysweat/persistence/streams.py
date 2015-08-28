def load_stream(mongo, activity_id, stream_type):
    return mongo.db.streams.find({u'activity_id': activity_id, u'type': stream_type})