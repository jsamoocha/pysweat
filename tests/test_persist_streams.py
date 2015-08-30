import unittest
from mock import patch
from pysweat.persistence.streams import load_stream


class StreamPersistenceTest(unittest.TestCase):
    @patch('pymongo.MongoClient')
    def test_load_stream_by_activity_id_and_type(self, mongo_mock):
        """Should load single stream"""
        mongo_mock.db.streams.find_one.return_value = {
            u'_id': u'123',
            u'activity_id': 456,
            u'data': [101, 102, 103],
            u'type': u'time'
        }

        result = load_stream(mongo_mock, activity_id=456, stream_type=u'time')
        self.assertDictEqual(result, {u'_id': u'123', u'activity_id': 456, u'data': [101, 102, 103], u'type': u'time'})
