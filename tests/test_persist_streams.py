import unittest
from mock import patch
import pandas as pd
from pysweat.persistence.streams import load_stream


class StreamPersistenceTest(unittest.TestCase):
    @patch('pymongo.MongoClient')
    def test_load_stream_by_activity_id_and_type(self, mongo_mock):
        """Should load single stream as 1-column dataframe, indexed by time"""
        mongo_mock.db.streams.find_one.return_value = {
            '_id': '123',
            'activity_id': 456,
            'data': [101, 102, 103],
            'type': 'velocity_smooth'
        }

        result = load_stream(mongo_mock, activity_id=456, stream_type='velocity_smooth')
        self.assertIs(type(result), pd.DataFrame)
        self.assertEqual(list(result.columns.values), ['velocity_smooth'])
        self.assertEqual(list(result.velocity_smooth.values), [101, 102, 103])
        self.assertEqual(list(result.index.values), [101, 102, 103])
