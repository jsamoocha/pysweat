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

    @patch('pymongo.MongoClient')
    def test_load_stream_non_existing(self, mongo_mock):
        """Should return None if stream does not exist"""
        mongo_mock.db.streams.find_one.return_value = None

        self.assertIsNone(load_stream(mongo_mock, activity_id=000, stream_type='non_existing'))

    @patch('pymongo.MongoClient')
    def test_load_stream_duplicate_timestamps(self, mongo_mock):
        """Should use the last observation in case of non-unique timestamps"""
        mongo_mock.db.streams.find_one.return_value = {
            '_id': '123',
            'activity_id': 456,
            'data': [101, 101, 103],
            'type': 'velocity_smooth'
        }

        result = load_stream(mongo_mock, activity_id=456, stream_type='velocity_smooth')
        self.assertEqual(len(result), 2)
        self.assertItemsEqual(result.velocity_smooth, [101, 103])
