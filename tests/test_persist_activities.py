import unittest
import pandas as pd
from mock import patch
from pymongo import UpdateOne
from pysweat.persistence.activities import load_activities, save_activities


class ActivityPersistenceTest(unittest.TestCase):
    @patch('pymongo.MongoClient')
    def test_load_all_activities(self, mongo_mock):
        mongo_mock.db.activities.find.return_value = iter([{'id': '123', 'strava_id': 456, 'average_speed': 20.1},
                                                           {'id': '124', 'strava_id': 457, 'average_speed': 30.1}])

        result = list(load_activities(mongo_mock))
        self.assertEqual(len(result), 2)
        self.assertIn({'id': '123', 'strava_id': 456, 'average_speed': 20.1}, result)
        self.assertIn({'id': '124', 'strava_id': 457, 'average_speed': 30.1}, result)

    @patch('pymongo.MongoClient')
    def test_load_activities_with_simple_filter(self, mongo_mock):
        load_activities(mongo_mock, strava_id=456)
        mongo_mock.db.activities.find.assert_called_with({'strava_id': 456})

    @patch('pymongo.MongoClient')
    def test_load_activities_with_complex_filter(self, mongo_mock):
        load_activities(mongo_mock, athlete_id=123, ride_type={'$exists': False})
        mongo_mock.db.activities.find.assert_called_with({'athlete_id': 123, 'ride_type': {'$exists': False}})

    @patch('pymongo.MongoClient')
    def test_save_activities_all(self, mock_mongo):
        test_df = pd.DataFrame({'strava_id': [11, 12], 'a': [1, 2], 'b': [4, 5]})

        save_activities(mock_mongo, test_df)

        calls = mock_mongo.method_calls
        self.assertEqual(len(calls), 1)
        mock_mongo.db.activities.bulk_write.assert_called_with([
            UpdateOne({'strava_id': 11}, {'$set': {'a': 1, 'b': 4}}, upsert=True),
            UpdateOne({'strava_id': 12}, {'$set': {'a': 2, 'b': 5}}, upsert=True)
        ])
