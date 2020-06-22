import unittest
from mock import patch
from pysweat.persistence.athletes import load_athletes

class AthletePersistenceTest(unittest.TestCase):
    @patch('pymongo.MongoClient')
    def test_load_all_athletes(self, mongo_mock):
        mongo_mock.db.athletes.find.return_value = iter([
            {'id': 123, 'name': 'foobar', 'sex': 'M'},
            {'id': 456, 'name': 'baz', 'sex': 'F'}
        ])

        result = load_athletes(mongo_mock)

        self.assertEqual(len(result), 2)
        self.assertCountEqual(result.id, [123, 456])
        self.assertCountEqual(result.name, ['foobar', 'baz'])
        self.assertCountEqual(result.sex, ['M', 'F'])

    @patch('pymongo.MongoClient')
    def test_load_athletes_with_simple_filter(self, mongo_mock):
        load_athletes(mongo_mock, sex='F')
        mongo_mock.db.athletes.find.assert_called_with({'sex': 'F'})
