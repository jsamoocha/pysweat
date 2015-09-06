import unittest
from pysweat.transformation.similarities import cosine_similarity, cosine_to_deviation


class SimilarityTransformationTest(unittest.TestCase):
    def test_cosine_similarity_similar_vectors_2d(self):
        """Should return 1 for vectors pointing in same directions"""
        v1 = (1, 1)
        v2 = (2, 2)

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1, 9)

    def test_cosine_similarity_orthogonal_vectors_2d(self):
        """Should return 0 for orthogonal vectors"""
        v1 = (1, 1)
        v2 = (2, -2)

        self.assertAlmostEqual(cosine_similarity(v1, v2), 0, 9)

    def test_cosine_similarity_opposite_vectors_2d(self):
        """Should return -1 for vectors pointing in opposite directions"""
        v1 = (1, 1)
        v2 = (-2, -2)

        self.assertAlmostEqual(cosine_similarity(v1, v2), -1, 9)

    def test_cosine_similarity_similar_vectors_3d_lists(self):
        """Should compute cosine similarity regardless vector representation or dimension"""
        v1 = [1, 1, 1]
        v2 = [2, 2, 2]

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1, 9)

    def test_cosine_to_deviation(self):
        """Should return normalized angle between vectors, given a cosine similarity"""
        self.assertAlmostEqual(cosine_to_deviation(1), 0, 9)
        self.assertAlmostEqual(cosine_to_deviation(0), 0.5, 9)
        self.assertAlmostEqual(cosine_to_deviation(-1), 1, 9)
        self.assertAlmostEqual(cosine_to_deviation(0.5), 0.33333, 5)
