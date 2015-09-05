import unittest
from pysweat.transformation.similarities import cosine_similarity


class SimilarityTransformationTest(unittest.TestCase):
    def test_cosine_similarity_similar_vectors_2d(self):
        """Should return 1 for vectors pointing in same directions"""
        v1 = (1, 1)
        v2 = (2, 2)

        self.assertAlmostEqual(1, cosine_similarity(v1, v2), 9)
