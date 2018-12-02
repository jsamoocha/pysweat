import unittest

import pandas as pd

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

    def test_cosine_similarity_floating_point_rounding_error_positive(self):
        """Should return 1 as maximum in case of floating point rounding errors"""
        v1 = (0.0000015006504264572506635033732891315594, 0.0000006050474740115774352489097509533167)
        v2 = (0.0000015006504264503117695994660607539117, 0.0000006050474739005551327863940969109535)

        self.assertTrue(cosine_similarity(v1, v2) <= 1)

    def test_cosine_similarity_floating_point_rounding_error_negative(self):
        """Should return -1 as minimum in case of floating point rounding errors"""
        v1 = (0.0000015006504264572506635033732891315594, 0.0000006050474740115774352489097509533167)
        v2 = (-0.0000015006504264503117695994660607539117, -0.0000006050474739005551327863940969109535)

        self.assertTrue(cosine_similarity(v1, v2) >= -1)

    def test_cosine_similarity_similar_vectors_3d_lists(self):
        """Should compute cosine similarity regardless vector representation or dimension"""
        v1 = [1, 1, 1]
        v2 = [2, 2, 2]

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1, 9)

    def test_cosine_to_deviation(self):
        """Should return normalized angle between vectors, given a cosine similarity"""
        test_df = pd.DataFrame({'cos': [1, 0, -1, 0.5]})
        deviations = cosine_to_deviation(test_df).deviation.values
        self.assertAlmostEqual(deviations[0], 0, 9)
        self.assertAlmostEqual(deviations[1], 0.5, 9)
        self.assertAlmostEqual(deviations[2], 1, 9)
        self.assertAlmostEqual(deviations[3], 0.33333, 5)
