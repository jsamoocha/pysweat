from __future__ import division
from math import acos, pi
import numpy as np


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cosine_to_deviation(cosine_of_angle):
    return acos(cosine_of_angle) / pi
