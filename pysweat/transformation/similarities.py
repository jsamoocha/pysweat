from __future__ import division

from math import pi

import numpy as np


def cosine_similarity(v1, v2):
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 if cosine > 1 else (-1 if cosine < -1 else cosine)  # Correcting for floating point rounding errors


def cosine_to_deviation(stream_df, cosine_col='cos'):
    # deviation corresponds (linearly) to turn severity, e.g. 45 deg = 0.25, 90 deg = 0.5, 180 deg = 1
    return stream_df.assign(deviation=np.arccos(stream_df[cosine_col]) / pi)
