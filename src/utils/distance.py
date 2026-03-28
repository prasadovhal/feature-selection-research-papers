
import numpy as np

def hamming_distance(s1, s2):
    """Compute Hamming distance between two binary vectors."""
    return np.sum(s1 != s2)
