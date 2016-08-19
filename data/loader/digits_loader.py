import numpy as np

"""
A dummy iterator that returns a large number of repeated counts from 0 to 9
"""


def load_digits():
    result = []
    for _ in range(1000):
        for digit in range(10):
            result.append(digit)
    return np.array(result)
