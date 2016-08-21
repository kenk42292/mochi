import numpy as np

"""
A dummy iterator that returns a large number of repeated counts from 0 to 9
"""


def load_digits():
    digits_seq = []
    index_to_digit = []
    for _ in range(1000):
        for digit in range(10):
            if not digit % 2:
                digits_seq.append(digit)
                if digit not in index_to_digit:
                    index_to_digit.append(digit)
    digit_to_index = dict(zip(index_to_digit, range(len(index_to_digit))))
    return np.array([digit_to_index[d] for d in digits_seq]), len(index_to_digit), index_to_digit, digit_to_index
