import numpy as np

"""
Implementation of loader by Andrej Karpathy here: https://gist.github.com/karpathy/d4dee566867f8291f086
"""

def load_chars(path):
    data = open(path, 'r').read() # should be simple plain text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print 'data has %d characters, %d unique.' % (data_size, vocab_size)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    return np.array([char_to_ix[el] for el in data]), vocab_size, ix_to_char, char_to_ix