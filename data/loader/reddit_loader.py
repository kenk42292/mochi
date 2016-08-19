import numpy as np
import pickle
import csv
import itertools
import nltk
import utils

VOCAB_SIZE = 8000
REDDIT_TEXT_CSV = 'datasets/reddit_text.csv'
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENTENCE_START = "SENTENCE_START"
SENTENCE_END = "SENTENCE_END"

"""
Returns a massive list of words, in the orders given by concatenated sentences.
RESULT: The above-mentioned massive list of words
INDEX_TO_WORD: dictionary that converts indeces to words
WORD_TO_INDEX: dictionary that converts words to indeces
"""


def load_reddit(use_existing=True, data_path='datasets/reddit_text.csv', existing_path='datasets/training_data.pickle',
                vocabulary_size=VOCAB_SIZE,
                unknown_token=UNKNOWN_TOKEN,
                sentence_start_token=SENTENCE_START,
                sentence_end_token=SENTENCE_END):
    if use_existing:
        with open(existing_path, 'rU') as f:
            seq_all, index_to_word, word_to_index = pickle.load(f)
    else:
        print("Reading CSV file")
        with open(data_path, 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
        # Tokenize Sentences into words
        seq_all = []
        for sentence in sentences:
            seq_all += nltk.word_tokenize(sentence)
        # Count the word frequences
        word_freq = nltk.FreqDist(seq_all)  # *seq_all
        print("Found %d unique word tokens." % len(word_freq.items()))
        # Build both index_to_word and word_to_index vectors for the most common words
        vocab = word_freq.most_common(vocabulary_size - 3)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(sentence_start_token)
        index_to_word.append(sentence_end_token)
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        # Replace all words not in out vocabulary with unknown_token
        seq_all = [w if w in word_to_index else unknown_token for w in seq_all]
        seq_all = np.array([word_to_index[w] for w in seq_all])
        with open(existing_path, 'w') as f:
            pickle.dump([seq_all, index_to_word, word_to_index], f)

    return seq_all, VOCAB_SIZE, index_to_word, word_to_index
