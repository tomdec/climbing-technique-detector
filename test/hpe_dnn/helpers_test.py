from pandas import Series
from numpy import array
from random import shuffle

from src.hpe_dnn.helpers import __binarize_labels

label_column = Series([1, 2, 3, 4, 5, 7, 8], name="technique")
expected_list = array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
])

def test_binarize_labels():
    actual_list = __binarize_labels(label_column)

    for (actual, expected) in zip(actual_list, expected_list):
        assert list(actual) == list(expected)

def test_binarize_labels_with_random_order():
    shuffled_idxs = list(range(len(label_column)))
    shuffle(shuffled_idxs)
    
    shuffled_labels = label_column[shuffled_idxs]
    shuffled_expected = expected_list[shuffled_idxs]

    actual_list = __binarize_labels(shuffled_labels)

    for (actual, expected) in zip(actual_list, shuffled_expected):
        assert list(actual) == list(expected)