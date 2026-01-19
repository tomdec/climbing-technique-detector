import pytest
from pandas import Series
from numpy import array
from random import shuffle

from src.labels import iterate_valid_labels, name_to_value
from src.hpe_dnn.helpers import binarize_labels

label_names = Series(list(iterate_valid_labels()), name="label")
label_values = label_names.map(name_to_value)
expected_list = array(
    [
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
    ]
)


@pytest.mark.parametrize(
    "input,expected",
    [(label_names, expected_list), (label_values, expected_list)],
)
def test_binarize_labels(input, expected):
    actual = binarize_labels(input).values

    for actual_element, expected_element in zip(actual, expected):
        assert list(actual_element) == list(expected_element)


def test_binarize_labels_with_random_order():
    shuffled_idxs = list(range(len(label_values)))
    shuffle(shuffled_idxs)

    shuffled_labels = label_values[shuffled_idxs]
    shuffled_expected = expected_list[shuffled_idxs]

    actual_df = binarize_labels(shuffled_labels)

    for idx in range(len(label_values)):
        actual = actual_df.iloc[idx].values
        expected = shuffled_expected[idx]
        assert all(actual == expected)
