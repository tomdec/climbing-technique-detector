from pandas import DataFrame, Series, concat
from numpy import arange
from numpy.random import rand, choice

from src.labels import iterate_valid_labels
from src.rnn.data import get_group_split


def get_test_data():

    test_features = DataFrame(rand(100, 20), columns=[f"feat{n}" for n in arange(20)])
    test_groups = Series(arange(0, 100), name="group")
    test_labels = Series(choice(list(iterate_valid_labels()), 100), name="label")

    return concat([test_features, test_groups, test_labels], axis=1)


def test_deterministic_group_splits():
    data = get_test_data()

    expected_train_groups = [
        0,
        1,
        3,
        5,
        6,
        7,
        8,
        10,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        20,
        22,
        23,
        24,
        25,
        26,
        27,
        29,
        30,
        31,
        32,
        33,
        34,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        44,
        45,
        47,
        48,
        49,
        52,
        54,
        55,
        56,
        57,
        58,
        59,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        69,
        70,
        71,
        72,
        73,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        89,
        90,
        91,
        93,
        94,
        95,
        97,
        98,
        99,
    ]
    expected_val_groups = [2, 4, 19, 35, 46, 50, 53, 68, 88]
    expected_test_groups = [9, 11, 21, 28, 43, 51, 60, 74, 87, 92, 96]

    actual_train_groups, actual_val_groups, actual_test_groups = get_group_split(
        data, 1
    )

    assert all(expected_train_groups == actual_train_groups)
    assert all(expected_val_groups == actual_val_groups)
    assert all(expected_test_groups == actual_test_groups)
