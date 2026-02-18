import pytest
from pandas import DataFrame, Series, concat
from numpy import arange, ones
from numpy.random import rand, choice

from src.labels import iterate_valid_labels
from src.rnn.data import WindowGenerator


def get_test_data():

    test_features = DataFrame(rand(100, 20), columns=[f"feat{n}" for n in arange(20)])
    test_videos = Series([f"{x}" for x in arange(0, 100)], name="video")
    test_frames = Series(arange(0, 200, 2), name="frame_num")
    test_groups = Series(arange(0, 100), name="group")
    test_labels = Series(choice(list(iterate_valid_labels()), 100), name="label")

    return concat(
        [test_features, test_videos, test_frames, test_groups, test_labels], axis=1
    )


@pytest.mark.parametrize(
    "input_width,spacing,expected",
    [
        (5, 1, 5),
        (5, 2, 9),
        (5, 3, 13),
    ],
)
def test_window_generator_total_window_size(input_width, spacing, expected):
    data = get_test_data()

    wg = WindowGenerator(data, [0], [1], [2], input_width, spacing)

    assert wg.total_window_size == expected


@pytest.mark.parametrize(
    "input_width,spacing,expected_input,expected_label",
    [
        (5, 1, [0, 1, 2, 3, 4], [4]),
        (5, 2, [0, 2, 4, 6, 8], [8]),
        (5, 3, [0, 3, 6, 9, 12], [12]),
    ],
)
def test_window_generator_spacing(input_width, spacing, expected_input, expected_label):
    data = get_test_data()

    wg = WindowGenerator(data, [0], [1], [2], input_width, spacing)

    assert all(wg.input_indices == expected_input)
    assert all(wg.label_indices == expected_label)
