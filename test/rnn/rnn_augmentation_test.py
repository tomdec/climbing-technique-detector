import pytest
from pandas import DataFrame, Series
from numpy import concatenate
from os.path import exists

from src.labels import iterate_valid_labels
from src.sampling.dataframe import generate_correlated_data, append_to_row
from src.hpe.mp.landmarks import get_feature_labels
from src.hpe_dnn.helpers import binarize_labels

from src.rnn.augmentation import AugmentationPipeline

TEST_VIDEO_PATH = "test/data/video/test_video.mp4"


def __get_test_df(value: float = 1.0) -> DataFrame:

    features = get_feature_labels()
    label_names = list(iterate_valid_labels())
    label_names.sort()

    def append_test_video_path():
        return lambda row: append_to_row(row, TEST_VIDEO_PATH)

    def append_group():
        group = 0
        return lambda row: append_to_row(row, group)

    columns = [*features, *label_names, "video", "frame_num", "group"]
    labels = range(1, 8)
    labels_series = Series(labels, name="label")
    labels_binarized = binarize_labels(labels_series)

    matrix = generate_correlated_data(
        features, labels, value=value, background_value=0.45
    )
    matrix = concatenate([matrix, labels_binarized], axis=1)
    matrix = list(map(append_test_video_path(), matrix))
    matrix = [[*row, frame_num - 1] for (row, frame_num) in zip(matrix, labels)]
    matrix = list(map(append_group(), matrix))

    return DataFrame(data=matrix, columns=columns)


def test_testdata():
    df = __get_test_df(value=0.5)

    assert len(df.index) == 7


@pytest.mark.parametrize("test_data", [__get_test_df()])
def test_conservation_of_column_order(test_data: DataFrame):
    if not exists(TEST_VIDEO_PATH):
        pytest.skip(
            "Test video not found. Make sure it exists when running tests locally."
        )

    aug_pipeline = AugmentationPipeline()

    augmented = test_data.apply(aug_pipeline, axis=1)

    assert all(test_data.columns == augmented.columns)


@pytest.mark.parametrize(
    "test_data,expected",
    [
        (
            __get_test_df(0.5),
            [
                0.522142254864728,
                0.4669912267614294,
                0.4669912267614294,
                0.4669912267614294,
                0.4669912267614294,
                0.4669912267614294,
                0.4669912267614294,
            ],
        )
    ],
)
def test_seeded_tranformations(test_data: DataFrame, expected):
    if not exists(TEST_VIDEO_PATH):
        pytest.skip(
            "Test video not found. Make sure it exists when running tests locally."
        )

    aug_pipeline = AugmentationPipeline()
    aug_pipeline.set_seed(123)

    actual = test_data.apply(aug_pipeline, axis=1)

    assert all(actual[actual.columns[1]].values == expected)


@pytest.mark.parametrize(
    "test_data",
    [__get_test_df(0.5)],
)
def test_multiple_seeded_tranformations(test_data: DataFrame):
    if not exists(TEST_VIDEO_PATH):
        pytest.skip(
            "Test video not found. Make sure it exists when running tests locally."
        )

    aug_pipeline = AugmentationPipeline()
    aug_pipeline.set_seed(123)

    first_actual = test_data.apply(aug_pipeline, axis=1)
    second_actual = test_data.apply(aug_pipeline, axis=1)

    assert all(
        first_actual[first_actual.columns[1]].values
        == second_actual[second_actual.columns[1]].values
    )
