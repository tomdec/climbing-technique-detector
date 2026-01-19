from pandas import DataFrame, Series
from math import isnan
from cv2 import imread
from cv2.typing import MatLike
from typing import List, Tuple
import pytest

from src.sampling.dataframe import (
    generate_correlated_data,
    append_to_row,
    prepend_to_row,
)
from src.hpe.mp.landmarks import get_feature_labels as get_mp_features
from src.hpe.yolo.landmarks import get_feature_labels as get_yolo_features
from src.hpe_dnn.augmentation import AugmentationFunc, AugmentationPipeline


def __get_test_df(features: List[str], value: float = 1.0) -> DataFrame:

    def append_test_image_path():
        test_image_path = "test/data/img/NONE/test_image.jpg"
        return lambda row: append_to_row(row, test_image_path)

    columns = ["label", *features, "image_path"]
    labels = range(1, 8)

    matrix = generate_correlated_data(features, labels, value=value)
    matrix = list(map(prepend_to_row, matrix, labels))
    matrix = list(map(append_test_image_path(), matrix))

    return DataFrame(data=matrix, columns=columns)


def __get_mp_test_df(value: float = 1.0) -> DataFrame:
    features = get_mp_features()
    return __get_test_df(features, value)


def __get_yolo_test_df(value: float = 1.0) -> DataFrame:
    features = get_yolo_features()
    return __get_test_df(features, value)


identity_augmentation: AugmentationFunc = lambda _, coordinates, visibility: (
    coordinates,
    visibility,
)


@pytest.mark.parametrize(
    "test_data, dim", [(__get_mp_test_df(), 3), (__get_yolo_test_df(), 2)]
)
def test_identity_transformations(test_data: DataFrame, dim: int):

    aug_pipeline = AugmentationPipeline(dim, identity_augmentation)

    def assert_indentity_transformation(input: Series):

        output = aug_pipeline(input)

        for header in input.index:
            assert (input[header] == output[header]) or (
                isnan(input[header]) and isnan(output[header])
            )

    test_data.apply(assert_indentity_transformation, axis=1)


@pytest.mark.parametrize("test_data", [__get_mp_test_df(0.5), __get_yolo_test_df(0.5)])
def test_tranformations(test_data: DataFrame):
    aug_pipeline = AugmentationPipeline.for_dataframe(test_data)

    def assert_transformation(input: Series):

        output = aug_pipeline(input)

        for header in input.index:
            if header.endswith("visibility"):
                assert (input[header] == output[header]) or (output[header] == 0)

    test_data.apply(assert_transformation, axis=1)
