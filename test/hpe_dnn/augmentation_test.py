from os.path import exists
from pandas import DataFrame, Series
from math import isnan
from cv2 import imread
from typing import List

from src.hpe_dnn.model import read_dataframe
from src.hpe_dnn.augmentation import __to_augmenting_array, __to_df_row, __transform_pipeline

def __get_train_df():
    df_path = "data/df/techniques/train.pkl"
    if (not exists(df_path)):
        raise FileNotFoundError('Make sure to generate the hpe dataset before running this test.')

    return read_dataframe(df_path)

def __get_test_df(features: List[str]) -> DataFrame:
    matrix = []
    columns={*features, "label", "image_path"}

    

    return DataFrame(data=matrix, columns=columns)


def test_identity_transformations():
    test_data = __get_train_df().sample(10)

    def assert_indentity_transformation(input: Series):
        xyz, vis = __to_augmenting_array(input, 480, 640)
        output = __to_df_row(input, xyz, vis, 480, 640)

        for header in input.index:
            assert (input[header] == output[header]) or \
                (isnan(input[header]) and isnan(output[header]))

    test_data.apply(assert_indentity_transformation, axis=1)

def test_tranformations():
    test_data = __get_train_df().sample(1)

    def assert_transformation(input: Series):
        img_path = input["image_path"]
        image = imread(img_path)
        height, width, _ = image.shape

        xyz, vis = __to_augmenting_array(input, height, width)
        transformed = __transform_pipeline(image=image, keypoints=xyz)
        output = __to_df_row(input, transformed['keypoints'], vis, height, width)

        for header in input.index:
            if header.endswith('visibility'):
                assert (input[header] == output[header]) or \
                    (isnan(input[header]) and isnan(output[header]))

    test_data.apply(assert_transformation, axis=1)