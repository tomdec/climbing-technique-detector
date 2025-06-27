from os.path import exists
from pandas import Series
from math import isnan

from src.hpe_dnn.model import read_data
from src.hpe_dnn.augmentation import __to_augmenting_array, __to_df_row

def __get_train_df():
    df_path = "data/df/techniques/train.pkl"
    if (not exists(df_path)):
        raise FileNotFoundError('Make sure to generate the hpe dataset before running this test.')

    return read_data(df_path)

def test_transformations():
    train = __get_train_df()

    def assert_indentity_transformation(input: Series):
        xyz, vis = __to_augmenting_array(input, 480, 640)
        output = __to_df_row(input, xyz, vis, 480, 640)

        for header in input.index:
            assert (input[header] == output[header]) or \
                (isnan(input[header]) and isnan(output[header]))

    train.head().apply(assert_indentity_transformation, axis=1)