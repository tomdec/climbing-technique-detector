from pandas import DataFrame, Series, concat
from numpy import reshape, concatenate
import tensorflow as tf
from typing import Tuple

from src.labels import get_valid_label_count
from src.hpe_dnn.helpers import unbinarize_labels


def take_groups(df: DataFrame, groups: list) -> DataFrame:
    filtered = list(map(lambda group: df.query(f"group == {group}"), groups))
    return concat(filtered, axis=0, ignore_index=True)


def output_to_labels(output: list, label_names: list) -> Series:
    output_2d = reshape(output, (-1, get_valid_label_count()))
    output_df = DataFrame(output_2d, columns=label_names)
    return unbinarize_labels(output_df)


def split_input_output(data: tf.data.Dataset) -> Tuple[list, list]:
    input = []
    output = []
    for test_batch in data.as_numpy_iterator():
        input.append(test_batch[0])
        output.append(test_batch[1])

    input = concatenate(input, axis=0)
    output = concatenate(output, axis=0)

    return input, output
