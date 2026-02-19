from pandas import DataFrame, concat
from os.path import join, isdir, exists, basename
from os import listdir, makedirs, mkdir
from numpy import ones
from glob import glob
from typing import Any, List
from re import search

from src.labels import get_label_value_from_path, value_to_name, get_dataset_name
from src.common.helpers import read_dataframe
from src.hpe.mp.landmarks import get_feature_labels


def __fetch_from_segment_features(image_paths: List[str]) -> List[List[Any]]:
    segment_df = None
    segment_name = ""
    result = []

    for image_path in image_paths:
        image_name = basename(image_path)
        label = value_to_name(get_label_value_from_path(image_path))

        m = search(r"(.+)__(\d+)__(\d+)\.png", image_name)
        video_name = m.group(1)
        segment_start = int(m.group(2))
        image_idx = int(m.group(3))

        if segment_df is None or segment_name != f"{video_name}__{segment_start}.pkl":
            segment_name = f"{video_name}__{segment_start}.pkl"
            segment_df = read_dataframe(
                join("data", "df", "segments", label, segment_name)
            )

        features = segment_df.iloc[image_idx].drop("frame_num").values
        result.append([*features, image_path])

    return result


def generate_hpe_feature_df(data_root, dataset_name):

    img_path = join(data_root, "img", dataset_name)
    df_path = join(data_root, "df", dataset_name)
    makedirs(df_path, exist_ok=True)

    feature_names = get_feature_labels()
    column_names = ["label", *feature_names, "image_path"]

    for data_split in listdir(img_path):
        data_split_path = join(img_path, data_split)
        if isdir(data_split_path):
            image_paths = glob(data_split_path + "/**/*.*", recursive=True)
            matrix = __fetch_from_segment_features(image_paths)
            df = DataFrame(data=matrix, columns=column_names)
            df.to_pickle(join(df_path, f"{data_split}.pkl"))


# TODO: reuse for unity dataset, if needed
def generate_correlated_data(
    feature_names: List[str],
    labels: List[int],
    value: float = 1.0,
    background_value: float = 0.25,
) -> List[List[int]]:
    """
    Generate data directly correlated to the labels.
    For testing purposes.

    Args:
        feature_names (List[str]): The columns for which data is generated.
        labels (List[int]): The labels the data will be correlated with.
        value (float): Value to assign the feature.

    Returns:
        List[List[Any]]: List of features, correlated to the labels.
    """

    def generate_features(label: int) -> List[int]:
        features = ones(len(feature_names)) * background_value
        features[label] = value
        return features

    return list(map(generate_features, labels))


def prepend_to_row(row: List[Any], addition: Any) -> List[Any]:
    return [addition, *row]


def append_to_row(row: List[Any], addition: Any) -> List[Any]:
    return [*row, addition]


def generate_unity_df(
    data_root_path: str,
    dataset_name: str,
    feature_names: List[str],
    combine_for_kfold: bool = False,
):
    column_names = ["label", *feature_names, "image_path"]

    img_ds_name = get_dataset_name()
    img_path = join(data_root_path, "img", img_ds_name)
    df_path = join(data_root_path, "df", dataset_name)
    if not exists(df_path):
        makedirs(df_path)

    for data_split in listdir(img_path):
        matrix = []
        data_split_path = join(img_path, data_split)
        if isdir(data_split_path):
            image_paths = glob(data_split_path + "/**/*.*", recursive=True)
            labels = list(map(get_label_value_from_path, image_paths))

            matrix = generate_correlated_data(feature_names, labels)
            matrix = list(map(prepend_to_row, matrix, labels))
            matrix = list(map(append_to_row, matrix, image_paths))

            df = DataFrame(data=matrix, columns=column_names)
            df.to_pickle(join(df_path, f"{data_split}.pkl"))

    if combine_for_kfold:
        combine_dataset(data_root_path, "unity")


def combine_dataset(data_root, dataset_name):
    og_dataset_path = join(data_root, "df", dataset_name)
    train = read_dataframe(join(og_dataset_path, "train.pkl"))
    test = read_dataframe(join(og_dataset_path, "test.pkl"))
    val = read_dataframe(join(og_dataset_path, "val.pkl"))

    all = concat([train, test, val], ignore_index=True)

    kf_dataset_path = join(data_root, "df", dataset_name + "_kf")
    if not exists(kf_dataset_path):
        mkdir(kf_dataset_path)

    all.to_pickle(join(kf_dataset_path, "all.pkl"))
