from pandas import DataFrame, concat
from os.path import join, isdir, exists
from os import listdir, makedirs, mkdir
from numpy import zeros
from glob import glob
from typing import Any, List, Callable

from src.labels import get_label_value_from_path, name_to_value
from src.common.helpers import read_dataframe

def generate_hpe_feature_df(data_path,
        feature_names: List[str],
        evaluate_func: Callable[[List[str]], List[List[Any]]],
        img_dataset_name = "techniques",
        df_dataset_name = "techniques"):

    column_names = [*feature_names, "label", "image_path"]
    
    img_path = join(data_path, "img", img_dataset_name)
    df_path = join(data_path, "df", df_dataset_name)
    if (not exists(df_path)):
        makedirs(df_path)

    for data_split in listdir(img_path):
        data_split_path = join(img_path, data_split)
        if (isdir(data_split_path)):
            image_paths = glob(data_split_path + "/**/*.*", recursive=True)
            matrix = evaluate_func(image_paths)
            df = DataFrame(data=matrix, columns=column_names)
            df.to_pickle(join(df_path, f"{data_split}.pkl"))

#TODO: reuse for unity dataset, if needed
def generate_correlated_data(feature_names: List[str],
        labels: List[int]) -> List[List[int]]:
    """
    Generate data directly correlated to the labels.
    For testing purposes.

    Args:
        feature_names (List[str]): The columns for which data is generated.
        labels (List[int]): The labels the data will be correlated with.

    Returns:
        List[List[Any]]: List of features, correlated to the labels.
    """
    def generate_features(label: int) -> List[int]:
        features = zeros(len(feature_names))
        features[label] = 1
        return features
    
    return list(map(generate_features, labels))

def append_to_row(row: List[Any], addition: Any) -> List[Any]:
    return [*row, addition]

def generate_unity_df(data_root_path: str,
        dataset_name: str,
        feature_names: List[str],
        combine_for_kfold: bool = False):
    column_names = [*feature_names, "label", "image_path"]

    img_path = join(data_root_path, "img", "techniques")
    df_path = join(data_root_path, "df", dataset_name)
    if (not exists(df_path)):
        makedirs(df_path)

    for data_split in listdir(img_path):
        matrix = []
        data_split_path = join(img_path, data_split)
        if (isdir(data_split_path)):
            image_paths = glob(data_split_path + "/**/*.*", recursive=True)
            labels = list(map(get_label_value_from_path, image_paths))
            
            matrix = generate_correlated_data(feature_names, labels)
            matrix = list(map(append_to_row, matrix, labels))
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

    kf_dataset_path = join(data_root, "df", dataset_name + '_kf')
    if not exists(kf_dataset_path):
        mkdir(kf_dataset_path)
    
    all.to_pickle(join(kf_dataset_path, "all.pkl"))