from argparse import ArgumentError
from os import listdir, makedirs
from os.path import splitext, split, exists, dirname
from numpy import array
from pandas import DataFrame, read_pickle
from cv2 import imread as cv2_imread, cvtColor, COLOR_BGR2RGB
from cv2.typing import MatLike


def get_filename(path: str):
    _, tail = split(path)
    name, _ = splitext(tail)
    return name


def get_split_limits(data_split_ratios):
    """
    data_split_ratios = (train, val, test)
    Returns a pair in floats that represent the data split.
    """
    normalized_split_ratios = array(data_split_ratios) / sum(data_split_ratios)
    train_limit = normalized_split_ratios[0]
    val_limit = normalized_split_ratios[0] + normalized_split_ratios[1]
    return train_limit, val_limit


def get_runs(root_path: str, run_type: str):
    return [dir for dir in listdir(root_path) if dir.startswith(run_type)]


def safe_index(totals: DataFrame, index: str, fallback=0) -> int:
    return totals[index] if index in totals.keys() else fallback


def __get_next_run(root_path: str, run_type: str):
    if run_type == "":
        raise ArgumentError(run_type, "Cannot be an empty string")

    if not exists(root_path):
        return run_type

    runs = get_runs(root_path, run_type)
    return f"{run_type}{len(runs)+1}"


def __get_current_run(root_path: str, run_type: str):
    runs = get_runs(root_path, run_type)
    return runs[-1]


def get_next_train_run(root_path: str):
    return __get_next_run(root_path, "train")


def get_current_train_run(root_path: str):
    return __get_current_run(root_path, "train")


def get_next_test_run(root_path: str):
    return __get_next_run(root_path, "test")


def get_current_test_run(root_path: str):
    return __get_current_run(root_path, "test")


def raise_not_implemented_error(class_name, function_name):
    raise NotImplementedError(
        f"Invalid use of the class '{class_name}', it needs to implement"
        + f"the function '{function_name}'."
    )


def read_dataframe(location, verbose=False) -> DataFrame:
    data_frame = read_pickle(location)
    if verbose and (data_frame is DataFrame):
        print(data_frame.head())
    return data_frame


def ensure_path_exists(location: str):
    path_to_location = dirname(location)
    makedirs(path_to_location, exist_ok=True)


def save_dataframe(location: str, df: DataFrame):
    ensure_path_exists(location=location)
    df.to_pickle(location)


def make_file(filepath):
    with open(filepath, "w"):
        pass


def imread(file_path: str, convert_to_rgb: bool = True) -> MatLike:
    image = cv2_imread(file_path)
    if convert_to_rgb:
        image = cvtColor(image, COLOR_BGR2RGB)
    return image
