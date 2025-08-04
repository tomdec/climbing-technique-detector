from os import listdir
from os.path import splitext, split, exists
from numpy import array
from pandas import DataFrame, read_pickle
from sympy import root

def get_filename(path: str):
    _, tail = split(path)
    name, _ = splitext(tail)
    return name

def get_split_limits(data_split_ratios):
    '''
    data_split_ratios = (train, val, test)
    Returns a pair in floats that represent the data split.
    '''
    normalized_split_ratios = array(data_split_ratios)/sum(data_split_ratios)
    train_limit = normalized_split_ratios[0]
    val_limit = normalized_split_ratios[0] + normalized_split_ratios[1]
    return train_limit, val_limit

def __get_train_runs(root_path: str):
    return [dir for dir in listdir(root_path) if "train" in dir]

def get_next_train_run(root_path: str):
    if not exists(root_path):
        return "train1"
    
    train_runs = __get_train_runs(root_path)
    return f"train{len(train_runs)+1}"

def get_current_train_run(root_path: str):
    train_runs = __get_train_runs(root_path)
    return train_runs[-1]

def raise_not_implemented_error(class_name, function_name):
    raise NotImplementedError(f"Invalid use of the class '{class_name}', it needs to implement the function 'f{function_name}'.")

def read_dataframe(location, verbose=False) -> DataFrame:
    data_frame = read_pickle(location)
    if verbose and (data_frame is DataFrame):
        print(data_frame.head())
    return data_frame

def make_file(filepath):
    with open(filepath, 'w'):
        pass
