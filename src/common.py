from os import listdir
from os.path import splitext, split, exists
from numpy import array

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

def get_next_train_run(root_path: str):
    if not exists(root_path):
        return "train1"
    
    train_runs = [dir for dir in listdir(root_path) if "train" in dir]
    return f"train{len(train_runs)+1}"