from os.path import splitext, split
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
