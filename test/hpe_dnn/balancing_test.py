from pandas import DataFrame
from numpy import random
from src.hpe_dnn.balancing import balance_func_factory

from os.path import exists
from src.hpe_dnn.model import read_data

def __get_test_df():
    data = zip([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    return DataFrame(data, columns=['feature1', 'technique'])

def test_balance_func_factory():
    random.seed(10)
    test_data = __get_test_df().sample(10)

    fut = balance_func_factory(test_data)
    balanced = test_data.apply(fut, axis=1)

    assert all(balanced['technique'] == [3, 3, 1, 2, 2, 1, 4, 4, 2, 4])
