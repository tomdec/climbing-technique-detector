from sklearn.model_selection import KFold
from typing import Tuple
from numpy import ndarray, array
from random import sample

from src.common.helpers import raise_not_implemented_error
from src.common.model import ClassificationModel

class AbstractFoldCrossValidation:

    def __init__(self, data_root):
        n_splits = 10
        self._kf = KFold(n_splits=n_splits, shuffle=True)
        self._train_ratio = (n_splits - 2) / (n_splits - 1)

        self._data_root = data_root

    def get_full_data_list(self):
        raise_not_implemented_error(self.__class__.__name__, self.get_full_data_list.__name__)

    def build_fold(self, fold_num, train, val, test, full_data):
        raise_not_implemented_error(self.__class__.__name__, self.build_fold.__name__)

    def init_fold_model(self, fold_num) -> ClassificationModel:
        raise_not_implemented_error(self.__class__.__name__, self.init_fold_model.__name__)

    def execute_train_runs(self, model: ClassificationModel):
        raise_not_implemented_error(self.__class__.__name__, self.execute_train_runs.__name__)
        
    def clear_fold(self):
        raise_not_implemented_error(self.__class__.__name__, self.clear_fold.__name__)
        
    def print_box_plot(self):
        raise_not_implemented_error(self.__class__.__name__, self.print_box_plot.__name__)

    def __split_val(train, test) -> Tuple[ndarray, ndarray, ndarray]:
        val_set = set(sample(list(train), len(test)))
        train = array(list(set(train) - val_set))
        val = array(list(val_set))

        return (train, val, test)

    def train_folds(self):
        full_data = self.get_full_data_list()

        for i, (train, test) in enumerate(self._kf.split(full_data)):
            fold_num = i + 1
            (train, val, test) = self.__split_val(train, test)
            self.build_fold(fold_num, train, val, test, full_data)
            
            model = self.init_fold_model(fold_num)
            self.execute_train_runs(model)

            model.test_model()

            self.clear_fold()

        self.print_box_plot()
