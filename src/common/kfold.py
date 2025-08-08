from os import makedirs
from sklearn.model_selection import KFold
from typing import Tuple, Type
from numpy import ndarray, array, save, load
from random import sample
from os.path import join, exists

from src.common.helpers import raise_not_implemented_error
from src.common.model import ClassificationModel, ModelConstructorArgs, MultiRunTrainArgs

class AbstractFoldCrossValidation:

    _model_args: ModelConstructorArgs
    _train_run_args: MultiRunTrainArgs
    _model_type: Type[ClassificationModel]

    def __init__(self, model_args: ModelConstructorArgs,
            train_run_args: MultiRunTrainArgs,
            model_type: Type[ClassificationModel]):
        n_splits = 10
        self._kf = KFold(n_splits=n_splits, shuffle=True)
        self._train_ratio = (n_splits - 2) / (n_splits - 1)
        
        self._model_args = model_args \
            if model_args.dataset_name.endswith('_kf') \
            else ModelConstructorArgs(
                name=model_args.name,
                data_root_path=model_args.data_root_path,
                dataset_name=model_args.dataset_name + '_kf')

        self._train_run_args = train_run_args
        self._model_type = model_type

    def get_full_data_list(self):
        raise_not_implemented_error(self.__class__.__name__, self.get_full_data_list.__name__)

    def build_fold(self, fold_num, train, val, test, full_data):
        raise_not_implemented_error(self.__class__.__name__, self.build_fold.__name__)

    def __init_fold_model(self, fold_num) -> ClassificationModel:
        adapted_args = ModelConstructorArgs(
            name=f"{self._model_args.name}-fold{fold_num}",
            data_root_path=self._model_args.data_root_path,
            dataset_name=join(self._model_args.dataset_name, "current_fold"))
        
        return self._model_type(adapted_args)

    def clear_fold(self):
        raise_not_implemented_error(self.__class__.__name__, self.clear_fold.__name__)
        
    def print_box_plot(self):
        raise_not_implemented_error(self.__class__.__name__, self.print_box_plot.__name__)

    def __split_val(self, train, test) -> Tuple[ndarray, ndarray, ndarray]:
        val_set = set(sample(list(train), len(test)))
        train = array(list(set(train) - val_set))
        val = array(list(val_set))

        return (train, val, test)

    def __split_files_exist(self, model_dir):
        return exists(join(model_dir, "split", "train.npy")) and \
            exists(join(model_dir, "split", "val.npy")) and \
            exists(join(model_dir, "split", "test.npy"))

    def __load_split(self, model_dir) -> Tuple[ndarray, ndarray, ndarray]:
        return (
            load(join(model_dir, "split", "train.npy")),
            load(join(model_dir, "split", "val.npy")),
            load(join(model_dir, "split", "test.npy"))
        )

    def __save_split(self, model_dir, split: Tuple[ndarray, ndarray, ndarray]):
        (train, val, test) = split
        
        makedirs(join(model_dir, "split"), exist_ok=True)
        save(join(model_dir, "split", "train.npy"), train)
        save(join(model_dir, "split", "val.npy"), val)
        save(join(model_dir, "split", "test.npy"), test)

    def train_folds(self):
        full_data = self.get_full_data_list()

        for i, (train, test) in enumerate(self._kf.split(full_data)):
            fold_num = i + 1
            
            model = self.__init_fold_model(fold_num)
            model_dir = model._get_model_dir()
            
            if self.__split_files_exist(model_dir):
                (train, val, test) = self.__load_split(model_dir)
            else:
                (train, val, test) = self.__split_val(train, test)
                self.__save_split(model_dir, (train, val, test))
            
            self.build_fold(fold_num, train, val, test, full_data)
            
            model.execute_train_runs(self._train_run_args)

            model.test_model()

            self.clear_fold()

        self.print_box_plot()

    def test_folds(self):
        #TODO: build folds and test each model without training
        pass
