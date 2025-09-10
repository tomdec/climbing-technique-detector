from os import mkdir
from os.path import join, exists
from typing import override, Dict
from shutil import rmtree
from pandas import DataFrame
from numpy import average
import matplotlib.pyplot as plt

from src.common.helpers import read_dataframe
from src.common.kfold import AbstractFoldCrossValidation
from src.hpe_dnn.model import HpeDnn, HpeDnnConstructorArgs, HpeDnnMultiRunTrainArgs

class HpeDnnFoldCrossValidation(AbstractFoldCrossValidation):
    
    @staticmethod
    def evaluation_instance(name: str):
        """Create instance of the HPE DNN fold cross validation model only used for evaluation.

        Args:
            name (str): Name of the model.

        Returns:
            HpeDnnFoldCrossValidation: K-fold model instance
        """
        return HpeDnnFoldCrossValidation(model_args=HpeDnnConstructorArgs(name=name))

    @override
    @property
    def train_run_args(self) -> HpeDnnMultiRunTrainArgs | None:
        return self._train_run_args

    def __init__(self, model_args: HpeDnnConstructorArgs,
            train_run_args: HpeDnnMultiRunTrainArgs | None = None):
        AbstractFoldCrossValidation.__init__(self, model_args, train_run_args, HpeDnn)
    
    def __get_fold_dataset_path(self):
        return join(self._model_args.data_root_path, "df", self._model_args.dataset_name)

    @override
    def _get_additional_config(self, context_config: Dict = {}) -> Dict:
        return super()._get_additional_config(context_config) | {
            "dropout_rate": self.train_run_args.model_initialize_args.dropout_rate
        }

    @override
    def get_full_data_list(self) -> DataFrame:
        path_to_all = join(self.__get_fold_dataset_path(), "all.pkl")
        return read_dataframe(path_to_all)

    @override
    def build_fold(self, fold_num, train, val, test, full_data: DataFrame):
        path_to_current = join(self.__get_fold_dataset_path(), "current_fold")
        if not exists(path_to_current):
            mkdir(path_to_current)
            
        print(f"Building fold {fold_num} ...")
        print(f"Fold {fold_num}: Train size = {len(train)}, Val size = {len(val)}, Test size = {len(test)}")

        train_df = full_data.iloc[train]
        train_df.to_pickle(join(path_to_current, "train.pkl"))

        test_df = full_data.iloc[test]
        test_df.to_pickle(join(path_to_current, "test.pkl"))

        val_df = full_data.iloc[val]
        val_df.to_pickle(join(path_to_current, "val.pkl"))
    
    @override
    def train_folds(self, 
            train_run_args: HpeDnnMultiRunTrainArgs | None = None):
        super().train_folds(train_run_args)

    @override
    def clear_fold(self):
        rmtree(join(self.__get_fold_dataset_path(), "current_fold"))

    @override        
    def print_box_plot(self):
        metrics = self.get_test_accuracy_metrics()

        print(f"Average Top 1 categorical accuracy: {average(metrics)}")
        
        plt.figure()
        plt.boxplot(metrics)
        plt.show()