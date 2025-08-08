from os import mkdir, listdir
from os.path import join, exists
from typing import override
from shutil import rmtree
from pandas import DataFrame
from numpy import average
import matplotlib.pyplot as plt

from src.common.helpers import read_dataframe
from src.common.kfold import AbstractFoldCrossValidation
from src.common.model import ModelConstructorArgs
from src.hpe_dnn.model import HpeDnn, HpeDnnMultiRunTrainArgs

class HpeDnnFoldCrossValidation(AbstractFoldCrossValidation):
    
    def __init__(self, model_args: ModelConstructorArgs,
            train_run_args: HpeDnnMultiRunTrainArgs):
        AbstractFoldCrossValidation.__init__(self, model_args, train_run_args, HpeDnn)
    
    def __get_fold_dataset_path(self):
        return join(self._model_args.data_root_path, "df", self._model_args.dataset_name)

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
    def clear_fold(self):
        rmtree(join(self.__get_fold_dataset_path(), "current_fold"))

    @override        
    def print_box_plot(self):
        model_root = join(self._model_args.data_root_path, "runs", "hpe_dnn")
        fold_models = [model_name for model_name in listdir(model_root) if f"{self._model_args.name}-fold" in model_name]
        metrics = []
        for fold_model in fold_models:
            model = HpeDnn(args=ModelConstructorArgs(
                name=fold_model,
                data_root_path=self._model_args.data_root_path,
                dataset_name=join(self._model_args.dataset_name, "current_fold")
            ))
            metrics.append(model.get_test_metrics()["accuracy"])

        print(f"Average Top 1 accuracy: {average(metrics)}")
        
        plt.figure()
        plt.boxplot(metrics)
        plt.show()