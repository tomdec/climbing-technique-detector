from os import listdir
from os.path import join
from glob import glob
from typing import override
from shutil import rmtree, copy
from numpy import average
import matplotlib.pyplot as plt

from common.model import ModelConstructorArgs
from src.sampling.images import build_image_dirs
from src.common.kfold import AbstractFoldCrossValidation
from src.sota.model import SOTA, SOTAMultiRunTrainArgs

class SOTAFoldCrossValidation(AbstractFoldCrossValidation):

    def __init__(self, model_args: ModelConstructorArgs,
            train_run_args: SOTAMultiRunTrainArgs):
        AbstractFoldCrossValidation.__init__(self, model_args, train_run_args, SOTA)

    def __get_fold_dataset_path(self):
        return join(self._model_args.data_root_path, "img", self._model_args.dataset_name)
    
    @override
    def get_full_data_list(self):
        path_to_all = join(self.__get_fold_dataset_path(), "all")
        return glob(path_to_all + "/**/*.*", recursive=True)

    @override
    def build_fold(self, fold_num, train, val, test, full_data):
        path_to_current = join(self.__get_fold_dataset_path(), "current_fold")
        build_image_dirs(path_to_current)
        
        print(f"Building fold {fold_num} ...")
        print(f"Fold {fold_num}: Train size = {len(train)}, Val size = {len(val)}, Test size = {len(test)}")

        for filename_idx in train:
            src = full_data[filename_idx]
            dest = src.replace("/all/", "/current_fold/train/")
            copy(src, dest)

        for filename_idx in val:
            src = full_data[filename_idx]
            dest = src.replace("/all/", "/current_fold/val/")
            copy(src, dest)
        
        for filename_idx in test:
            src = full_data[filename_idx]
            dest = src.replace("/all/", "/current_fold/test/")
            copy(src, dest)
        
    @override
    def clear_fold(self):
        rmtree(join(self.__get_fold_dataset_path(), "current_fold"))

    @override        
    def print_box_plot(self):
        model_root = join(self._data_root, "runs", "sota")
        fold_models = [model_name for model_name in listdir(model_root) if f"{self._model_name}-fold" in model_name]
        metrics = []
        for fold_model in fold_models:
            sota = SOTA(args=ModelConstructorArgs(
                name=fold_model,
                data_root_path=self._model_args.data_root_path,
                dataset_name=join(self._dataset_name, "current_fold")
            ))
            metrics.append(sota.get_test_metrics()["metrics/accuracy_top1"])

        print(f"Average Top 1 accuracy: {average(metrics)}")
        
        plt.figure()
        plt.boxplot(metrics)
        plt.show()
