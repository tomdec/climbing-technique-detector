from os.path import join
from glob import glob
from typing import override
from shutil import rmtree, copy
from numpy import average
import matplotlib.pyplot as plt

from src.sampling.images import build_image_dirs
from src.common.kfold import AbstractFoldCrossValidation
from src.sota.model import SOTA, SOTAConstructorArgs, SOTAMultiRunTrainArgs

class SOTAFoldCrossValidation(AbstractFoldCrossValidation):

    @staticmethod
    def evaluation_instance(name: str, model: str) -> 'SOTAFoldCrossValidation':
        """Create instance of the SOTA fold cross validation model only used for evaluation.

        Args:
            name (str): Name of the model
            model (str): Model name to load from Ultralytics.

        Returns:
            SOTAFoldCrossValidation: K-fold model instance
        """
        return SOTAFoldCrossValidation(model_args=SOTAConstructorArgs(name=name, model_arch=model))

    @override
    @property
    def train_run_args(self) -> SOTAMultiRunTrainArgs | None:
        return self._train_run_args

    def __init__(self, model_args: SOTAConstructorArgs,
            train_run_args: SOTAMultiRunTrainArgs | None = None):
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
        print(f"Fold {fold_num}: Train size = {len(train)}, Val size = {len(val)}, " +
            f"Test size = {len(test)}")

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
    def train_folds(self, 
            train_run_args: SOTAMultiRunTrainArgs | None = None):
        super().train_folds(train_run_args)

    @override
    def clear_fold(self):
        rmtree(join(self.__get_fold_dataset_path(), "current_fold"))

    @override        
    def print_box_plot(self):
        metrics = self.get_test_accuracy_metrics()

        print(f"Average Top 1 accuracy: {average(metrics)}")
        
        plt.figure()
        plt.boxplot(metrics)
        plt.show()
