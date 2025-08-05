from pyclbr import Class
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
from os.path import join, exists
from os import listdir, rename
from typing import Optional, override
from wandb import finish, init
from wandb.integration.ultralytics import add_wandb_callback
from json import dump, load
from glob import glob
from random import random
from shutil import rmtree, copy
from numpy import average
import matplotlib.pyplot as plt

from src.common.model import ModelInitializeArgs, TrainArgs, MultiRunTrainArgs, ClassificationModel
from src.common.kfold import AbstractFoldCrossValidation
from src.sota.balancing import WeightedTrainer
from src.sampling.images import build_image_dirs

class SOTATrainArgs(TrainArgs):

    @property
    def optimizer(self) -> str:
        """The optimizer to use during training"""
        return self._optimizer

    @property
    def lr0(self) -> float:
        """Initial learing rate for each training run"""
        return self._lr0

    def __init__(self, epochs=20, balanced=False,
            optimizer: str = "auto", lr0: float = 0.01):
        TrainArgs.__init__(self, epochs, balanced)

        self._optimizer = optimizer
        self._lr0 = lr0

class SOTAModelInitializeArgs(ModelInitializeArgs):

    @override
    @property
    def model(self) -> str:
        """Name of yolo model to use when initializing the model"""
        return self._model
    
    def __init__(self, model: str = ""):
        self._model = model
    
class SOTAMultiRunTrainArgs(MultiRunTrainArgs):
    
    def __init__(self, model_initialize_args: SOTAModelInitializeArgs = SOTAModelInitializeArgs(), 
            runs: int = 5, 
            train_args: SOTATrainArgs = SOTATrainArgs()):
        MultiRunTrainArgs.__init__(self, model_initialize_args, runs, train_args)

class SOTAFoldCrossValidation(AbstractFoldCrossValidation):

    def __init__(self, data_root: str, model_name: str, 
            train_run_args: SOTAMultiRunTrainArgs, 
            dataset_name: str = "techniques"):
        
        AbstractFoldCrossValidation.__init__(self, data_root=data_root)
        self._model_name = model_name
        self._train_run_args = train_run_args
        
        self._dataset_name = dataset_name \
            if dataset_name.endswith('_kf') \
            else dataset_name + '_kf'

    def __get_fold_dataset_path(self):
        return join(self._data_root, "img", self._dataset_name)
    
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
    def init_fold_model(self, fold_num) -> ClassificationModel:
        return SOTA(self._data_root, f"{self._model_name}-fold{fold_num}", dataset_name=join(self._dataset_name, "current_fold"))
    
    @override
    def execute_train_runs(self, model):
        model.execute_train_runs(args=self._train_run_args)

    @override
    def clear_fold(self):
        rmtree(join(self.__get_fold_dataset_path(), "current_fold"))

    @override        
    def print_box_plot(self):
        model_root = join(self._data_root, "runs", "sota")
        fold_models = [model_name for model_name in listdir(model_root) if f"{self._model_name}-fold" in model_name]
        metrics = []
        for fold_model in fold_models:
            sota = SOTA("data", fold_model, dataset_name=join(self._dataset_name, "current_fold"))
            metrics.append(sota.get_test_metrics()["metrics/accuracy_top1"])

        print(f"Average Top 1 accuracy: {average(metrics)}")
        
        plt.figure()
        plt.boxplot(metrics)
        plt.show()

class SOTA(ClassificationModel):

    model: Optional[YOLO] = None

    @override
    def execute_train_runs(self, args: SOTAMultiRunTrainArgs):
        ClassificationModel.execute_train_runs(self, args)

    @override
    def initialize_model(self, args: SOTAModelInitializeArgs):
        ClassificationModel.initialize_model(self, args)

    @override
    def train_model(self, args: SOTATrainArgs):
        if (self.model is None):
            raise Exception("Cannot train before model is initialized")
        
        trainer = WeightedTrainer if args.balanced else None

        dataset_path = self.__get_dataset_dir()
        project_path = self.__get_project_dir()

        config = {
            'name': self.name,
            'dataset_name': self.dataset_name,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'balanced': args.balanced,
            'augmented': True,
            'run': self.__get_next_train_run()
        }
        init(project="detect-climbing-technique", job_type="train", group="sota", name=self.name, 
            config=config, dir=self.data_root_path)
        add_wandb_callback(self.model, enable_model_checkpointing=True)

        results = self.model.train(trainer=trainer,
            data=dataset_path, 
            epochs=args.epochs,
            imgsz=640,
            project=project_path,
            optimizer=args.optimizer,
            lr0=args.lr0)
        
        finish()
        
        print(results)

    @override
    def _get_model_dir(self):
        return join(self.data_root_path, "runs", "sota", self.name)

    @override
    def _get_best_model_path(self):
        model_dir = self.__get_model_dir()
        train_list = [dir for dir in listdir(model_dir) if "train" in dir]
        return join(model_dir, train_list[-1], "weights", "best.pt")

    @override
    def _load_model(self, best_weights_path):
        print(f"loading the model '{self.name}' with the weights at '{best_weights_path}'")
        self.model = YOLO(best_weights_path)

    @override
    def _fresh_model(self, args: SOTAModelInitializeArgs):
        name = args.name
        if (name == ""):
            name = self.name + ".yaml"

        print(f"loading a fresh model '{name}'")
        self.model = YOLO(name)
    
    @override
    def test_model(self, write_to_wandb = True) -> DetMetrics:
        self.initialize_model()
        
        dataset_path = self.__get_dataset_dir()
        project_path = self.__get_project_dir()

        rename(join(dataset_path, "val"), join(dataset_path, "val_temp"))
        rename(join(dataset_path, "test"), join(dataset_path, "val"))
        try:
            
            if write_to_wandb:
                config = {
                    'name': self.name,
                    'dataset_name': self.dataset_name,
                    'balanced': False,
                    'augmented': False,
                    'run': 'test'
                }
                init(project="detect-climbing-technique", job_type="eval", group="sota", name=self.name, 
                    config=config, dir=self.data_root_path)
                add_wandb_callback(self.model, enable_model_checkpointing=True)
            
            metrics = self.model.val(project=project_path, name="test")
            
            saved_metrics = metrics.results_dict.copy()
            saved_metrics['speed'] = metrics.speed.copy()
            
            with open(join(project_path, "test", "metrics.json"), "w") as file:
                dump(saved_metrics, file)

            if write_to_wandb:
                finish()

            return metrics
            
        except Exception as ex:
            print(f"stopped with error: {ex.message}")
        finally:
            rename(join(dataset_path, "val"), join(dataset_path, "test"))
            rename(join(dataset_path, "val_temp"), join(dataset_path, "val"))
    
    def __get_dataset_dir(self):
        return join(self.data_root_path, "img", self.dataset_name)

    def __get_next_train_run(self):
        model_dir = self.__get_model_dir()
        if not exists(model_dir):
            return "train1"
        
        train_runs = [dir for dir in listdir(model_dir) if "train" in dir]
        return f"train{len(train_runs)+1}"

    def __get_project_dir(self):
        return join(self.data_root_path, "runs", "sota", self.name)

    def get_test_metrics(self):
        with open(join(self.__get_project_dir(), "test", "metrics.json"), "r") as file:
            return load(file)

# results = model.predict(img, verbose = False)
# result = results[0]
# idx = result.probs.top1
# conf = result.probs.top1conf.item()
# label = model.names[idx]

#TODO: try https://github.com/rigvedrs/YOLO-V11-CAM for activation heatmaps


