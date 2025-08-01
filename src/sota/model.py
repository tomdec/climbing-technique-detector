from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
from os.path import join, exists
from os import listdir, rename
from typing import Optional
from wandb import finish, init
from wandb.integration.ultralytics import add_wandb_callback
from json import dump, load
from sklearn.model_selection import KFold
from glob import glob
from random import random
from shutil import rmtree, copy
from numpy import average
import matplotlib.pyplot as plt


from src.sota.balancing import WeightedTrainer
from src.sampling.images import build_image_dirs

class SOTATrainArgs:

    @property
    def optimizer(self) -> str:
        """The optimizer to use during training"""
        return self._optimizer

    @property
    def lr0(self) -> float:
        """Initial learing rate for each training run"""
        return self._lr0

    @property
    def epochs(self) -> int:
        """Amount of epochs to train during each run"""
        return self._epochs

    @property
    def balanced(self) -> bool:
        """Indicates if the training data is balanced between labels"""
        return self._balanced
    
    def __init__(self, optimizer: str = "auto", lr0: float = 0.01, 
            epochs=20, balanced=False):
        self._optimizer = optimizer
        self._lr0 = lr0
        self._epochs = epochs
        self._balanced = balanced

class MultiRunSOTATrainArgs(SOTATrainArgs):
        
    @property
    def runs(self) -> int:
        """Amount of different runs to train the model"""
        return self._runs
    
    def __init__(self, runs=5, optimizer: str = "auto", lr0: float = 0.01, 
            epochs=10, balanced=False):
        SOTATrainArgs.__init__(self, optimizer, lr0, epochs, balanced)
        self._runs = runs


def raise_not_implemented_error(class_name, function_name):
    raise NotImplementedError(f"Invalid use of the class '{class_name}', it needs to implement the function 'f{function_name}'.")

class AbstractFoldCrossValidation:

    def __init__(self, data_root):
        n_splits = 10
        self._kf = KFold(n_splits=n_splits, shuffle=True)
        self._train_ratio = (n_splits - 2) / (n_splits - 1)

        self._data_root = data_root

    def get_full_data_list(self):
        raise_not_implemented_error(self.__class__.__name__, self.get_full_data_list.__name__)

    def build_fold(self, fold_num, train, test, full_data):
        raise_not_implemented_error(self.__class__.__name__, self.build_fold.__name__)

    def init_fold_model(self, fold_num) -> object:
        raise_not_implemented_error(self.__class__.__name__, self.init_fold_model.__name__)

    def execute_train_runs(self, model):
        raise_not_implemented_error(self.__class__.__name__, self.execute_train_runs.__name__)
        
    def test_model(self, model):
        raise_not_implemented_error(self.__class__.__name__, self.test_model.__name__)
        
    def clear_fold(self):
        raise_not_implemented_error(self.__class__.__name__, self.clear_fold.__name__)
        
    def print_box_plot(self):
        raise_not_implemented_error(self.__class__.__name__, self.print_box_plot.__name__)

    def train_folds(self):
        full_data = self.get_full_data_list()

        for i, (train, test) in enumerate(self._kf.split(full_data)):
            fold_num = i + 1
            self.build_fold(fold_num, train, test, full_data)
            
            model = self.init_fold_model(fold_num)
            self.execute_train_runs(model)

            self.test_model(model)

            self.clear_fold()

        self.print_box_plot()

class SOTAFoldCrossValidation(AbstractFoldCrossValidation):

    def __init__(self, data_root, model_name, train_run_args: MultiRunSOTATrainArgs, 
            dataset_name: str = "techniques", yolo_model = None):
        
        AbstractFoldCrossValidation.__init__(self, data_root=data_root)
        self._model_name = model_name
        self._train_run_args = train_run_args

        self._dataset_name = dataset_name \
            if dataset_name.endswith('_kf') \
            else dataset_name + '_kf'
        self._yolo_model = yolo_model \
            if yolo_model is not None \
            else model_name

    def __get_fold_dataset_path(self):
        return join(self._data_root, "img", self._dataset_name)

    def get_full_data_list(self):
        path_to_all = join(self.__get_fold_dataset_path(), "all")
        return glob(path_to_all + "/**/*.*", recursive=True)

    def build_fold(self, fold_num, train, test, full_data):
        path_to_current = join(self.__get_fold_dataset_path(), "current_fold")
        build_image_dirs(path_to_current)
        
        print(f"Building fold {fold_num} ...")

        for filename_idx in train:
            src = full_data[filename_idx]
            dest = src.replace("/all/", "/current_fold/train/") \
                if random() < self._train_ratio \
                else src.replace("/all/", "/current_fold/val/")
            copy(src, dest)
            
        for filename_idx in test:
            src = full_data[filename_idx]
            dest = src.replace("/all/", "/current_fold/test/")
            copy(src, dest)

        train_len = len(glob(path_to_current + "/train/**/*.*", recursive=True))
        val_len = len(glob(path_to_current + "/val/**/*.*", recursive=True))
        test_len = len(glob(path_to_current + "/test/**/*.*", recursive=True))
        print(f"Fold {fold_num}: Train size = {train_len}, Val size = {val_len}, Test size = {test_len}")
        
    def init_fold_model(self, fold_num) -> object:
        return SOTA(self._data_root, f"{self._model_name}-fold{fold_num}", dataset_name=join(self._dataset_name, "current_fold"))

    def execute_train_runs(self, model):
        model.execute_train_runs(model=self._yolo_model, 
            runs=self._train_run_args.runs, 
            epochs=self._train_run_args.epochs, 
            balanced=self._train_run_args.balanced)
        
    def test_model(self, model):
        model.test_model()
        
    def clear_fold(self):
        rmtree(join(self.__get_fold_dataset_path(), "current_fold"))
        
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

class SOTA:

    data_root_path: str
    name: str
    dataset_name: str 
    model: Optional[YOLO] = None

    def __init__(self, data_root_path: str, name: str, 
            dataset_name: str = "techniques"):
    
        if (name == ""):
            raise Exception(f"name cannot be an empty string")
        
        self.data_root_path = data_root_path
        self.name = name
        self.dataset_name = dataset_name

    def initialize_model(self, name = ""):
        if (exists(self.__get_model_dir())):
            weights_path = self.__get_best_weights_path()
            self.__load_model(weights_path)
        else:
            self.__fresh_model(name)

    def execute_train_runs(self, model, runs=1, optimizer: str = "auto", lr0: float = 0.01, 
            epochs=20, balanced=False):
        
        for run in range(runs):
            print(f"starting run #{run}")
            self.initialize_model(name=model)
            self.train_model(optimizer=optimizer, lr0=lr0, epochs=epochs, balanced=balanced)

    def train_model(self, optimizer: str = "auto", lr0: float = 0.01, epochs=20, 
            balanced=False):
        if (self.model is None):
            raise Exception("Cannot train before model is initialized")
        
        trainer = WeightedTrainer if balanced else None

        dataset_path = self.__get_dataset_dir()
        project_path = self.__get_project_dir()

        config = {
            'name': self.name,
            'dataset_name': self.dataset_name,
            'optimizer': optimizer,
            'lr0': lr0,
            'balanced': balanced,
            'augmented': True,
            'run': self.__get_next_train_run()
        }
        init(project="detect-climbing-technique", job_type="train", group="sota", name=self.name, 
            config=config, dir=self.data_root_path)
        add_wandb_callback(self.model, enable_model_checkpointing=True)

        results = self.model.train(trainer=trainer,
            data=dataset_path, 
            epochs=epochs,
            imgsz=640,
            project=project_path,
            optimizer=optimizer,
            lr0=lr0)
        
        finish()
        
        print(results)

    def __get_model_dir(self):
        return join(self.data_root_path, "runs", "sota", self.name)

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

    def __get_best_weights_path(self):
        model_dir = self.__get_model_dir()
        train_list = [dir for dir in listdir(model_dir) if "train" in dir]
        return join(model_dir, train_list[-1], "weights", "best.pt")

    def __fresh_model(self, name):
        if (name == ""):
            name = self.name + ".yaml"

        print(f"loading a fresh model '{name}'")
        self.model = YOLO(name)
    
    def __load_model(self, best_weights_path):
        print(f"loading the model '{self.name}' with the weights at '{best_weights_path}'")
        self.model = YOLO(best_weights_path)

    def get_test_metrics(self):
        with open(join(self.__get_project_dir(), "test", "metrics.json"), "r") as file:
            return load(file)

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
    
# results = model.predict(img, verbose = False)
# result = results[0]
# idx = result.probs.top1
# conf = result.probs.top1conf.item()
# label = model.names[idx]

#TODO: try https://github.com/rigvedrs/YOLO-V11-CAM for activation heatmaps


