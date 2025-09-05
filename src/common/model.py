from json import dump, load
from typing import Any
from os.path import exists, join

from src.common.helpers import get_runs, raise_not_implemented_error, get_next_train_run, get_current_train_run,\
    get_current_test_run

class ModelConstructorArgs:

    @property
    def name(self) -> str:
        """Name of the model. Will be used to store results under the `data/runs` folder."""
        return self._name

    @property
    def model_arch(self) -> Any:
        """Key to determine the structure of the model when initializing."""
        return self._model_arch
    
    @property
    def data_root_path(self) -> str:
        """
        Path to the `data` folder relative to the directory where to code is executed.
        Default: "data", path relative to the jupyter notebook and when running code from the root of the project.
        """
        return self._data_root_path
    
    @property
    def dataset_name(self) -> str:
        """
        Name of the dataset to use.
        Default: "techniques", most general and complete dataset. 
        """
        return self._dataset_name

    def __init__(self, name: str, 
            model_arch: Any,
            data_root_path: str = "data",
            dataset_name: str = "techniques"):
        
        if (name == ""):
            raise Exception(f"name cannot be an empty string")
        
        self._name = name
        self._model_arch = model_arch
        self._data_root_path = data_root_path
        self._dataset_name = dataset_name

class TrainArgs:

    @property
    def epochs(self) -> int:
        """Amount of epochs to train during each run"""
        return self._epochs

    @property
    def balanced(self) -> bool:
        """Indicates if the training data is balanced between labels"""
        return self._balanced

    @property    
    def additional_config(self) -> dict:
        """Optional configuration to add to the config dictionary for weights and biases"""
        return self._additional_config
    
    @additional_config.setter
    def additional_config(self, value: dict):
        self._additional_config = value

    def __init__(self, epochs=20, balanced=False, additional_config={}):
        self._epochs = epochs
        self._balanced = balanced
        self._additional_config = additional_config

class TestArgs:

    @property
    def write_to_wandb(self) -> bool:
        """Write test results to Weigths and Biases"""
        return self._write_to_wandb
    
    @property    
    def additional_config(self) -> dict:
        """Optional configuration to add to the config dictionary for weights and biases"""
        return self._additional_config
    

    def __init__(self, write_to_wandb = False, 
            additional_config={}):
        self._write_to_wandb = write_to_wandb
        self._additional_config = additional_config

class ModelInitializeArgs:
    
    def __init__(self):
        pass

class MultiRunTrainArgs:

    @property
    def model_initialize_args(self) -> ModelInitializeArgs:
        """Arguments for initializing the AI model."""
        return self._model_initialize_args

    @property
    def runs(self) -> int:
        """Amount of different runs to train the model."""
        return self._runs

    @property
    def train_args(self) -> TrainArgs:
        """Arguments to use during training."""
        return self._train_args

    def __init__(self, model_initialize_args: ModelInitializeArgs, 
            runs: int = 5, 
            train_args: TrainArgs = TrainArgs()):
        self._model_initialize_args = model_initialize_args
        self._runs = runs
        self._train_args = train_args

class ClassificationModel:
    
    name: str

    @property
    def model_arch(self) -> Any:
        """Architecture of the AI model"""
        return self._model_arch

    data_root_path: str
    dataset_name: str
    
    def __init__(self, args: ModelConstructorArgs):
        self.name = args.name
        self._model_arch = args.model_arch
        self.data_root_path = args.data_root_path
        self.dataset_name = args.dataset_name

    def initialize_model(self, args: ModelInitializeArgs):
        if (self.__has_trained()):
            self._load_best_model()
        else:
            self._fresh_model(args)

    def execute_train_runs(self, args: MultiRunTrainArgs):
        for run in range(args.runs):
            print(f"starting run #{run}")
            self.initialize_model(args.model_initialize_args)
            self.train_model(args.train_args)
    
    def train_model(self, args: TrainArgs):
        raise_not_implemented_error(self.__class__.__name__, self.train_model.__name__)

    def test_model(self, args: TestArgs):
        raise_not_implemented_error(self.__class__.__name__, self.test_model.__name__)

    def _get_model_dir(self):
        raise_not_implemented_error(self.__class__.__name__, self._get_model_dir.__name__)

    def _get_best_model_path(self):
        raise_not_implemented_error(self.__class__.__name__, self._get_best_model_path.__name__)

    def _load_model(self, model_path: str):
        raise_not_implemented_error(self.__class__.__name__, self._load_model.__name__)

    def _load_best_model(self):
        model_path = self._get_best_model_path()
        self._load_model(model_path)

    def _fresh_model(self, args: ModelInitializeArgs):
        raise_not_implemented_error(self.__class__.__name__, self._fresh_model.__name__)

    def _get_next_train_run(self):
        model_dir = self._get_model_dir()
        return get_next_train_run(model_dir)

    def _get_next_train_dir(self):
        model_dir = self._get_model_dir()
        return join(model_dir, get_next_train_run(model_dir))

    def _get_current_train_run(self):
        model_dir = self._get_model_dir()
        return get_current_train_run(model_dir)

    def _get_current_test_run_path(self):
        model_dir = self._get_model_dir()
        return join(model_dir, get_current_test_run(model_dir))

    def _get_common_wandb_config(self) -> dict:
        return {
            'model_arch': self.model_arch,
            'dataset_name': self.dataset_name
        }
    
    def _save_test_metrics(self, metrics: dict):
        file_path = join(self._get_current_test_run_path(), "metrics.json")
        with open(file_path, 'w') as file:
            dump(metrics, file)

    def get_test_metrics(self) -> dict:
        file_path = join(self._get_current_test_run_path(), "metrics.json")
        with open(file_path, 'r') as file:
            return load(file)
        
    def get_test_accuracy_metric(self) -> float:
        raise_not_implemented_error(self.__class__.__name__, self.get_test_accuracy_metric.__name__)

    def __has_trained(self) -> bool:
        model_dir = self._get_model_dir()
        if not exists(model_dir):
            return False
        
        train_runs = get_runs(model_dir, "train")
        return len(train_runs) > 0