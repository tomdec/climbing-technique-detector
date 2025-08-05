from typing import Any
from os.path import exists

from src.common.helpers import raise_not_implemented_error

class ModelConstructorArgs:

    @property
    def name(self) -> str:
        """Name of the model. Will be used to store results under the `data/runs` folder."""
        return self._name

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

    def __init__(self, name: str, data_root_path: str = "data",
            dataset_name: str = "techniques"):
        
        if (name == ""):
            raise Exception(f"name cannot be an empty string")
        
        self._name = name
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
    
    def __init__(self, epochs=20, balanced=False):
        self._epochs = epochs
        self._balanced = balanced

class ModelInitializeArgs:
    
    @property
    def model(self) -> Any:
        """Key to determine the structure of the model when initializing."""
        return self._model
    
    def __init__(self, model: Any):
        self._model = model

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
    data_root_path: str
    dataset_name: str
    
    def __init__(self, args: ModelConstructorArgs):
        self.data_root_path = args.data_root_path
        self.name = args.name
        self.dataset_name = args.dataset_name
    
    def train_model(self, args: TrainArgs):
        raise_not_implemented_error(self.__class__.__name__, self.train_model.__name__)

    def _get_model_dir(self):
        raise_not_implemented_error(self.__class__.__name__, self._get_model_dir.__name__)

    def _get_best_model_path(self):
        raise_not_implemented_error(self.__class__.__name__, self._get_best_model_path.__name__)

    def _load_model(self, model_path: str):
        raise_not_implemented_error(self.__class__.__name__, self._load_model.__name__)

    def _fresh_model(self, args: ModelInitializeArgs):
        raise_not_implemented_error(self.__class__.__name__, self._fresh_model.__name__)

    def initialize_model(self, args: ModelInitializeArgs):
        if (exists(self._get_model_dir())):
            model_path = self._get_best_model_path()
            self._load_model(model_path)
        else:
            self._fresh_model(args)

    def execute_train_runs(self, args: MultiRunTrainArgs):
        for run in range(args.runs):
            print(f"starting run #{run}")
            self.initialize_model(arch=args.model_initialize_args)
            self.train_model(args=args.train_args)

    def test_model(self):
        raise_not_implemented_error(self.__class__.__name__, self.test_model.__name__)

