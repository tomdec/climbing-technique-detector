from src.common.helpers import raise_not_implemented_error

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

class MultiRunTrainArgs:

    @property
    def model(self) -> str:
        """Model or architecture to use when initializing the model"""
        return self._model

    @property
    def runs(self) -> int:
        """Amount of different runs to train the model"""
        return self._runs

    @property
    def train_args(self) -> TrainArgs:
        """Arguments to use during training"""
        return self._train_args

    def __init__(self, model, runs=5, train_args: TrainArgs = TrainArgs()):
        self._model = model
        self._runs = runs
        self._train_args = train_args

class ClassificationModel:

    def test_model(self):
        raise_not_implemented_error(self.__class__.__name__, self.test_model.__name__)

    def execute_train_runs(self, args: MultiRunTrainArgs):
        raise_not_implemented_error(self.__class__.__name__, self.execute_train_runs.__name__)
