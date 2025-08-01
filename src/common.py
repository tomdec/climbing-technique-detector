from os import listdir
from os.path import splitext, split, exists
from numpy import array
from sklearn.model_selection import KFold

def get_filename(path: str):
    _, tail = split(path)
    name, _ = splitext(tail)
    return name

def get_split_limits(data_split_ratios):
    '''
    data_split_ratios = (train, val, test)
    Returns a pair in floats that represent the data split.
    '''
    normalized_split_ratios = array(data_split_ratios)/sum(data_split_ratios)
    train_limit = normalized_split_ratios[0]
    val_limit = normalized_split_ratios[0] + normalized_split_ratios[1]
    return train_limit, val_limit

def get_next_train_run(root_path: str):
    if not exists(root_path):
        return "train1"
    
    train_runs = [dir for dir in listdir(root_path) if "train" in dir]
    return f"train{len(train_runs)+1}"

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


def raise_not_implemented_error(class_name, function_name):
    raise NotImplementedError(f"Invalid use of the class '{class_name}', it needs to implement the function 'f{function_name}'.")


class ClassificationModel:

    def test_model(self):
        raise_not_implemented_error(self.__class__.__name__, self.test_model.__name__)

    def execute_train_runs(self, args: MultiRunTrainArgs):
        raise_not_implemented_error(self.__class__.__name__, self.execute_train_runs.__name__)


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

    def init_fold_model(self, fold_num) -> ClassificationModel:
        raise_not_implemented_error(self.__class__.__name__, self.init_fold_model.__name__)

    def execute_train_runs(self, model: ClassificationModel):
        raise_not_implemented_error(self.__class__.__name__, self.execute_train_runs.__name__)
        
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

            model.test_model()

            self.clear_fold()

            #TEMP
            if i == 1:
                break

        self.print_box_plot()
