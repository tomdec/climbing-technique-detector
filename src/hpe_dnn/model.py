from json import dump, load
import tensorflow as tf
from keras import Model
from os.path import join
from os import listdir, mkdir
from keras._tf_keras.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from os import makedirs
from keras._tf_keras.keras.models import load_model
from typing import Optional, override
from wandb import init, finish
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from numpy import concatenate

from src.common.model import ClassificationModel, ModelConstructorArgs, ModelInitializeArgs, TrainArgs, MultiRunTrainArgs
from src.common.helpers import get_current_test_run, get_current_train_run, get_next_test_run, read_dataframe, make_file, get_next_train_run
from src.hpe_dnn.architecture import DnnArch, get_model_factory
from src.hpe_dnn.helpers import df_to_dataset
from src.common.plot import plot_confusion_matrix

class HpeDnnConstructorArgs(ModelConstructorArgs):
    
    @override
    @property
    def model_arch(self) -> DnnArch:
        """Enum value specifying the architecture of the neural network model."""
        return self._model_arch
    
    def __init__(self, name: str, 
            model_arch: DnnArch = DnnArch.ARCH1,
            data_root_path: str = "data",
            dataset_name: str = "techniques"):
        ModelConstructorArgs.__init__(self, name, model_arch, data_root_path, dataset_name)


class HpeDnnTrainArgs(TrainArgs):

    @property
    def augment(self) -> bool:
        """Indicates if the training data will be augmented"""
        return self._augment
    
    def __init__(self, epochs=20, balanced=False, additional_config={}, augment=False):
        TrainArgs.__init__(self, epochs, balanced, additional_config)
        self._augment = augment

        if (balanced and not augment):
            print("Warning: avoid reusing the exact same image multiple times by also enabling augmentation when balancing the dataset")
        
class HpeDnnModelInitializeArgs(ModelInitializeArgs):

    @property
    def normalize(self) -> bool:
        """Normalize the numeric columns of the input data."""
        return self._normalize

    @property
    def dropout_rate(self) -> float:
        """Dropout rate to use between each layer."""
        return self._dropout_rate

    def __init__(self, normalize: bool = True,
            dropout_rate = 0.1):
        self._normalize = normalize
        self._dropout_rate = dropout_rate
    
class HpeDnnMultiRunTrainArgs(MultiRunTrainArgs):

    def __init__(self, model_initialize_args: HpeDnnModelInitializeArgs = HpeDnnModelInitializeArgs(), 
            runs: int = 5, 
            train_args: HpeDnnTrainArgs = HpeDnnTrainArgs()):
        MultiRunTrainArgs.__init__(self, model_initialize_args, runs, train_args)

class HpeDnn(ClassificationModel):

    model: Optional[Model] = None
    
    @override
    @property
    def model_arch(self) -> DnnArch:
        """Enum that is mapped to a factory function"""
        return self._model_arch

    @override
    def __init__(self, args: HpeDnnConstructorArgs):
        ClassificationModel.__init__(self, args)

    @override
    def execute_train_runs(self, args: HpeDnnMultiRunTrainArgs):
        ClassificationModel.execute_train_runs(self, args)

    @override
    def initialize_model(self, args: HpeDnnModelInitializeArgs):
        ClassificationModel.initialize_model(self, args)
    
    @override
    def train_model(self, args: HpeDnnTrainArgs):

        if self.model is None:
            raise Exception("Cannot train before model is initialized")
        
        train_ds = self.__get_data_from_split("train", augment=args.augment, balance=args.balanced)
        val_ds = self.__get_data_from_split("val", augment=False, balance=False)

        checkpoint_dir = self.__get_checkpoint_dir()
        log_dir = self.__get_tensorboard_log_dir()
        results_file = self.__get_results_file_path()

        config = {
            'model_arch': self.model_arch,
            'dataset_name': self.dataset_name,
            #'optimizer': optimizer,
            #'lr0': lr0,
            #'architecture': f"{arch}",
            'balanced': args.balanced,
            'augmented': args.augment,
            'run': self.__get_next_train_run()
        } | args.additional_config

        init(project="detect-climbing-technique", job_type="train", group="hpe_dnn", name=self.name, 
            config=config, dir=self.data_root_path)

        makedirs(checkpoint_dir)
        makedirs(log_dir)
        make_file(results_file)
        
        checkpoint_path = join(checkpoint_dir, "epoch_{epoch:02d}__val_accuracy_{val_accuracy:.4f}.keras")
        cp_callback = ModelCheckpoint(checkpoint_path, 
            save_best_only=True, 
            save_weights_only=False, 
            verbose=1,
            monitor="val_accuracy")
        
        tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        csv_callback = CSVLogger(filename=results_file)

        self.model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, 
            callbacks=[cp_callback, tb_callback, csv_callback, WandbMetricsLogger(), 
                WandbModelCheckpoint(join(log_dir, "wandb.keras"))])
        
        finish()

    @override
    def _get_model_dir(self):
        return join(self.data_root_path, "runs", "hpe_dnn", self.name)
    
    @override
    def _get_best_model_path(self):
        model_dir = self._get_model_dir()
        train_run = get_current_train_run(model_dir)
        model_path = join(model_dir, train_run, "models")
        model_list = listdir(model_path)

        return join(model_path, model_list[-1])

    @override
    def _fresh_model(self, args: HpeDnnModelInitializeArgs):
        print(f"loading a fresh model '{self.model_arch}'")

        train_ds = self.__get_data_from_split("train", augment=False, balance=False)
        debugging = False
        model_func = get_model_factory(self.model_arch)
        self.model = model_func(train_ds, args.normalize, debugging, args.dropout_rate)
    
    @override
    def _load_model(self, best_model_path):
        print(f"loading the model '{self.name}' from '{best_model_path}'")
        self.model = load_model(best_model_path)

    @override
    def test_model(self):
        model_path = self._get_model_dir()
        test_run = get_next_test_run(model_path)
        test_run_path = join(model_path, test_run)
        mkdir(test_run_path)

        test_data = self.__get_data_from_split(split="test", augment=False, balance=False)
        
        predictions = self.model.predict(test_data)
        labels = concatenate([y for _, y in test_data], axis=0)
        plot_confusion_matrix(labels, predictions, join(test_run_path, "confusion_matrix.png"))

        results = self.model.evaluate(test_data, return_dict=True)
        results_file = join(test_run_path, "metics.json")
        with open(join(results_file), "w") as file:
            dump(results, file)
        
        return results

    def get_test_metrics(self):
        model_path = self._get_model_dir()
        test_run = get_current_test_run(model_path)
        test_run_path = join(model_path, test_run)
        
        results_file = join(test_run_path, "metics.json")
        with open(results_file, 'r') as file:
            return load(file)

    def __get_next_train_run(self):
        model_dir = self._get_model_dir()
        return get_next_train_run(model_dir)

    def __get_next_train_dir(self):
        model_dir = self._get_model_dir()
        return join(model_dir, get_next_train_run(model_dir))

    def __get_checkpoint_dir(self):
        train_dir = self.__get_next_train_dir()
        return join(train_dir, "models")

    def __get_tensorboard_log_dir(self):
        train_dir = self.__get_next_train_dir()
        return join(train_dir, "logs")

    def __get_results_file_path(self):
        train_dir = self.__get_next_train_dir()
        return join(train_dir, "results.csv")

    def __get_dataset_dir(self):
        return join(self.data_root_path, "df", self.dataset_name)

    def __get_data_from_split(self, split: str, augment, balance) -> tf.data.Dataset:
        df = read_dataframe(join(self.__get_dataset_dir(), f"{split}.pkl"))
        return df_to_dataset(df, augment=augment, balance=balance)
