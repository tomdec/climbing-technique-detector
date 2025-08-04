import tensorflow as tf
from keras import Model
from os.path import join
from os import listdir
from keras._tf_keras.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from os import makedirs
from keras._tf_keras.keras.models import load_model
from os.path import exists
from typing import Optional
from wandb import init, finish
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from src.common.model import ClassificationModel, TrainArgs, MultiRunTrainArgs
from src.common.helpers import read_dataframe, make_file, get_next_train_run
from src.hpe_dnn.architecture import DnnArch, get_model_factory
from src.hpe_dnn.helpers import df_to_dataset

def evaluate(model: Model, data: tf.data.Dataset):
    results = model.evaluate(data, return_dict=True)
    print(results)

class HpeDnn:

    data_root_path: str
    name: str
    dataset_name: str 
    model: Optional[Model]

    def __init__(self, data_root_path: str, name: str, 
            dataset_name: str = "techniques"):
        
        if (name == ""):
            raise Exception(f"'{name}' is not a valid name")
        
        self.data_root_path = data_root_path
        self.name = name
        self.dataset_name = dataset_name
    
    def execute_train_runs(self, arch: DnnArch = DnnArch.ARCH1, runs=1, epochs=20, 
            augment=False, balanced=False):
        
        for run in range(runs):
            print(f"starting run #{run}")
            self.initialize_model(arch=arch)
            self.train_model(epochs=epochs, augment=augment, balanced=balanced)

    def initialize_model(self, arch: DnnArch = DnnArch.ARCH1, normalize: bool = True,
            dropout_rate = 0.1):
        if (exists(self.__get_model_dir())):
            model_path = self.__get_best_model_path()
            self.__load_model(model_path)
        else:
            self.__fresh_model(arch, normalize, dropout_rate)
    
    def train_model(self, epochs=20, augment=False, balanced=False):
        if (self.model is None):
            raise Exception("Cannot train before model is initialized")
        
        if (balanced and not augment):
            print("Warning: avoid reusing the exact same image multiple times by also enabling augmentation when balancing the dataset")
        
        train_ds = self.__get_data_from_split("train", augment=augment, balance=balanced)
        val_ds = self.__get_data_from_split("val", augment=False, balance=False)

        checkpoint_dir = self.__get_checkpoint_dir()
        log_dir = self.__get_tensorboard_log_dir()
        results_file = self.__get_results_file_path()

        config = {
            'name': self.name,
            'dataset_name': self.dataset_name,
            #'optimizer': optimizer,
            #'lr0': lr0,
            #'architecture': f"{arch}",
            'balanced': balanced,
            'augmented': augment,
            'run': self.__get_next_train_run()
        }
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

        self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, 
            callbacks=[cp_callback, tb_callback, csv_callback, WandbMetricsLogger(), 
                WandbModelCheckpoint(join(checkpoint_dir, "wandb.keras"))])
        
        finish()

    def __get_next_train_run(self):
        model_dir = self.__get_model_dir()
        return get_next_train_run(model_dir)

    def __get_next_train_dir(self):
        model_dir = self.__get_model_dir()
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

    def __get_model_dir(self):
        return join(self.data_root_path, "runs", "hpe_dnn", self.name)
    
    def __get_dataset_dir(self):
        return join(self.data_root_path, "df", self.dataset_name)

    def __get_best_model_path(self):
        model_dir = self.__get_model_dir()
        train_list = listdir(model_dir)
        model_path = join(model_dir, train_list[-1], "models")
        model_list = listdir(model_path)

        return join(model_path, model_list[-1])

    def __load_model(self, best_model_path):
        print(f"loading the model '{self.name}' from '{best_model_path}'")
        self.model = load_model(best_model_path)

    def __fresh_model(self, arch: DnnArch, normalize, dropout_rate):
        print(f"loading a fresh model '{self.name}'")

        train_ds = self.__get_data_from_split("train", augment=False, balance=False)
        debugging = False
        model_func = get_model_factory(arch)
        self.model = model_func(train_ds, normalize, debugging, dropout_rate)
    
    def __get_data_from_split(self, split: str, augment, balance) -> tf.data.Dataset:
        df = read_dataframe(join(self.__get_dataset_dir(), f"{split}.pkl"))
        return df_to_dataset(df, augment=augment, balance=balance)
