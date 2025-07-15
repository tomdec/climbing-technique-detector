from pandas import DataFrame, read_pickle
from numpy import split, nan
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from keras import layers, Input, Model, losses, utils
from os.path import join
from os import listdir
from keras._tf_keras.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from os import makedirs
from keras._tf_keras.keras.models import load_model
from typing import Mapping
from enum import Enum
from os.path import exists
from typing import Optional
from src.hpe_dnn.balancing import balance_func_factory

from src.hpe_dnn.augmentation import augment_keypoints

def read_data(location, verbose=False) -> DataFrame:
    data_frame = read_pickle(location)
    if verbose and (data_frame is DataFrame):
        print(data_frame.head())
    return data_frame

def df_to_dataset(dataframe: DataFrame, 
        balance=False,
        augment=False,
        shuffle=True,
        batch_size=32) -> tf.data.Dataset:
    df = dataframe.copy()
    
    if balance:
        balance_func = balance_func_factory(df)
        df = df.apply(balance_func, axis=1)

    if augment:
        df = df.apply(augment_keypoints, axis=1)

    labels = df.pop("technique")
    _ = df.pop("image_path")
    
    imp = SimpleImputer(missing_values=nan, strategy='mean')
    df = DataFrame(imp.fit_transform(df), columns=df.keys())
    df = {key: value.to_numpy()[:,tf.newaxis] for key, value in df.items()}

    encoder = LabelBinarizer()
    y = encoder.fit_transform(labels)

    ds = tf.data.Dataset.from_tensor_slices((dict(df), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def get_normalization_layer(name, dataset):
    # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, _: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer

def get_category_encoding_layer(name, dataset, max_tokens=None):
  
    index = layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, _: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))

def demo_batch(dataset: tf.data.Dataset):
    [(features, label_batch)] = dataset.take(1)

    print('Every feature:', list(features.keys()))
    print('A batch of Nose x-coordinates:', features['NOSE_x'])
    print('A batch of techniques:', label_batch )

def make_input_layer(train: tf.data.Dataset, normalize=True):
    numeric_features = ['NOSE_x', 'NOSE_y', 'NOSE_z', 'NOSE_visibility', 'LEFT_SHOULDER_x',
       'LEFT_SHOULDER_y', 'LEFT_SHOULDER_z', 'LEFT_SHOULDER_visibility',
       'LEFT_ELBOW_x', 'LEFT_ELBOW_y', 'LEFT_ELBOW_z', 'LEFT_ELBOW_visibility',
       'LEFT_WRIST_x', 'LEFT_WRIST_y', 'LEFT_WRIST_z', 'LEFT_WRIST_visibility',
       'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y', 'RIGHT_SHOULDER_z',
       'RIGHT_SHOULDER_visibility', 'RIGHT_ELBOW_x', 'RIGHT_ELBOW_y',
       'RIGHT_ELBOW_z', 'RIGHT_ELBOW_visibility', 'RIGHT_WRIST_x',
       'RIGHT_WRIST_y', 'RIGHT_WRIST_z', 'RIGHT_WRIST_visibility',
       'LEFT_HIP_x', 'LEFT_HIP_y', 'LEFT_HIP_z', 'LEFT_HIP_visibility',
       'LEFT_KNEE_x', 'LEFT_KNEE_y', 'LEFT_KNEE_z', 'LEFT_KNEE_visibility',
       'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'LEFT_ANKLE_z', 'LEFT_ANKLE_visibility',
       'LEFT_HEEL_x', 'LEFT_HEEL_y', 'LEFT_HEEL_z', 'LEFT_HEEL_visibility',
       'LEFT_FOOT_INDEX_x', 'LEFT_FOOT_INDEX_y', 'LEFT_FOOT_INDEX_z',
       'LEFT_FOOT_INDEX_visibility', 'RIGHT_HIP_x', 'RIGHT_HIP_y',
       'RIGHT_HIP_z', 'RIGHT_HIP_visibility', 'RIGHT_KNEE_x', 'RIGHT_KNEE_y',
       'RIGHT_KNEE_z', 'RIGHT_KNEE_visibility', 'RIGHT_ANKLE_x',
       'RIGHT_ANKLE_y', 'RIGHT_ANKLE_z', 'RIGHT_ANKLE_visibility',
       'RIGHT_HEEL_x', 'RIGHT_HEEL_y', 'RIGHT_HEEL_z', 'RIGHT_HEEL_visibility',
       'RIGHT_FOOT_INDEX_x', 'RIGHT_FOOT_INDEX_y', 'RIGHT_FOOT_INDEX_z',
       'RIGHT_FOOT_INDEX_visibility', 'RIGHT_THUMB_MCP_x', 'RIGHT_THUMB_MCP_y',
       'RIGHT_THUMB_MCP_z', 'RIGHT_THUMB_IP_x', 'RIGHT_THUMB_IP_y',
       'RIGHT_THUMB_IP_z', 'RIGHT_THUMB_TIP_x', 'RIGHT_THUMB_TIP_y',
       'RIGHT_THUMB_TIP_z', 'RIGHT_INDEX_FINGER_MCP_x',
       'RIGHT_INDEX_FINGER_MCP_y', 'RIGHT_INDEX_FINGER_MCP_z',
       'RIGHT_PINKY_MCP_x', 'RIGHT_PINKY_MCP_y', 'RIGHT_PINKY_MCP_z',
       'LEFT_THUMB_MCP_x', 'LEFT_THUMB_MCP_y', 'LEFT_THUMB_MCP_z',
       'LEFT_THUMB_IP_x', 'LEFT_THUMB_IP_y', 'LEFT_THUMB_IP_z',
       'LEFT_THUMB_TIP_x', 'LEFT_THUMB_TIP_y', 'LEFT_THUMB_TIP_z',
       'LEFT_INDEX_FINGER_MCP_x', 'LEFT_INDEX_FINGER_MCP_y',
       'LEFT_INDEX_FINGER_MCP_z', 'LEFT_PINKY_MCP_x', 'LEFT_PINKY_MCP_y',
       'LEFT_PINKY_MCP_z']
    
    all_inputs = {}
    encoded_feature = []

    for header in numeric_features:
        numerical_col = Input(shape=(1,), name=header)
        all_inputs[header] = numerical_col

        if normalize:
            normalization_layer = get_normalization_layer(header, train)
            normalized_numerical_col = normalization_layer(numerical_col)
            encoded_feature.append(normalized_numerical_col)
        else:
            encoded_feature.append(numerical_col)

    return all_inputs, layers.concatenate(encoded_feature)

def make_hpe_dnn_model(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch1.
        Don't change, make new DnnArch and function.
    """

    input_dict, all_features = make_input_layer(train, normalize=normalize)
    intermediate = layers.Dense(256, activation="relu")(all_features)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(128, activation="relu")(intermediate)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(64, activation="relu")(intermediate)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(32, activation="relu")(intermediate)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    output = layers.Dense(7, activation="softmax")(intermediate)

    model = Model(input_dict, output)

    model.compile(optimizer='adam', 
        loss=losses.BinaryCrossentropy(), 
        metrics=['accuracy'], 
        run_eagerly=debugging)
    
    return model

def make_hpe_dnn_model2(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch2.
        Don't change, make new DnnArch and function.
    """

    input_dict, all_features = make_input_layer(train, normalize=normalize)
    intermediate = layers.Dense(128, activation="relu")(all_features)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(64, activation="relu")(intermediate)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(32, activation="relu")(intermediate)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    output = layers.Dense(7, activation="softmax")(intermediate)

    model = Model(input_dict, output)

    model.compile(optimizer='adam', 
        loss=losses.BinaryCrossentropy(), 
        metrics=['accuracy'], 
        run_eagerly=debugging)
    
    return model

def make_hpe_dnn_model3(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch3.
        Don't change, make new DnnArch and function.
    """

    input_dict, all_features = make_input_layer(train, normalize=normalize)
    intermediate = layers.Dense(64, activation="relu")(all_features)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(32, activation="relu")(intermediate)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    output = layers.Dense(7, activation="softmax")(intermediate)

    model = Model(input_dict, output)

    model.compile(optimizer='adam', 
        loss=losses.BinaryCrossentropy(), 
        metrics=['accuracy'], 
        run_eagerly=debugging)
    
    return model

class DnnArch(Enum):
    ARCH1 = 0
    ARCH2 = 1
    ARCH3 = 2

_arch_mapping: Mapping[DnnArch, object] = {
    DnnArch.ARCH1: make_hpe_dnn_model,
    DnnArch.ARCH2: make_hpe_dnn_model2,
    DnnArch.ARCH3: make_hpe_dnn_model3
}

def plot_model(model: Model):
    utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir="TB")

def get_current_train_run(hpe_dnn_run_path):
    train_runs = [dir for dir in listdir(hpe_dnn_run_path) if "train" in dir]
    return f"train{len(train_runs)+1}"

def get_last_train_run(hpe_dnn_run_path):
    train_runs = [dir for dir in listdir(hpe_dnn_run_path) if "train" in dir]
    return train_runs[-1]

def train_model(model: Model, 
        train: tf.data.Dataset, 
        val: tf.data.Dataset,
        data_root_path: str):
    
    hpe_dnn_run_path = join(data_root_path, "runs", "hpe_dnn")
    current_train_run = get_current_train_run(hpe_dnn_run_path)
    
    checkpoint_dir = join(hpe_dnn_run_path, current_train_run, "models")
    makedirs(checkpoint_dir)
    checkpoint_path = join(checkpoint_dir, "epoch_{epoch:02d}__val_accuracy_{val_accuracy:.4f}.keras")
    cp_callback = ModelCheckpoint(checkpoint_path, 
        save_best_only=True, 
        save_weights_only=False, 
        verbose=1,
        monitor="val_accuracy")
    
    tensorboard_dir = join(hpe_dnn_run_path, current_train_run, "logs")
    makedirs(tensorboard_dir)
    tb_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
    
    model.fit(train, epochs=10, validation_data=val, 
        callbacks=[cp_callback, tb_callback])

def evaluate(model: Model, data: tf.data.Dataset):
    results = model.evaluate(data, return_dict=True)
    print(results)

def make_file(filepath):
    with open(filepath, 'w'):
        pass

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
    
    def execute_train_runs(self, runs=1, epochs=20, augment=False, balanced=False):
        for run in range(runs):
            print(f"starting run #{run}")
            self.initialize_model()
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
            callbacks=[cp_callback, tb_callback, csv_callback])

    def __get_latest_train_dir(self):
        model_dir = self.__get_model_dir()
        if (exists(model_dir)):
            train_runs = [dir for dir in listdir(model_dir) if "train" in dir]
            return join(model_dir, f"train{len(train_runs)+1}")
        else:
            return join(model_dir, "train1")

    def __get_checkpoint_dir(self):
        train_dir = self.__get_latest_train_dir()
        return join(train_dir, "models")

    def __get_tensorboard_log_dir(self):
        train_dir = self.__get_latest_train_dir()
        return join(train_dir, "logs")

    def __get_results_file_path(self):
        train_dir = self.__get_latest_train_dir()
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
        model_func = _arch_mapping[arch]
        self.model = model_func(train_ds, normalize, debugging, dropout_rate)
    
    def __get_data_from_split(self, split: str, augment, balance) -> tf.data.Dataset:
        df = read_data(join(self.__get_dataset_dir(), f"{split}.pkl"))
        return df_to_dataset(df, augment=augment, balance=balance)
