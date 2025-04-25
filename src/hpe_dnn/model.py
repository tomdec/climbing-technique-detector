from pandas import DataFrame, read_pickle
from numpy import split, nan
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from keras import layers, Input, Model, losses, utils

from common import get_split_limits

__data_location_from_notebook = "../data/df/hpe-dnn-data.pkl"

def read_data(location) -> DataFrame:
    data_frame = read_pickle(location)
    print(data_frame.head())
    return data_frame

def split_data(data: DataFrame, data_split):
    '''
    data_split = (train, val, test)
    returns tuple with splits of data in same order
    '''
    data_length = len(data)
    train_limit, val_limit = get_split_limits(data_split)

    train, val, test = split(data.sample(frac=1), [int(train_limit*data_length), int(val_limit*data_length)])

    print(len(train), 'training examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    return train, val, test

def df_to_dataset(dataframe: DataFrame, shuffle=True, batch_size=32) -> tf.data.Dataset:
    df = dataframe.copy()
    labels = df.pop("technique")

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

def generate_split_datasets(data: DataFrame, split, batch_size=32) -> tuple[tf.data.Dataset]:
    train, val, test = split_data(data, split)
    
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds

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

def make_hpe_dnn_model(train: tf.data.Dataset, normalize=True, debugging=False) -> Model:

    input_dict, all_features = make_input_layer(train, normalize=normalize)
    intermediate = layers.Dense(256, activation="relu")(all_features)
    intermediate = layers.Dropout(0.1)(intermediate)
    intermediate = layers.Dense(128, activation="relu")(intermediate)
    intermediate = layers.Dropout(0.1)(intermediate)
    intermediate = layers.Dense(64, activation="relu")(intermediate)
    intermediate = layers.Dropout(0.1)(intermediate)
    intermediate = layers.Dense(32, activation="relu")(intermediate)
    intermediate = layers.Dropout(0.1)(intermediate)
    output = layers.Dense(4, activation="softmax")(intermediate)

    model = Model(input_dict, output)

    model.compile(optimizer='adam', 
        loss=losses.BinaryCrossentropy(), 
        metrics=['accuracy'], 
        run_eagerly=debugging)
    
    return model

def plot_model(model: Model):
    utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir="TB")

def train_model(model: Model, train: tf.data.Dataset, val: tf.data.Dataset):
    model.fit(train, epochs=10, validation_data=val)

def evaluate(model: Model, data: tf.data.Dataset):
    results = model.evaluate(data, return_dict=True)
    print(results)