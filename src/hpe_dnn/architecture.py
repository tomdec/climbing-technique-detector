from enum import Enum
from typing import Mapping
from typing import Callable
from keras import Model, layers, losses, metrics, optimizers, Input
import tensorflow as tf

class DnnArch(Enum):
    ARCH1 = 0
    ARCH2 = 1
    ARCH3 = 2
    ARCH4 = 3
    ARCH5 = 4

def __get_normalization_layer(name, dataset):
    # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, _: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer

def __get_category_encoding_layer(name, dataset, max_tokens=None):
  
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

def __make_input_layer(train: tf.data.Dataset, normalize=True):
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
            normalization_layer = __get_normalization_layer(header, train)
            normalized_numerical_col = normalization_layer(numerical_col)
            encoded_feature.append(normalized_numerical_col)
        else:
            encoded_feature.append(numerical_col)

    return all_inputs, layers.concatenate(encoded_feature)

def __arch1_factory(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch1.
        4 layers, starting from 256 nodes, and each layer halving in size.
        Don't change, make new DnnArch and function.
    """

    input_dict, all_features = __make_input_layer(train, normalize=normalize)
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
    
    model.compile(optimizer=optimizers.Adam(),
        loss=losses.CategoricalCrossentropy(), 
        metrics=[metrics.CategoricalAccuracy()], 
        run_eagerly=debugging)

    return model

def __arch2_factory(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch2.
        5 layers, starting from 512 nodes, and each layer halving in size.
        Don't change, make new DnnArch and function.
    """

    input_dict, all_features = __make_input_layer(train, normalize=normalize)
    intermediate = layers.Dense(512, activation="relu")(all_features)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(256, activation="relu")(intermediate)
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
        loss=losses.CategoricalCrossentropy(), 
        metrics=[metrics.CategoricalAccuracy()], 
        run_eagerly=debugging)
    
    return model

def __arch3_factory(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch3.
        Symmetrical V-shaped network of 7 layers, with the middle layer having 256 nodes.
        Don't change, make new DnnArch and function.
    """

    input_dict, all_features = __make_input_layer(train, normalize=normalize)
    intermediate = layers.Dense(32, activation="relu")(all_features)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(64, activation="relu")(intermediate)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(128, activation="relu")(intermediate)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(256, activation="relu")(intermediate)
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
        loss=losses.CategoricalCrossentropy(), 
        metrics=[metrics.CategoricalAccuracy()], 
        run_eagerly=debugging)
    
    return model

def __arch4_factory(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch4.
        Assymmetrical V-shaped network of 5 layers, with the middle layer having 256 nodes.
        Starting at a layer with 128 nodes, so that we don't have less nodes in the first layer
        than we have features. Only reduce dimensionality in the second half of the model.
        Don't change, make new DnnArch and function.
    """

    input_dict, all_features = __make_input_layer(train, normalize=normalize)
    intermediate = layers.Dense(128, activation="relu")(all_features)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(256, activation="relu")(intermediate)
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
        loss=losses.CategoricalCrossentropy(), 
        metrics=[metrics.CategoricalAccuracy()], 
        run_eagerly=debugging)
    
    return model

def __arch5_factory(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch5.
        Architecture with two parts, the first part has layers of a consistent size.
        In the second part, each layer has half the nodes of the previous layer, down to 32 nodes.
        This specific instance, has 3 layers of 128 nodes for the first part.

        Don't change, make new DnnArch and function.
    """

    input_dict, all_features = __make_input_layer(train, normalize=normalize)
    intermediate = layers.Dense(128, activation="relu")(all_features)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    intermediate = layers.Dense(128, activation="relu")(intermediate)
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
        loss=losses.CategoricalCrossentropy(), 
        metrics=[metrics.CategoricalAccuracy()], 
        run_eagerly=debugging)
    
    return model

_arch_mapping: Mapping[DnnArch, Callable[[tf.data.Dataset, bool, bool, float], Model]] = {
    DnnArch.ARCH1: __arch1_factory,
    DnnArch.ARCH2: __arch2_factory,
    DnnArch.ARCH3: __arch3_factory,
    DnnArch.ARCH4: __arch4_factory,
    DnnArch.ARCH5: __arch5_factory
}

def get_model_factory(architecture: DnnArch):
    return _arch_mapping[architecture]

