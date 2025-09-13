from enum import Enum
from typing import Mapping, Callable, List
from keras import Model, layers, losses, metrics, optimizers, Input
import tensorflow as tf

from src.labels import get_valid_label_count

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
    # these are all the input features, they're all numeric
    numeric_features = list(map(lambda x: f"{x}", train.element_spec[0].keys()))
    
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

def __arch_factory(train: tf.data.Dataset,
        nodes: List[int],
        normalize: bool,
        debugging: bool,
        dropout_rate: float,
        activation: str) -> Model:
    
    input_dict, all_features = __make_input_layer(train, normalize=normalize)
    
    intermediate = layers.Dense(nodes[0], activation=activation)(all_features)
    intermediate = layers.Dropout(dropout_rate)(intermediate)
    
    for node in nodes[1:]:    
        intermediate = layers.Dense(node, activation=activation)(intermediate)
        intermediate = layers.Dropout(dropout_rate)(intermediate)
    
    output = layers.Dense(get_valid_label_count(), activation="softmax")(intermediate)

    model = Model(input_dict, output)
    
    model.compile(optimizer=optimizers.Adam(),
        loss=losses.CategoricalCrossentropy(), 
        metrics=[metrics.CategoricalAccuracy()], 
        run_eagerly=debugging)
    
    return model

def __arch1_factory(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch1.
        4 layers, starting from 256 nodes, and each layer halving in size.
        Don't change, make new DnnArch and function.
    """

    return __arch_factory(train=train, 
        nodes=[256, 128, 64, 32], 
        normalize=normalize, 
        debugging=debugging, 
        dropout_rate=dropout_rate, 
        activation="relu")

def __arch2_factory(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch2.
        5 layers, starting from 512 nodes, and each layer halving in size.
        Don't change, make new DnnArch and function.
    """

    return __arch_factory(train=train, 
        nodes=[512, 256, 128, 64, 32], 
        normalize=normalize, 
        debugging=debugging, 
        dropout_rate=dropout_rate, 
        activation="relu")

def __arch3_factory(train: tf.data.Dataset, 
        normalize=True, 
        debugging=False,
        dropout_rate=0.1) -> Model:
    """
        Arch3.
        Symmetrical V-shaped network of 7 layers, with the middle layer having 256 nodes.
        Don't change, make new DnnArch and function.
    """

    return __arch_factory(train=train, 
        nodes=[32, 64, 128, 256, 128, 64, 32], 
        normalize=normalize, 
        debugging=debugging, 
        dropout_rate=dropout_rate, 
        activation="relu")

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

    return __arch_factory(train=train, 
        nodes=[128, 256, 128, 64, 32], 
        normalize=normalize, 
        debugging=debugging, 
        dropout_rate=dropout_rate, 
        activation="relu")

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

    return __arch_factory(train=train, 
        nodes=[128, 128, 128, 64, 32], 
        normalize=normalize, 
        debugging=debugging, 
        dropout_rate=dropout_rate, 
        activation="relu")

_arch_mapping: Mapping[DnnArch, Callable[[tf.data.Dataset, bool, bool, float], Model]] = {
    DnnArch.ARCH1: __arch1_factory,
    DnnArch.ARCH2: __arch2_factory,
    DnnArch.ARCH3: __arch3_factory,
    DnnArch.ARCH4: __arch4_factory,
    DnnArch.ARCH5: __arch5_factory
}

def get_model_factory(architecture: DnnArch):
    return _arch_mapping[architecture]

