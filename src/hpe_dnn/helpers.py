from keras import Model, utils
import tensorflow as tf
from numpy import nan, ndarray
from pandas import DataFrame, Series
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

from src.hpe_dnn.balancing import balance_func_factory
from src.hpe_dnn.augmentation import AugmentationPipeline

def plot_model(model: Model):
    utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir="TB")

def demo_batch(dataset: tf.data.Dataset):
    [(features, label_batch)] = dataset.take(1)

    feature_names = list(dataset.element_spec[0].keys())
    print('Every feature:', feature_names)
    print(f'A batch of {feature_names[0]}:', features[feature_names[0]])
    print('A batch of techniques:', label_batch)

def __binarize_labels(labels: Series) -> ndarray:
    encoder = LabelBinarizer()
    return encoder.fit_transform(labels)

def df_to_dataset(dataframe: DataFrame, 
        balance=False,
        augment=False,
        shuffle=True,
        batch_size=32,
        prefetch=True) -> tf.data.Dataset:
    df = dataframe.copy()
    
    if balance:
        balance_func = balance_func_factory(df)
        df = df.apply(balance_func, axis=1)

    if augment:
        aug_pipeline = AugmentationPipeline.for_dataframe(df)
        df = df.apply(aug_pipeline, axis=1)

    labels = df.pop("label")
    _ = df.pop("image_path")
    
    imp = SimpleImputer(missing_values=nan, strategy='mean')
    df = DataFrame(imp.fit_transform(df), columns=df.keys())
    df = {key: value.to_numpy()[:,tf.newaxis] for key, value in df.items()}

    y = __binarize_labels(labels)

    ds = tf.data.Dataset.from_tensor_slices((dict(df), y))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    
    if batch_size:
        ds = ds.batch(batch_size)
    
    if prefetch:
        ds = ds.prefetch(batch_size)
    
    return ds