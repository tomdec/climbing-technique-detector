from keras import Model, utils
import tensorflow as tf
from numpy import nan
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

from src.hpe_dnn.balancing import balance_func_factory
from src.hpe_dnn.augmentation import augment_keypoints

def plot_model(model: Model):
    utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir="TB")

def demo_batch(dataset: tf.data.Dataset):
    [(features, label_batch)] = dataset.take(1)

    print('Every feature:', list(features.keys()))
    print('A batch of Nose x-coordinates:', features['NOSE_x'])
    print('A batch of techniques:', label_batch )

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
