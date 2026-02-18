from idna import encode
from keras import Model, utils
import tensorflow as tf
from numpy import nan, ndarray
from pandas import DataFrame, Series
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

from src.labels import iterate_valid_labels, value_to_name
from src.hpe_dnn.balancing import balance_func_factory
from src.hpe_dnn.augmentation import AugmentationPipeline


def plot_model(model: Model):
    utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir="TB")


def demo_batch(dataset: tf.data.Dataset):
    [(features, label_batch)] = dataset.take(1)

    feature_names = list(dataset.element_spec[0].keys())
    print("Every feature:", feature_names)
    print(f"A batch of {feature_names[0]}:", features[feature_names[0]])
    print("A batch of techniques:", label_batch)


def __encoder() -> LabelBinarizer:
    encoder = LabelBinarizer()
    encoder.fit(list(iterate_valid_labels()))
    return encoder


def binarize_labels(labels: Series) -> DataFrame:
    if labels.dtype == "int64":
        labels = labels.map(value_to_name)
    encoder = __encoder()
    labels_bin = encoder.transform(labels)
    return DataFrame(data=labels_bin, columns=encoder.classes_)


def unbinarize_labels(logits: DataFrame) -> Series:
    encoder = __encoder()
    if not all(logits.columns == encoder.classes_):
        raise Exception(
            f"Unexpected columns of logits ({logits.columns}), expected: {encoder.classes_}"
        )

    values = encoder.inverse_transform(logits.values)
    return Series(data=values, name="label")


def df_to_dataset(
    dataframe: DataFrame,
    balance=False,
    augment=False,
    shuffle=True,
    batch_size=32,
    prefetch=True,
) -> tf.data.Dataset:
    df = dataframe.copy()

    if balance:
        balance_func = balance_func_factory(df)
        df = df.apply(balance_func, axis=1)

    if augment:
        aug_pipeline = AugmentationPipeline.for_dataframe(df)
        df = df.apply(aug_pipeline, axis=1)

    labels = df.pop("label")
    _ = df.pop("image_path")

    imp = SimpleImputer(
        missing_values=nan, strategy="constant", fill_value=0, keep_empty_features=True
    )
    df = DataFrame(imp.fit_transform(df), columns=df.keys())
    df = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}

    y = binarize_labels(labels).values

    ds = tf.data.Dataset.from_tensor_slices((dict(df), y))

    if shuffle:
        ds = ds.shuffle(buffer_size=ds.cardinality(), reshuffle_each_iteration=True)

    if batch_size:
        ds = ds.batch(batch_size)

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
