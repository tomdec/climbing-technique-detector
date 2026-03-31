from cv2 import imread
from pandas import DataFrame
from numpy import arange, reshape, concatenate, ndarray, sum
import tensorflow as tf
from pandas import Series, concat
from typing import Iterator, Tuple, Callable
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from cv2.typing import MatLike
from os.path import join

from src.common.helpers import get_filename
from src.common.data import take_groups
from src.labels import get_valid_label_count
from src.hpe_dnn.helpers import binarize_labels, unbinarize_labels
from src.rnn.helpers import (
    get_admin_columns,
)
from src.rnn.augmentation import AugmentationPipeline
from src.sampling.images import FRAME_SIZE


def get_frame(row: Series) -> MatLike:
    frame_root = "data/frames"
    video_name = get_filename(row["video"])
    frame_idx = row["frame_num"]
    image = imread(join(frame_root, video_name, f"{frame_idx}.png"))
    image = cvtColor(image, COLOR_BGR2RGB)
    return image


class EvaluationWindowGenerator:

    @property
    def raw_train_df(self) -> DataFrame:
        """Return training data as DataFrame without preprocessing applied.

        Returns:
            DataFrame: Raw training data
        """
        return take_groups(self.raw_data, self.train_groups)

    @property
    def raw_val_df(self) -> DataFrame:
        """Return validation data as DataFrame without preprocessing applied.

        Returns:
            DataFrame: Raw validation data
        """
        return take_groups(self.raw_data, self.val_groups)

    @property
    def raw_test_df(self) -> DataFrame:
        """Return test data as DataFrame without preprocessing applied.

        Returns:
            DataFrame: Raw test data
        """
        return take_groups(self.raw_data, self.test_groups)

    def __init__(
        self, data: DataFrame, train_groups: list, val_groups: list, test_groups: list
    ):
        df = data.copy()
        labels_str: Series = df.pop("label")
        admin_cols = get_admin_columns(df)

        labels = binarize_labels(labels_str)
        self.raw_data = concat([labels, admin_cols], axis=1)
        self.raw_data.insert(0, "frames", None)

        # Store input and output column names
        self.input_columns = ["frames"]
        self.label_columns = labels.columns

        self.train_groups = train_groups
        self.val_groups = val_groups
        self.test_groups = test_groups

    def inspect_fold_split(self):
        print(f"Input features ({len(self.input_columns)}): ", self.input_columns)
        print(f"Output columns ({len(self.label_columns)}): ", self.label_columns)

        print(f"Train: groups={self.train_groups}")
        print(f"Val:  groups={self.val_groups}")
        print(f"Test:  groups={self.test_groups}")

        train_df = self.raw_train_df
        val_df = self.raw_val_df
        test_df = self.raw_test_df
        print("\nAll shapes are: (frames, features)")
        print(f"Training data: {train_df.shape}")
        print(f"Val data: {val_df.shape}")
        print(f"Test data: {test_df.shape}")

        train_count = self._count_label_frames(train_df)
        val_count = self._count_label_frames(val_df)
        test_count = self._count_label_frames(test_df)

        print("\nData splits (train/val/test):")
        for key in train_count.keys():
            total = train_count[key] + val_count[key] + test_count[key]
            print(
                f"{key}: {train_count[key] / total:.1%} / "
                + f"{val_count[key] / total:.1%} / "
                + f"{test_count[key] / total:.1%}"
            )

        print("\nTotals (train/val/test):")
        for key in train_count.keys():
            print(f"{key}: {train_count[key]} / {val_count[key]} / {test_count[key]}")

    def get_label_counts(self) -> Tuple[dict, dict, dict]:
        train_count = self._count_label_frames(self.raw_train_df)
        val_count = self._count_label_frames(self.raw_val_df)
        test_count = self._count_label_frames(self.raw_test_df)

        return (train_count, val_count, test_count)

    def _count_label_frames(self, df: DataFrame) -> dict:
        return {label: sum(df[label]) for label in self.label_columns}


class WindowGenerator(EvaluationWindowGenerator):

    @property
    def train_df(self) -> DataFrame:
        """Return training data as DataFrame with preprocessing applied.

        Returns:
            DataFrame: Training data
        """
        return self.get_processed_data(self.raw_train_df, isTraining=True)

    @property
    def val_df(self) -> DataFrame:
        """Return validation data as DataFrame with preprocessing applied.

        Returns:
            DataFrame: Validation data
        """
        return self.get_processed_data(self.raw_val_df)

    @property
    def test_df(self) -> DataFrame:
        """Return test data as DataFrame with preprocessing applied.

        Returns:
            DataFrame: Test data
        """
        return self.get_processed_data(self.raw_test_df)

    @property
    def train_ds(self) -> tf.data.Dataset:
        """Return processed training data as Tensorflow Dataset, structured how the Tensorflow model
        expects it during training.

        Returns:
            tf.data.Dataset: Training dataset.
        """
        return self.make_ds(self.train_df)

    @property
    def val_ds(self) -> tf.data.Dataset:
        """Return processed validation data as Tensorflow Dataset, structured how the Tensorflow
        model expects it during training.

        Returns:
            tf.data.Dataset: Validation dataset.
        """
        return self.make_ds(self.val_df)

    @property
    def test_ds(self) -> tf.data.Dataset:
        """Return processed test data as Tensorflow Dataset, structured how the Tensorflow
        model expects it during testing.

        Returns:
            tf.data.Dataset: Test dataset.
        """
        return self.make_ds(self.test_df)

    @property
    def augmentation(self) -> AugmentationPipeline | None:
        return self._augmentation

    def __init__(
        self,
        data: DataFrame,
        train_groups: list,
        val_groups: list,
        test_groups: list,
        input_width: int,
        spacing: int = 1,
        batch_size: int = 32,
    ):
        super().__init__(data, train_groups, val_groups, test_groups)

        self._augmentation = None

        # Work out the label column indices.
        self.column_indices = {name: i for i, name in enumerate(self.raw_data.columns)}
        self.input_column_indices = {
            name: i for i, name in enumerate(self.input_columns)
        }
        self.label_columns_indices = {
            name: i for i, name in enumerate(self.label_columns)
        }

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = 1
        self.spacing = spacing

        self.total_window_size = (input_width - 1) * spacing + 1

        self.input_slice = slice(0, input_width * spacing, spacing)
        self.input_indices = arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = arange(self.total_window_size)[self.labels_slice]

        self.batch_size = batch_size

    def get_processed_data(
        self, data: DataFrame, isTraining: bool = False
    ) -> DataFrame:
        return data

    @tf.autograph.experimental.do_not_convert
    def split_window(self, window: DataFrame) -> Tuple[tf.Tensor, tf.Tensor]:
        df_frames = window.apply(get_frame, axis=1)
        inputs = tf.stack(df_frames.values)

        output = window[self.labels_slice]
        output = tf.stack(
            [output[name] for name in self.label_columns],
            axis=-1,
        )
        output.set_shape([None, get_valid_label_count()])

        return inputs, output

    def make_window_batches(self, data: DataFrame, group: int) -> Iterator[DataFrame]:
        group_data = data.query(f"group == {group}")
        group_len = len(group_data)
        start_points = arange(0, group_len - self.input_width + 1)
        slices = [slice(start, start + self.input_width) for start in start_points]
        for slc in slices:
            yield group_data[slc]

    def get_generator(self, data: DataFrame) -> Callable[[], Iterator]:

        def generator() -> Iterator:
            groups = data["group"].unique()
            for group in groups:
                windows_iter = self.make_window_batches(data, group)
                for window in windows_iter:
                    yield self.split_window(window)

        return generator

    def get_signature(self) -> Tuple[tf.TensorSpec, tf.TensorSpec]:
        return (
            tf.TensorSpec(shape=(self.input_width, FRAME_SIZE, FRAME_SIZE, 3)),
            tf.TensorSpec(shape=(self.label_width, get_valid_label_count())),
        )

    def make_ds(self, data: DataFrame) -> tf.data.Dataset:

        ds = tf.data.Dataset.from_generator(
            generator=self.get_generator(data), output_signature=self.get_signature()
        )
        ds = ds.batch(self.batch_size)
        return ds

    def get_class_weights(self, verbose: bool = False) -> ndarray:
        class_counts = self._count_label_frames(self.raw_train_df)
        if verbose:
            print("Class counts:\n", class_counts)

        count_list = list(class_counts.values())
        class_weights = sum(count_list) / count_list
        if verbose:
            print(
                [
                    f"{name}: {weight}"
                    for (name, weight) in zip(self.label_columns, class_weights)
                ]
            )

        return class_weights

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
            ]
        )
