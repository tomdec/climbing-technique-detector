from pandas import DataFrame
from sklearn.impute import SimpleImputer
from numpy import nan, arange, array, mean, float32, reshape, concatenate, ndarray, sum
from numpy.random import default_rng
from matplotlib import pyplot as plt
import tensorflow as tf
from pandas import Series, concat
from typing import Tuple
from functools import reduce
from keras.api.preprocessing import timeseries_dataset_from_array

from src.labels import get_valid_label_count
from src.hpe_dnn.helpers import binarize_labels, unbinarize_labels
from src.rnn.helpers import (
    get_features,
    get_admin_columns,
    split_df,
    combine_df,
)
from src.rnn.augmentation import AugmentationPipeline


def take_groups(df: DataFrame, groups: list) -> DataFrame:
    filtered = list(map(lambda group: df.query(f"group == {group}"), groups))
    return concat(filtered, axis=0, ignore_index=True)


def split_input_output(data: tf.data.Dataset) -> Tuple[list, list]:
    input = []
    output = []
    for test_batch in data.as_numpy_iterator():
        input.append(test_batch[0])
        output.append(test_batch[1])

    input = concatenate(input, axis=0)
    output = concatenate(output, axis=0)

    return input, output


def output_to_labels(output: list, label_names: list) -> Series:
    output_2d = reshape(output, (-1, get_valid_label_count()))
    output_df = DataFrame(output_2d, columns=label_names)
    return unbinarize_labels(output_df)


class WindowGenerator:

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
    ):
        self._augmentation = None

        df = data.copy()
        features = get_features(df)
        labels_str: Series = df.pop("label")
        admin_cols = get_admin_columns(df)

        labels = binarize_labels(labels_str)
        self.raw_data = combine_df(features, labels, admin_cols)

        # Store input and output column names
        self.input_columns = features.columns
        self.label_columns = labels.columns

        train_df = take_groups(self.raw_data, train_groups)
        train_features = get_features(train_df)
        self.train_mean = train_features.mean(skipna=True)
        self.train_std = train_features.std(skipna=True)

        self.train_groups = train_groups
        self.val_groups = val_groups
        self.test_groups = test_groups

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

    def set_augmentation(self, augmentation: AugmentationPipeline):
        self._augmentation = augmentation

    def get_processed_data(
        self, data: DataFrame, isTraining: bool = False
    ) -> DataFrame:
        if isTraining:
            data = self.augment_features(data)
        data = self.normalize_features(data)
        data = self.impute_features(data)
        return data

    def augment_features(self, data: DataFrame) -> DataFrame:
        if self._augmentation is None:
            return data

        def apply_augmentation_per_group(group: int) -> DataFrame:
            rng = default_rng()
            seed = int(rng.integers(1000))
            print(f"Applying augmentation seed {seed} for group {group}")
            self._augmentation.set_seed(seed)
            group_df = take_groups(data, [group])
            group_df = group_df.apply(self._augmentation, axis=1)
            return group_df

        groups = data["group"].unique()
        group_df_arr = list(map(apply_augmentation_per_group, groups))
        data = concat(group_df_arr, axis=0, ignore_index=True)

        return data

    def normalize_features(self, data: DataFrame) -> DataFrame:
        features, labels, admin = split_df(data)

        features = (features - self.train_mean) / self.train_std

        return combine_df(features, labels, admin)

    def impute_features(self, data: DataFrame) -> DataFrame:
        features, labels, admin = split_df(data)

        imp = SimpleImputer(
            missing_values=nan,
            strategy="constant",
            fill_value=0,
            keep_empty_features=True,
        )
        features = features.copy()
        features = DataFrame(imp.fit_transform(features), columns=features.keys())

        return combine_df(features, labels, admin)

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

        train_count = self.__count_label_frames(train_df)
        val_count = self.__count_label_frames(val_df)
        test_count = self.__count_label_frames(test_df)

        print("\nData splits (train/val/test):")
        for key in train_count.keys():
            total = train_count[key] + val_count[key] + test_count[key]
            print(
                f"{key}: {train_count[key] / total:.1%} / {val_count[key] / total:.1%} / {test_count[key] / total:.1%}"
            )

        print("\nTotals (train/val/test):")
        for key in train_count.keys():
            print(f"{key}: {train_count[key]} / {val_count[key]} / {test_count[key]}")

    @tf.autograph.experimental.do_not_convert
    def split_window(self, batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        inputs: tf.Tensor = batch[:, self.input_slice, :]
        inputs = tf.stack(
            [inputs[:, :, self.column_indices[name]] for name in self.input_columns],
            axis=-1,
        )
        output = batch[:, self.labels_slice, :]
        output = tf.stack(
            [output[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1,
        )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        output.set_shape([None, self.label_width, None])

        return inputs, output

    def get_example(self):
        data = self.get_processed_data(self.raw_data)
        data = data.drop(columns=["video", "frame_num", "group"])
        data = array(data, dtype=float32)

        # Stack three slices, the length of the total window.
        example_batch = tf.stack(
            [
                data[: self.total_window_size],
                data[100 : 100 + self.total_window_size],
                data[200 : 200 + self.total_window_size],
            ]
        )

        example_inputs, example_ouputs = self.split_window(example_batch)

        print("All shapes are: (batch, time, features)")
        print(f"Window shape: {example_batch.shape}")
        print(f"Inputs shape: {example_inputs.shape}")
        print(f"Labels shape: {example_ouputs.shape}")

        return example_inputs, example_ouputs

    def plot(self, model=None, plot_col="NOSE_x", max_subplots=3):
        inputs, labels = self.get_example()
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [norm]")
            feature_values = inputs[n, :, plot_col_index]
            feature_spread = max(feature_values) - min(feature_values)
            # x_axis = inputs[n, :, frame_num_index]
            x_axis = self.input_indices

            plt.xlim((x_axis[0], x_axis[-1] + 1))
            plt.plot(x_axis, feature_values, label="Inputs", marker=".", zorder=-10)

            label_col_index = plot_col_index
            if label_col_index is None:
                continue

            logits = DataFrame(data=array(labels[n, :, :]), columns=self.label_columns)
            label_name = unbinarize_labels(logits)[0]
            label_x_position = x_axis[-1] + 0.1
            label_y_position = mean(feature_values) + feature_spread * 0.2
            plt.text(
                label_x_position,
                label_y_position,
                f"Label: {label_name}",
                c="#2ca02c",
                label="Label",
            )

            if model is not None:
                predictions = model(inputs)
                pred_2d = reshape(predictions, (-1, get_valid_label_count()))
                pred_df = DataFrame(pred_2d, columns=self.label_columns)
                pred_label = unbinarize_labels(pred_df)[n]
                pred_y_position = mean(feature_values) - feature_spread * 0.1
                plt.text(
                    label_x_position,
                    pred_y_position,
                    f"Pred: {pred_label}",
                    c="#ff7f0e",
                    label="Prediction",
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Frame number")

    def make_window_batches(self, data: DataFrame, group: int) -> tf.data.Dataset:
        group_data = data.query(f"group == {group}")
        group_data = group_data.drop(columns=["video", "frame_num", "group"])
        group_arr = array(group_data, dtype=float32)

        return timeseries_dataset_from_array(
            data=group_arr,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            sampling_rate=1,
            shuffle=False,
            batch_size=32,
        )

    def make_ds(self, data: DataFrame) -> tf.data.Dataset:
        groups = data["group"].unique()
        windows = map(lambda group: self.make_window_batches(data, group), groups)
        windows = reduce(tf.data.Dataset.concatenate, windows)
        data_points = windows.map(self.split_window)

        print(f"Generated {len(data_points)} batches")
        print("All shapes are: (batch, time, features)")
        print("Input data:", data_points.element_spec[0])
        print("Output data:", data_points.element_spec[1])

        return data_points

    def get_class_weights(self, verbose: bool = False) -> ndarray:
        class_counts = self.__count_label_frames(self.raw_train_df)
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

    def __count_label_frames(self, df: DataFrame) -> dict:
        return {label: sum(df[label]) for label in self.label_columns}

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
            ]
        )
