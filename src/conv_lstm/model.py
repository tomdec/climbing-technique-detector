import shutil
from numpy import concatenate
import tensorflow as tf
from typing import override, List
from os import makedirs, mkdir
from os.path import join
from wandb.sdk import init, finish
from keras.api.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.api.models import load_model, Sequential
from keras.api.losses import CategoricalCrossentropy
from wandb.data_types import Image
from wandb.integration.keras import WandbMetricsLogger
from glob import glob

from src.common.helpers import (
    get_next_test_run,
    make_file,
)
from src.common.data import output_to_labels
from src.common.model import (
    ClassificationModel,
    ModelConstructorArgs,
    ModelInitializeArgs,
    TrainArgs,
    MultiRunTrainArgs,
    TestArgs,
    get_best_tf_weights,
    DEFAULT_DATASET,
)
from src.common.plot import plot_confusion_matrix
from src.common.wandb import PROJECT_NAME
from src.common.data import split_input_output
from src.common.model import weighted_categorical_cross_entropy
from src.conv_lstm.architecture import ConvLstmArch, get_model
from src.conv_lstm.data import WindowGenerator


class ConvLstmModelInitializeArgs(ModelInitializeArgs):

    @override
    @property
    def model_arch(self) -> ConvLstmArch:
        """Enum that is mapped to a factory function"""
        return self._model_arch

    @property
    def input_width(self) -> int:
        return self._input_width

    @property
    def spacing(self) -> int:
        return self._spacing

    @override
    def __init__(
        self,
        model_arch: ConvLstmArch = ConvLstmArch.ARCH1,
        input_width: int = 5,
        spacing: int = 1,
    ):
        super().__init__(model_arch)
        self._input_width = input_width
        self._spacing = spacing


class ConvLstmConstructorArgs(ModelConstructorArgs):

    @override
    @property
    def model_initialize_args(self) -> ConvLstmModelInitializeArgs:
        return self._model_initialize_args

    @override
    def __init__(
        self,
        name: str,
        model_initialize_args: ConvLstmModelInitializeArgs = ConvLstmModelInitializeArgs(),
        data_root_path="data",
        dataset_name=DEFAULT_DATASET,
    ):
        super().__init__(name, model_initialize_args, data_root_path, dataset_name)

    @override
    def copy_with(self, name=None, dataset_name=None) -> "ConvLstmConstructorArgs":
        return ConvLstmConstructorArgs(
            name=self.name if name is None else name,
            model_initialize_args=self.model_initialize_args,
            data_root_path=self.data_root_path,
            dataset_name=self.dataset_name if dataset_name is None else dataset_name,
        )


class ConvLstmIntTrainArgs(TrainArgs):

    def __init__(
        self,
        epochs: int = 10,
        balanced: bool = False,
        additional_config: dict = {},
    ):
        super().__init__(epochs, balanced, additional_config)


class ConvLstmTrainArgs(ConvLstmIntTrainArgs):

    @staticmethod
    def from_intermediate(
        wg: WindowGenerator, args: ConvLstmIntTrainArgs
    ) -> "ConvLstmTrainArgs":
        return ConvLstmTrainArgs(
            window_generator=wg,
            epochs=args.epochs,
            balanced=args.balanced,
            additional_config=args.additional_config,
        )

    @property
    def window_generator(self) -> WindowGenerator:
        return self._window_generator

    def __init__(
        self,
        window_generator: WindowGenerator,
        epochs: int = 10,
        balanced: bool = False,
        additional_config: dict = {},
    ):
        super().__init__(epochs, balanced, additional_config)
        self._window_generator = window_generator


class ConvLstmTestArgs(TestArgs):

    @property
    def window_generator(self) -> WindowGenerator:
        return self._window_generator

    @override
    def __init__(
        self,
        window_generator: WindowGenerator,
        write_to_wandb: bool = False,
        additional_config: dict = {},
    ):
        super().__init__(write_to_wandb, additional_config)
        self._window_generator = window_generator


class ConvLstmIntMultiRunTrainArgs(MultiRunTrainArgs):

    @override
    @property
    def train_args(self) -> ConvLstmIntTrainArgs:
        return self._train_args

    @override
    def __init__(self, train_args: ConvLstmIntTrainArgs, runs=5):
        super().__init__(runs, train_args)


class ConvLstmMultiRunTrainArgs(MultiRunTrainArgs):

    @staticmethod
    def from_intermediate(
        wg: WindowGenerator, args: ConvLstmIntMultiRunTrainArgs
    ) -> "ConvLstmMultiRunTrainArgs":
        return ConvLstmMultiRunTrainArgs(
            train_args=ConvLstmTrainArgs.from_intermediate(wg, args.train_args),
            runs=args.runs,
        )

    @override
    @property
    def train_args(self) -> ConvLstmTrainArgs:
        return self._train_args

    @override
    def __init__(self, train_args: ConvLstmTrainArgs, runs=5):
        super().__init__(runs, train_args)


class ConvLstm(ClassificationModel):

    MODEL_TYPE = "conv_lstm"

    @override
    @property
    def model_initialize_args(self) -> ConvLstmModelInitializeArgs:
        return self._model_initialize_args

    @override
    @property
    def model_arch(self) -> ConvLstmArch:
        """Enum that is mapped to a factory function"""
        return self.model_initialize_args.model_arch

    @property
    def model(self) -> Sequential | None:
        return self._model

    @property
    def loss_function(self):
        return self._loss_function

    @override
    def __init__(self, args: ConvLstmConstructorArgs):
        super().__init__(args)
        self._model = None
        self._weights = None
        self._loss_function = CategoricalCrossentropy()

    @override
    def execute_train_runs(self, args: ConvLstmMultiRunTrainArgs):
        if args.train_args.balanced:
            weights = args.train_args.window_generator.get_class_weights()
            self._loss_function = weighted_categorical_cross_entropy(weights)
            print("Using weighted loss function to achieve balancing")
        return super().execute_train_runs(args)

    @override
    def train_model(self, args: ConvLstmTrainArgs):
        if self.model is None:
            raise Exception("Cannot train before model is initialized")

        if args.balanced and type(self.loss_function) == CategoricalCrossentropy:
            raise Exception("Unexpected loss function for balanced training.")

        train_ds = args.window_generator.train_ds
        val_ds = args.window_generator.val_ds

        checkpoint_dir = self.__get_checkpoint_dir()
        log_dir = self.__get_tensorboard_log_dir()
        results_file = self.__get_results_file_path()

        wandb_config = self.__get_train_wandb_config(args)
        init(
            project=PROJECT_NAME,
            job_type="train",
            group=self.MODEL_TYPE,
            name=self.name,
            config=wandb_config,
            dir=self.data_root_path,
        )

        makedirs(checkpoint_dir)
        makedirs(log_dir)
        make_file(results_file)
        try:
            cp_callback = self.__get_checkpoint_callback(checkpoint_dir, args)
            csv_callback = CSVLogger(filename=results_file)
            early_stopping_callback = self.__get_early_stopping_callback(args)

            self.model.fit(
                train_ds,
                epochs=args.epochs,
                steps_per_epoch=100,
                validation_data=val_ds,
                shuffle=False,
                callbacks=[
                    cp_callback,
                    csv_callback,
                    early_stopping_callback,
                    WandbMetricsLogger(),
                    # WandbModelCheckpoint(join(log_dir, "wandb.keras")),
                ],
            )

        except Exception as e:
            # remove result files
            shutil.rmtree(self._get_current_train_dir())
            raise e
        finally:
            finish()

    @override
    def test_model(self, args: ConvLstmTestArgs):
        # reset to default loss function
        self._loss_function = CategoricalCrossentropy()
        self._load_best_model()

        model_path = self._get_model_dir()
        test_run = get_next_test_run(model_path)
        test_run_path = join(model_path, test_run)
        mkdir(test_run_path)

        test_ds = args.window_generator.test_ds

        all_labels = []
        all_predictions = []
        for input, labels in test_ds.as_numpy_iterator():
            labels = output_to_labels(labels, args.window_generator.label_columns)
            all_labels.append(labels.values)

            output = self.model(input)
            predictions = output_to_labels(output, args.window_generator.label_columns)
            all_predictions.append(predictions.values)
        all_labels = concatenate(all_labels, axis=0)
        all_predictions = concatenate(all_predictions, axis=0)

        plot_confusion_matrix(
            all_labels,
            all_predictions,
            save_path=join(test_run_path, "confusion_matrix.png"),
            normalized=False,
        )
        plot_confusion_matrix(
            all_labels,
            all_predictions,
            save_path=join(test_run_path, "confusion_matrix_normalized.png"),
            normalized=True,
        )

        if args.write_to_wandb:
            return self.__evaluate_with_wandb(args, test_ds, test_run_path)
        else:
            return self.evaluate(test_ds)

    @override
    def get_test_accuracy_metric(self) -> float:
        return self.get_test_metrics()["categorical_accuracy"]

    def evaluate(self, data: tf.data.Dataset, callbacks: List[Callback] = []) -> dict:
        if self.model is None:
            self._load_best_model()

        performance = self.model.evaluate(data, return_dict=True, callbacks=callbacks)
        self._save_test_metrics(performance)
        return performance

    def __get_checkpoint_callback(
        self, checkpoint_dir: str, train_args: ConvLstmTrainArgs
    ) -> ModelCheckpoint:

        file_name = (
            "epoch_{epoch:02d}__val_accuracy_{val_categorical_accuracy:.4f}.keras"
            if not train_args.balanced
            else "epoch_{epoch:02d}__val_loss_{val_loss:.4f}.keras"
        )
        metric_to_monitor = (
            "val_categorical_accuracy" if not train_args.balanced else "val_loss"
        )
        mode = "max" if not train_args.balanced else "min"

        checkpoint_path = join(checkpoint_dir, file_name)
        cp_callback = ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            monitor=metric_to_monitor,
            mode=mode,
        )
        return cp_callback

    def __get_early_stopping_callback(
        self, train_args: ConvLstmTrainArgs
    ) -> EarlyStopping:
        metric_to_monitor = (
            "val_categorical_accuracy" if not train_args.balanced else "val_loss"
        )
        mode = "max" if not train_args.balanced else "min"
        return EarlyStopping(monitor=metric_to_monitor, patience=3, mode=mode)

    def __evaluate_with_wandb(
        self, args: TestArgs, data: tf.data.Dataset, test_run_path: str
    ) -> dict:

        config = self.__get_test_wandb_config(args)
        wandb_run = init(
            project=PROJECT_NAME,
            job_type="test",
            group=self.MODEL_TYPE,
            name=self.name,
            config=config,
            dir=self.data_root_path,
        )
        try:
            performance = self.evaluate(data, callbacks=[WandbMetricsLogger()])
            wandb_run.log(performance)
            wandb_run.log(
                {
                    "confusion_matrix": Image(
                        join(test_run_path, "confusion_matrix.png")
                    ),
                    "confusion_matrix_normalized": Image(
                        join(test_run_path, "confusion_matrix_normalized.png")
                    ),
                }
            )
            return performance
        finally:
            finish()

    @override
    def _get_model_dir(self):
        return join(self.data_root_path, "runs", self.MODEL_TYPE, self.name)

    @override
    def _get_best_model_path(self):
        model_dir = self._get_model_dir()
        weight_paths = glob(join(model_dir, "*", "models", "*.keras"))
        return get_best_tf_weights(weight_paths)

    @override
    def _fresh_model(self):
        print(f"loading a fresh model '{self.model_arch}'")
        self._model = get_model(self.model_arch, self.loss_function)

    @override
    def _load_model(self, best_model_path):
        print(f"loading the model '{self.name}' from '{best_model_path}'")
        self._model = load_model(
            best_model_path,
            custom_objects={"loss": self.loss_function},
        )

    @override
    def _get_common_wandb_config(self) -> dict:
        return super()._get_common_wandb_config() | {
            "input_width": self.model_initialize_args.input_width,
            "spacing": self.model_initialize_args.spacing,
        }

    def __get_train_wandb_config(self, args: ConvLstmTrainArgs) -> dict:
        return (
            self._get_common_wandb_config()
            | {
                "balanced": args.balanced,
                "augmented": False,
                "run": self._get_next_train_run(),
            }
            | args.additional_config
        )

    def __get_test_wandb_config(self, args: TestArgs) -> dict:
        return (
            self._get_common_wandb_config()
            | {
                "balanced": False,
                "augmented": False,
                "run": self._get_current_train_run(),
            }
            | args.additional_config
        )

    def __get_checkpoint_dir(self):
        train_dir = self._get_next_train_dir()
        return join(train_dir, "models")

    def __get_tensorboard_log_dir(self):
        train_dir = self._get_next_train_dir()
        return join(train_dir, "logs")

    def __get_results_file_path(self):
        train_dir = self._get_next_train_dir()
        return join(train_dir, "results.csv")
