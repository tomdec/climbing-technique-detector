import shutil
import tensorflow as tf
from typing import override, List
from os import makedirs, listdir, mkdir
from os.path import join
from wandb.sdk import init, finish
from keras.api.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.api.models import load_model, Sequential
from keras.api.callbacks import Callback
from wandb.data_types import Image
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from glob import glob

from src.common.helpers import (
    get_current_train_run,
    get_next_test_run,
    make_file,
)
from src.common.model import (
    ClassificationModel,
    ModelConstructorArgs,
    ModelInitializeArgs,
    TrainArgs,
    MultiRunTrainArgs,
    TestArgs,
    get_best_tf_weights,
)
from src.common.plot import plot_confusion_matrix
from src.rnn.data import split_input_output, WindowGenerator, output_to_labels
from src.rnn.architecture import get_model, RnnArch


class RnnModelInitializeArgs(ModelInitializeArgs):

    @override
    @property
    def model_arch(self) -> RnnArch:
        """Enum that is mapped to a factory function"""
        return self._model_arch

    @override
    def __init__(self, model_arch: RnnArch = RnnArch.ARCH1):
        super().__init__(model_arch)


class RnnConstructorArgs(ModelConstructorArgs):

    @override
    @property
    def model_initialize_args(self) -> RnnModelInitializeArgs:
        return self._model_initialize_args

    @override
    def __init__(
        self,
        name: str,
        model_initialize_args: RnnModelInitializeArgs = RnnModelInitializeArgs(),
        data_root_path="data",
        dataset_name="techniques",
    ):
        super().__init__(name, model_initialize_args, data_root_path, dataset_name)

    @override
    def copy_with(self, name=None, dataset_name=None) -> "RnnConstructorArgs":
        return RnnConstructorArgs(
            name=self.name if name is None else name,
            model_initialize_args=self.model_initialize_args,
            data_root_path=self.data_root_path,
            dataset_name=self.dataset_name if dataset_name is None else dataset_name,
        )


class RnnTrainArgs(TrainArgs):

    @override
    @property
    def balanced(self) -> bool:
        """Rnn models are never trained on balanced data."""
        return False

    @property
    def window_generator(self) -> WindowGenerator:
        return self._window_generator

    def __init__(
        self,
        window_generator: WindowGenerator,
        epochs: int = 10,
        additional_config: dict = {},
    ):
        super().__init__(epochs, False, additional_config)
        self._window_generator = window_generator


class RnnTestArgs(TestArgs):

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


class RnnMultiRunTrainArgs(MultiRunTrainArgs):

    @override
    @property
    def train_args(self) -> RnnTrainArgs:
        return self._train_args

    @override
    def __init__(self, train_args: RnnTrainArgs, runs=5):
        super().__init__(runs, train_args)


class Rnn(ClassificationModel):

    @override
    @property
    def model_initialize_args(self) -> RnnModelInitializeArgs:
        return self._model_initialize_args

    @override
    @property
    def model_arch(self) -> RnnArch:
        """Enum that is mapped to a factory function"""
        return self.model_initialize_args.model_arch

    @property
    def model(self) -> Sequential | None:
        return self._model

    @override
    def __init__(self, args: RnnConstructorArgs):
        super().__init__(args)
        self._model = None

    @override
    def execute_train_runs(self, args: RnnMultiRunTrainArgs):
        return super().execute_train_runs(args)

    @override
    def train_model(self, args: RnnTrainArgs):
        if self.model is None:
            raise Exception("Cannot train before model is initialized")

        train_ds = args.window_generator.train_ds
        val_ds = args.window_generator.val_ds

        checkpoint_dir = self.__get_checkpoint_dir()
        log_dir = self.__get_tensorboard_log_dir()
        results_file = self.__get_results_file_path()

        wandb_config = self.__get_train_wandb_config(args)
        init(
            project="detect-climbing-technique",
            job_type="train",
            group="rnn",
            name=self.name,
            config=wandb_config,
            dir=self.data_root_path,
        )

        makedirs(checkpoint_dir)
        makedirs(log_dir)
        make_file(results_file)
        try:
            checkpoint_path = join(
                checkpoint_dir,
                "epoch_{epoch:02d}__val_accuracy_{val_categorical_accuracy:.4f}.keras",
            )
            cp_callback = ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                monitor="val_categorical_accuracy",
            )

            csv_callback = CSVLogger(filename=results_file)

            early_stopping_callback = EarlyStopping(
                monitor="val_categorical_accuracy", patience=3
            )

            self.model.fit(
                train_ds,
                epochs=args.epochs,
                validation_data=val_ds,
                callbacks=[
                    cp_callback,
                    csv_callback,
                    early_stopping_callback,
                    WandbMetricsLogger(),
                    WandbModelCheckpoint(join(log_dir, "wandb.keras")),
                ],
            )

        except Exception as e:
            # remove result files
            shutil.rmtree(self._get_current_train_dir())
            raise e
        finally:
            finish()

    @override
    def test_model(self, args: RnnTestArgs):
        self._load_best_model()

        model_path = self._get_model_dir()
        test_run = get_next_test_run(model_path)
        test_run_path = join(model_path, test_run)
        mkdir(test_run_path)

        test_ds = args.window_generator.test_ds

        input, labels = split_input_output(test_ds)
        labels = output_to_labels(labels, args.window_generator.label_columns)
        predictions = self.model(input)
        predictions = output_to_labels(predictions, args.window_generator.label_columns)

        plot_confusion_matrix(
            labels.values,
            predictions.values,
            save_path=join(test_run_path, "confusion_matrix.png"),
            normalized=False,
        )
        plot_confusion_matrix(
            labels,
            predictions,
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

    def __evaluate_with_wandb(
        self, args: TestArgs, data: tf.data.Dataset, test_run_path: str
    ) -> dict:

        config = self.__get_test_wandb_config(args)
        wandb_run = init(
            project="detect-climbing-technique",
            job_type="test",
            group="rnn",
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
        return join(self.data_root_path, "runs", "rnn", self.name)

    @override
    def _get_best_model_path(self):
        model_dir = self._get_model_dir()
        weight_paths = glob(join(model_dir, "*", "models", "*.keras"))
        return get_best_tf_weights(weight_paths)

    @override
    def _fresh_model(self):
        print(f"loading a fresh model '{self.model_arch}'")
        self._model = get_model(self.model_arch)

    @override
    def _load_model(self, best_model_path):
        print(f"loading the model '{self.name}' from '{best_model_path}'")
        self._model = load_model(best_model_path)

    def __get_train_wandb_config(self, args: RnnTrainArgs) -> dict:
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
