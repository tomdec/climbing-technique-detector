from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.metrics import DetMetrics
from os.path import join
from os import rename
from typing import Optional, override
from wandb.sdk import init, finish
from wandb.data_types import Image
from wandb.integration.ultralytics import add_wandb_callback
from glob import glob

from src.common.helpers import get_current_train_run
from src.labels import get_label_value_from_path, value_to_name
from src.common.model import (
    ModelConstructorArgs,
    ModelInitializeArgs,
    TestArgs,
    TrainArgs,
    MultiRunTrainArgs,
    ClassificationModel,
)
from src.common.plot import plot_confusion_matrix
from src.sota.balancing import WeightedTrainer


class SOTAModelInitializeArgs(ModelInitializeArgs):

    @override
    @property
    def model_arch(self) -> str:
        """Name of yolo model (recognized by ultralytics) to use when initializing the model"""
        return self._model_arch

    def __init__(self, model_arch: str = "yolo11n-cls"):
        super().__init__(model_arch)


class SOTAConstructorArgs(ModelConstructorArgs):

    @override
    @property
    def model_initialize_args(self) -> SOTAModelInitializeArgs:
        return self._model_initialize_args

    def __init__(
        self,
        name: str,
        model_initialize_args: SOTAModelInitializeArgs = SOTAModelInitializeArgs(),
        data_root_path: str = "data",
        dataset_name: str = "techniques",
    ):
        super().__init__(name, model_initialize_args, data_root_path, dataset_name)

    @override
    def copy_with(self, name=None, dataset_name=None) -> "SOTAConstructorArgs":
        return SOTAConstructorArgs(
            name=self.name if name is None else name,
            model_initialize_args=self.model_initialize_args,
            data_root_path=self.data_root_path,
            dataset_name=self.dataset_name if dataset_name is None else dataset_name,
        )


class SOTATrainArgs(TrainArgs):

    @property
    def optimizer(self) -> str:
        """The optimizer to use during training"""
        return self._optimizer

    @property
    def lr0(self) -> float:
        """Initial learning rate for each training run"""
        return self._lr0

    def __init__(
        self,
        epochs=20,
        balanced=False,
        additional_config={},
        optimizer: str = "auto",
        lr0: float = 0.01,
    ):
        TrainArgs.__init__(self, epochs, balanced, additional_config)

        self._optimizer = optimizer
        self._lr0 = lr0


class SOTAMultiRunTrainArgs(MultiRunTrainArgs):

    def __init__(self, runs: int = 5, train_args: SOTATrainArgs = SOTATrainArgs()):
        super().__init__(runs, train_args)


class SOTA(ClassificationModel):

    model: Optional[YOLO] = None

    @override
    @property
    def model_initialize_args(self) -> SOTAModelInitializeArgs:
        return self._model_initialize_args

    @override
    @property
    def model_arch(self) -> str:
        """Name of yolo model to use when initializing the model"""
        return self.model_initialize_args.model_arch

    @override
    def __init__(self, args: SOTAConstructorArgs):
        ClassificationModel.__init__(self, args)

    @override
    def execute_train_runs(self, args: SOTAMultiRunTrainArgs):
        ClassificationModel.execute_train_runs(self, args)

    def __get_train_wandb_config(self, args: SOTATrainArgs) -> dict:
        return (
            self._get_common_wandb_config()
            | {
                "augmented": True,
                "balanced": args.balanced,
                "optimizer": args.optimizer,
                "lr0": args.lr0,
                "run": self._get_next_train_run(),
            }
            | args.additional_config
        )

    def __get_test_wandb_config(self, args: TrainArgs) -> dict:
        return (
            self._get_common_wandb_config()
            | {
                "balanced": False,
                "augmented": False,
                "run": self._get_current_train_run(),
            }
            | args.additional_config
        )

    @override
    def train_model(self, args: SOTATrainArgs):
        if self.model is None:
            raise Exception("Cannot train before model is initialized")

        trainer = WeightedTrainer if args.balanced else None

        dataset_path = self.__get_dataset_dir()
        project_path = self.__get_project_dir()

        config = self.__get_train_wandb_config(args)
        init(
            project="detect-climbing-technique",
            job_type="train",
            group="sota",
            name=self.name,
            config=config,
            dir=self.data_root_path,
        )
        add_wandb_callback(
            self.model,
            enable_model_checkpointing=False,
            enable_train_validation_logging=False,
            enable_validation_logging=False,
            enable_prediction_logging=False,
        )

        results = self.model.train(
            trainer=trainer,
            data=dataset_path,
            epochs=args.epochs,
            imgsz=640,
            project=project_path,
            optimizer=args.optimizer,
            lr0=args.lr0,
            patience=3,
        )

        finish()

        print(results)

    @override
    def _get_model_dir(self):
        return join(self.data_root_path, "runs", "sota", self.name)

    @override
    def _get_best_model_path(self):
        model_dir = self._get_model_dir()
        train_run = get_current_train_run(model_dir)
        # train_list = [dir for dir in listdir(model_dir) if "train" in dir]
        # train_list.sort()
        return join(model_dir, train_run, "weights", "best.pt")

    @override
    def _load_model(self, best_weights_path):
        print(
            f"loading the model '{self.name}' with the weights at '{best_weights_path}'"
        )
        self.model = YOLO(best_weights_path)

    @override
    def _fresh_model(self):
        print(f"loading a fresh model '{self.model_arch}'")
        self.model = YOLO(self.model_arch)

    @override
    def test_model(self, args: TestArgs) -> DetMetrics:
        self._load_best_model()

        dataset_path = self.__get_dataset_dir()
        project_path = self.__get_project_dir()

        if args.write_to_wandb:
            config = self.__get_test_wandb_config(args)
            wandb_run = init(
                project="detect-climbing-technique",
                job_type="test",
                group="sota",
                name=self.name,
                config=config,
                dir=self.data_root_path,
            )
            add_wandb_callback(
                self.model,
                enable_model_checkpointing=False,
                enable_train_validation_logging=False,
                enable_validation_logging=False,
                enable_prediction_logging=False,
            )

        # Calculate metrics
        rename(join(dataset_path, "val"), join(dataset_path, "val_temp"))
        rename(join(dataset_path, "test"), join(dataset_path, "val"))
        try:
            metrics = self.model.val(
                data=dataset_path, project=project_path, name="test"
            )

            saved_metrics = metrics.results_dict.copy()
            saved_metrics["speed"] = metrics.speed.copy()

            self._save_test_metrics(saved_metrics)

            if args.write_to_wandb:
                wandb_run.log(saved_metrics)

        except Exception as ex:
            print(f"stopped with error: {ex}")
            raise ex
        finally:
            rename(join(dataset_path, "val"), join(dataset_path, "test"))
            rename(join(dataset_path, "val_temp"), join(dataset_path, "val"))

        # Override confusion matrices
        image_paths = glob(join(dataset_path, "test") + "/**/*.*", recursive=True)
        labels = [get_label_value_from_path(image_path) for image_path in image_paths]
        y_pred = self.model.predict(image_paths)
        predictions = [
            self.__get_label_value_from_prediction(prediction) for prediction in y_pred
        ]

        labels = list(map(value_to_name, labels))

        test_run_path = self._get_current_test_run_path()
        plot_confusion_matrix(
            labels,
            predictions,
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
            wandb_run.finish()

        return metrics

    @override
    def get_test_accuracy_metric(self) -> float:
        return self.get_test_metrics()["metrics/accuracy_top1"]

    def __get_label_value_from_prediction(self, prediction: Results) -> str:
        pred_name = prediction.names[prediction.probs.top1]
        return pred_name

    def __get_dataset_dir(self):
        return join(self.data_root_path, "img", self.dataset_name)

    def __get_project_dir(self):
        return join(self.data_root_path, "runs", "sota", self.name)


# results = model.predict(img, verbose = False)
# result = results[0]
# idx = result.probs.top1
# conf = result.probs.top1conf.item()
# label = model.names[idx]

# TODO: try https://github.com/rigvedrs/YOLO-V11-CAM for activation heatmaps
