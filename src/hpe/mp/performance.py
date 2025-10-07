from typing import override, List, Callable, Tuple, Any
from cv2.typing import MatLike
from os.path import join
from pandas import DataFrame, read_pickle

from src.common.helpers import imread
from src.hpe.common.helpers import list_image_label_pairs
from src.hpe.common.landmarks import PredictedKeyPoint, PredictedKeyPoints, YoloLabels, build_yolo_labels, get_most_central, MyLandmark
from src.hpe.common.performance import AbstractDistanceCollector, AbstractEstimationCollector, AbstractPerformanceLogger, HpeEstimation
from src.hpe.mp.evaluate import predict_landmarks
from src.hpe.mp.landmarks import MediaPipePredictedKeyPoints
from src.hpe.mp.model import build_holistic_model

class PerformanceLogger(AbstractPerformanceLogger):

    @override
    def _get_predictions(self, image: MatLike) -> MediaPipePredictedKeyPoints:
        with build_holistic_model() as model:
            return predict_landmarks(image, model)

class DistanceCollector(AbstractDistanceCollector):

    @override
    def __init__(self, data_root="data"):
        super().__init__("mediapipe", data_root)

    @override
    def _get_predictions(self, image: MatLike) -> MediaPipePredictedKeyPoints:
        with build_holistic_model() as model:
            return predict_landmarks(image, model)

class EstimationCollector(AbstractEstimationCollector):

    @override
    def __init__(self, data_root="data"):
        super().__init__("mediapipe", data_root)

    @override
    def _get_predictions(self, image: MatLike) -> MediaPipePredictedKeyPoints:
        with build_holistic_model() as model:
            return predict_landmarks(image, model)

def read_distances(data_root: str = "data", dataset_name: str = "distances") -> DataFrame:
    result_path = join(data_root, "hpe", "mediapipe", f"{dataset_name}.pkl")
    df = read_pickle(result_path)
    return df

def read_estimations(data_root: str = "data", name: str = "estimations") -> DataFrame:
    result_path = join(data_root, "hpe", "mediapipe", f"{name}.pkl")
    df = read_pickle(result_path)
    df = df.map(HpeEstimation.from_dict)
    return df