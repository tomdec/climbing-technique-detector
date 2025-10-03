from typing import override
from cv2.typing import MatLike
from os.path import join
from pandas import DataFrame, read_pickle

from src.hpe.common.performance import AbstractDistanceCollector, AbstractPerformanceLogger
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

def read_distances(data_root: str = "data", dataset_name: str = "distances") -> DataFrame:
    result_path = join(data_root, "hpe", "mediapipe", f"{dataset_name}.pkl")
    df = read_pickle(result_path)
    return df
