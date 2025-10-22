from typing import override
from cv2.typing import MatLike
from os.path import join
from pandas import DataFrame, read_pickle

from src.hpe.common.landmarks import MyLandmark
from src.hpe.common.typing import HpeEstimation
from src.hpe.common.performance import AbstractDistanceCollector, AbstractEstimationCollector
from src.hpe.mp.evaluate import predict_landmarks
from src.hpe.mp.landmarks import MediaPipePredictedKeyPoints, get_recognizable_landmarks
from src.hpe.mp.model import build_holistic_model

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

def log_recognizable_landmarks():
    total = len(MyLandmark)
    recognizable = get_recognizable_landmarks()

    print(f"MediaPipe can recognize {recognizable} of {total} landmarks")