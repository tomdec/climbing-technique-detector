from typing import List, override
from cv2.typing import MatLike
from numpy import ndarray
from pandas import DataFrame, read_pickle
from os.path import join
from ultralytics import YOLO 

from src.hpe.common.typing import MyLandmark
from src.hpe.common.typing import HpeEstimation
from src.hpe.common.performance import AbstractDistanceCollector, AbstractEstimationCollector
from src.hpe.yolo.landmarks import YoloPredictedKeyPoints, get_recognizable_landmarks
from src.hpe.yolo.evaluate import predict_landmarks
from src.hpe.yolo.model import build_pose_model

class DistanceCollector(AbstractDistanceCollector):
    _model: YOLO | None

    @override
    def __init__(self, data_root="data"):
        super().__init__("yolo", data_root)
        self._model = None

    @override
    def _get_predictions(self, image: MatLike) -> YoloPredictedKeyPoints:
        if self._model is None:
            self._model = build_pose_model()
        return predict_landmarks(image, self._model)
    
    @override
    def _post_process(self, name: str, results: List[ndarray]) -> DataFrame:
        self._model = None
        return super()._post_process(name, results)

class EstimationCollector(AbstractEstimationCollector):
    _model: YOLO | None

    @override
    def __init__(self, data_root="data"):
        super().__init__("yolo", data_root)
        self._model = None
    
    @override
    def _get_predictions(self, image: MatLike) -> YoloPredictedKeyPoints:
        if self._model is None:
            self._model = build_pose_model()
        return predict_landmarks(image, self._model)
    
    @override
    def _post_process(self, name: str, results: List[List[HpeEstimation]]) -> DataFrame:
        self._model = None
        return super()._post_process(name, results)

def read_distances(data_root: str = "data",
        dataset_name: str = "distances") -> DataFrame:
    result_path = join(data_root, "hpe", "yolo", f"{dataset_name}.pkl")
    df = read_pickle(result_path)
    return df

def read_estimations(data_root: str = "data", name: str = "estimations") -> DataFrame:
    result_path = join(data_root, "hpe", "yolo", f"{name}.pkl")
    df = read_pickle(result_path)
    df = df.map(HpeEstimation.from_dict)
    return df

def log_recognizable_landmarks():
    total = len(MyLandmark)
    recognizable = get_recognizable_landmarks()

    print(f"YOLO can recognize {recognizable} of {total} landmarks")