from typing import List, override, Callable, Tuple
from cv2.typing import MatLike
from numpy import ndarray
from pandas import DataFrame, read_pickle
from os.path import join
from ultralytics import YOLO 

from src.common.helpers import imread
from src.hpe.common.helpers import list_image_label_pairs
from src.hpe.common.metrics import PerformanceMap
from src.hpe.common.landmarks import MyLandmark, build_yolo_labels, YoloLabels, PredictedKeyPoints, PredictedKeyPoint, get_most_central
from src.hpe.common.performance import AbstractDistanceCollector, AbstractPerformanceCollector, AbstractPerformanceLogger, \
    HpeEstimation
from src.hpe.yolo.landmarks import YoloPredictedKeyPoints
from src.hpe.yolo.evaluate import predict_landmarks
from src.hpe.yolo.model import build_pose_model

class PerformanceLogger(AbstractPerformanceLogger):
    _model: YOLO | None = None

    @override
    def _get_predictions(self, image: MatLike) -> YoloPredictedKeyPoints:
        if self._model is None:
            self._model = build_pose_model()
        return predict_landmarks(image, self._model)
    
    @override
    def _post_process(self, name: str, results: List[PerformanceMap]):
        self._model = None
        super()._post_process(name, results)

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

class EstimationCollector(AbstractPerformanceCollector):
    _model: YOLO | None

    @override
    def __init__(self, data_root = "data"):
        super().__init__(data_root)
        self._model = None

    @override
    def collect(self, name: str, split: str = "test",
            image_mutators: List[Callable[[MatLike], MatLike]] = []) -> DataFrame:
        root_image_dir = join(self._data_root, "hpe", "img", split, "images")
        data_pairs = list_image_label_pairs(root_image_dir)

        def func(pair: Tuple[str, str]) -> List[HpeEstimation]:
            image_path, label_path = pair
            image = imread(image_path)
            for mutator in image_mutators:
                image = mutator(image)
            
            labels_list = build_yolo_labels(label_path)
            predictions = self._get_predictions(image)

            return self.build_estimations(image_path, labels_list, predictions)

        results = list(map(func, data_pairs))

        return self._post_process(name, results)
    
    @override
    def _get_predictions(self, image: MatLike) -> YoloPredictedKeyPoints:
        if self._model is None:
            self._model = build_pose_model()
        return predict_landmarks(image, self._model)
    
    def _build_tn_or_fp(self, image_path: str,
            predictions: PredictedKeyPoints) -> List[HpeEstimation]:

        def factory(landmark: MyLandmark) -> HpeEstimation:
            can_predict = predictions.can_predict(landmark)
            prediction = predictions[landmark] if can_predict else PredictedKeyPoint.empty()

            return HpeEstimation(
                true_landmark=None,
                predicted_landmark=prediction,
                head_bone_link=None,
                image_path=image_path,
                can_predict=can_predict)

        return list(map(factory, MyLandmark))        

    def build_fn_or_tp(self, image_path: str, 
            labels: YoloLabels, 
            predictions: PredictedKeyPoints) -> List[HpeEstimation]:
        
        def factory(landmark: MyLandmark) -> HpeEstimation:
            can_predict = predictions.can_predict(landmark)
            prediction = predictions[landmark] if can_predict else PredictedKeyPoint.empty()

            return HpeEstimation(
                true_landmark=labels.get_keypoint(landmark),
                predicted_landmark=prediction,
                head_bone_link=labels.get_head_bone_link(),
                image_path=image_path,
                can_predict=can_predict)
        
        return list(map(factory, MyLandmark))
    
    def build_estimations(self, image_path: str,
            labels_list: List[YoloLabels],
            predictions: PredictedKeyPoints) -> List[HpeEstimation]:
        
        if len(labels_list) == 0:
            return self._build_tn_or_fp(image_path, predictions)
        elif len(labels_list) == 1:
            # 1 set of labels and predictions
            return self.build_fn_or_tp(image_path, labels_list[0], predictions)
        else:
            # Multiple labels (people in image), only evaluate most central.
            return self.build_fn_or_tp(image_path, get_most_central(labels_list), predictions)

    @override
    def _post_process(self, name: str, results: List[List[HpeEstimation]]) -> DataFrame:
        self._model = None
        result_path = join(self._data_root, "hpe", "yolo", f"{name}.pkl")
        df = DataFrame(results, columns=[landmark.name for landmark in MyLandmark])
        df_dict = df.map(HpeEstimation.as_dict)
        df_dict.to_pickle(result_path)
        return df

def read_distances(data_root: str = "data",
        dataset_name: str = "distances") -> DataFrame:
    result_path = join(data_root, "hpe", "yolo", f"{dataset_name}.pkl")
    df = read_pickle(result_path)
    return df
