from typing import Callable, Any, List, Dict, Tuple, override
from cv2.typing import MatLike
from os.path import join
from numpy import ndarray, full, nan
from pandas import DataFrame
from sympy import true

from src.hpe.common.metrics import PerformanceMap, PCKh50, distance
from src.common.helpers import imread, raise_not_implemented_error
from src.hpe.common.helpers import eucl_distance, list_image_label_pairs
from src.hpe.common.landmarks import KeyPoint, MyLandmark, PredictedKeyPoint, PredictedKeyPoints, \
    get_mylandmark_count, build_yolo_labels, YoloLabels, get_most_central

class HpeEstimation:

    @staticmethod
    def from_dict(values: dict) -> 'HpeEstimation':
        return HpeEstimation(
            true_landmark=None if values['true_landmark'] is None else KeyPoint.from_dict(values['true_landmark']),
            predicted_landmark=None if values['predicted_landmark'] is None else PredictedKeyPoint.from_dict(values['predicted_landmark']),
            head_bone_link=values['head_bone_link'],
            image_path=values['image_path'],
            can_predict=values['can_predict']
        )

    @property
    def true_landmark(self) -> KeyPoint | None:
        return self._true_landmark

    @property
    def predicted_landmark(self) -> PredictedKeyPoint:
        return self._predicted_landmark
    
    @property
    def head_bone_link(self) -> float | None:
        return self._head_bone_link
    
    @property
    def image_path(self) -> str:
        return self._image_path
    
    @property
    def can_predict(self) -> bool:
        return self._can_predict

    def __init__(self, true_landmark: KeyPoint | None,
            predicted_landmark: PredictedKeyPoint,
            head_bone_link: float | None,
            image_path: str,
            can_predict: bool):
        self._true_landmark = true_landmark
        self._predicted_landmark = predicted_landmark
        self._head_bone_link = head_bone_link
        self._image_path = image_path
        self._can_predict = can_predict

    def __str__(self) -> str:
        return f"{self.as_dict()}"

    def _get_estimation(self, conf_threshold: float) -> PredictedKeyPoint | None:
        if self._predicted_landmark.is_missing():
            return None
        elif self._predicted_landmark.visibility < conf_threshold:
            return None
        else:
            return self._predicted_landmark

    def prediction_result(self, conf_threshold: float = 0.5) -> str:
        if not self._can_predict:
            return ""
        estimation = self._get_estimation(conf_threshold)
        if self._true_landmark is None and estimation is None:
            return "TN"
        elif self._true_landmark is not None and estimation is None:
            return "FN"
        elif self._true_landmark is None and estimation is not None:
            return "FP"
        else:
            limit = self._head_bone_link / 2
            correct = eucl_distance(self._true_landmark.as_array(), self._predicted_landmark.as_array()) <= limit
            return "TP" if correct else "FP"

    def as_dict(self) -> dict:
        return {
            'true_landmark': None if self._true_landmark is None else self._true_landmark.as_dict(),
            'predicted_landmark': self._predicted_landmark.as_dict(),
            'head_bone_link': self._head_bone_link,
            'image_path': self._image_path,
            'can_predict': self._can_predict
        }

class AbstractPerformanceCollector:
    _data_root: str
    
    def __init__(self, data_root: str = "data"):
        self._data_root = data_root

    def collect(self, name: str, split: str = "test",
            image_mutators: List[Callable[[MatLike], MatLike]] = []) -> Any:
        root_image_dir = join(self._data_root, "hpe", "img", split, "images")
        data_pairs = list_image_label_pairs(root_image_dir)

        def func(pair: Tuple[str, str]) -> Any:
            image_path, label_path = pair
            image = imread(image_path)
            for mutator in image_mutators:
                image = mutator(image)
            
            labels = build_yolo_labels(label_path)
            predictions = self._get_predictions(image)

            return self._process_with_image_path(labels, predictions, image_path)

        results = list(map(func, data_pairs))

        return self._post_process(name, results)
    
    def _process_with_image_path(self, 
            labels: List[YoloLabels], 
            predictions: PredictedKeyPoints, 
            image_path: str) -> Any:
        return self._process(labels, predictions)

    def _get_predictions(self, image: MatLike) -> PredictedKeyPoints:
        raise_not_implemented_error(self.__class__.__name__, self._get_predictions.__name__)

    def _process(self, labels: List[YoloLabels], predictions: PredictedKeyPoints) -> Any:
        raise_not_implemented_error(self.__class__.__name__, self._process.__name__)

    def _post_process(self, name: str, results: List[Any]) -> Any:
        raise_not_implemented_error(self.__class__.__name__, self._post_process.__name__)
    
class AbstractPerformanceLogger(AbstractPerformanceCollector):
    
    @override
    def _process(self, labels: List[YoloLabels], predictions: PredictedKeyPoints) -> PerformanceMap:
        if len(labels) == 0:
            # True Negative and False Positives
            return predictions.ensure_empty()
        elif len(labels) == 1:
            return PCKh50(labels[0], predictions)
        else:
            return PCKh50(get_most_central(labels), predictions)

    @override
    def _post_process(self, name: str, results: List[PerformanceMap]):
        log_overall_performance(results, name)

class AbstractDistanceCollector(AbstractPerformanceCollector):
    __num_landmarks: int = len(MyLandmark)

    _result_dir_path: str

    @override
    def __init__(self, tool_name: str, 
            data_root = "data"):
        super().__init__(data_root)
        self._result_dir_path = join(data_root, "hpe", tool_name)

    @override
    def _process(self, labels: List[YoloLabels], predictions: PredictedKeyPoints) -> ndarray:
        if len(labels) == 0:
            if predictions.no_person_detected():
                # True Negative
                return full(self.__num_landmarks, 0, dtype=float)
            else:
                # False Postive
                return full(self.__num_landmarks, nan, dtype=float)
        elif len(labels) == 1:
            if predictions.no_person_detected():
                # False Negative
                return full(self.__num_landmarks, nan, dtype=float)
            else:    
                # 1 set of labels and predictions
                return distance(labels[0], predictions)
        else:
            # Multiple labels (people in image), only evaluate most central.
            return distance(get_most_central(labels), predictions)
        
    @override
    def _post_process(self, name: str, results: List[ndarray]) -> DataFrame:
        results_path = join(self._result_dir_path, f"{name}.pkl")
        df = DataFrame(results, columns=[landmark.name for landmark in MyLandmark])
        df.to_pickle(results_path)
        print(f"Distances saved to '{results_path}'")
        return df

class AbstractEstimationCollector(AbstractPerformanceCollector):
    _result_dir_path: str

    @override
    def __init__(self, tool_name: str, 
            data_root = "data"):
        super().__init__(data_root)
        self._result_dir_path = join(data_root, "hpe", tool_name)

    @override
    def _process_with_image_path(self, labels: List[YoloLabels],
            predictions: PredictedKeyPoints,
            image_path: str) -> List[HpeEstimation]:
        
        if len(labels) == 0:
            return self._build_tn_or_fp(image_path, predictions)
        elif len(labels) == 1:
            # 1 set of labels and predictions
            return self.build_fn_or_tp(image_path, labels[0], predictions)
        else:
            # Multiple labels (people in image), only evaluate most central.
            return self.build_fn_or_tp(image_path, get_most_central(labels), predictions)
    
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
    
    @override
    def _post_process(self, name: str, results: List[List[HpeEstimation]]) -> DataFrame:
        self._model = None
        result_path = join(self._result_dir_path, f"{name}.pkl")
        df = DataFrame(results, columns=[landmark.name for landmark in MyLandmark])
        df_dict = df.map(HpeEstimation.as_dict)
        df_dict.to_pickle(result_path)
        return df

def count_values(map: PerformanceMap, value: bool | None):
    count = 0
    for landmark in MyLandmark:
        if map[landmark] == value:
            count += 1

    return count

def performance(map: PerformanceMap):
    correct_num = count_values(map, True)
    cant_detect = count_values(map, None)

    return correct_num, get_mylandmark_count() - cant_detect

def log_overall_performance(maps: List[PerformanceMap],
        model_name: str = "The model"):
    total_correct = 0
    max_possible = 0
    _, detected = performance(maps[0])
    
    for map in maps:
        correct, possible = performance(map)
        total_correct += correct
        max_possible += possible

    print(f'Percentage of landmarks detected correctly: {total_correct / max_possible}%')
    print(f'{model_name} can only detect {detected} landmarks of our {get_mylandmark_count()} labels')