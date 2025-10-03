from typing import Callable, Any, List, Dict, Tuple, override
from cv2.typing import MatLike
from os.path import join
from numpy import ndarray, full, nan
from pandas import DataFrame

from src.hpe.common.metrics import PerformanceMap, PCKh50, distance
from src.common.helpers import imread, raise_not_implemented_error
from src.hpe.common.helpers import list_image_label_pairs
from src.hpe.common.landmarks import KeyPoint, MyLandmark, PredictedKeyPoint, PredictedKeyPoints, \
    get_mylandmark_count, build_yolo_labels, YoloLabels, get_most_central

class HpeEstimation:

    @property
    def true_landmark(self) -> KeyPoint | None:
        return self._true_landmark

    @property
    def predicted_landmark(self) -> PredictedKeyPoint | None:
        return self._predicted_landmark
    
    @property
    def head_bone_link(self) -> float:
        return self._head_bone_link
    
    @property
    def can_predict(self) -> bool:
        return self._can_predict
    
    @property
    def image_path(self) -> str:
        return self._image_path

    def __init__(self):
        pass

    def __str__(self) -> str:
        return f"{self.as_dict()}"

    def as_dict(self) -> dict:
        pass

class AbstractPerformanceCollector:
    _data_root: str
    
    def __init__(self, data_root: str = "data"):
        self._data_root = data_root

    def collect(self, name: str, split: str = "test",
            image_mutators: List[Callable[[MatLike], MatLike]] = []):
        root_image_dir = join(self._data_root, "hpe", "img", split, "images")
        data_pairs = list_image_label_pairs(root_image_dir)

        def func(pair: Tuple[str, str]) -> Any:
            image_path, label_path = pair
            image = imread(image_path)
            for mutator in image_mutators:
                image = mutator(image)
            
            labels_list = build_yolo_labels(label_path)
            predictions = self._get_predictions(image)

            return self._process(labels_list, predictions)

        results = list(map(func, data_pairs))

        return self._post_process(name, results)
    
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

def count_values(map: Dict[MyLandmark, bool | None], value: bool | None):
    count = 0
    for landmark in MyLandmark:
        if map[landmark] == value:
            count += 1

    return count

def performance(map):
    correct_num = count_values(map, True)
    cant_detect = count_values(map, None)

    return correct_num, get_mylandmark_count() - cant_detect

def log_overall_performance(maps,
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