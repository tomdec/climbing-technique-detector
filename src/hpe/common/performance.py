from typing import Callable, Any, List, Dict, Tuple
from cv2.typing import MatLike
from numpy import array
from os.path import join

from src.common.helpers import imread, raise_not_implemented_error
from src.hpe.common.helpers import eucl_distance, list_image_label_pairs
from src.hpe.common.landmarks import KeyPoint, MyLandmark, PredictedKeyPoint, \
    get_mylandmark_count, build_yolo_labels, YoloLabels

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

    def collect(self,
            name: str,
            split: str = "test",
            image_mutators: List[Callable[[MatLike], MatLike]] = []):
        root_image_dir = join(self._data_root, "hpe", "img", split, "images")
        data_pairs = list_image_label_pairs(root_image_dir)

        def func(pair: Tuple[str, str]) -> Any:
            image_path, label_path = pair
            image = imread(image_path)
            for mutator in image_mutators:
                image = mutator(image)
            labels_list = build_yolo_labels(label_path)
            
            return self._process(image, labels_list)

        results = list(map(func, data_pairs))

        return self._post_process(name, results)

    def _process(self, image: MatLike, labels: List[YoloLabels]) -> Any:
        raise_not_implemented_error(self.__class__.__name__, self._process.__name__)

    def _post_process(self, name: str, results: List[Any]) -> Any:
        raise_not_implemented_error(self.__class__.__name__, self._post_process.__name__)


def PCKh50_general(ytrue: YoloLabels, get_prediction: Callable[[MyLandmark], Any | None]) -> List[bool]:
    limit = ytrue.get_head_bone_link() / 2
    
    correct_pred = list()

    for landmark in MyLandmark:
        yhat_value = get_prediction(landmark)
        if yhat_value is None:
            continue
        ytrue_value = ytrue._key_points[landmark]
        
        is_correct = eucl_distance(ytrue_value.as_array(), array([yhat_value.x, yhat_value.y])) <= limit
        correct_pred.append(is_correct)
    
    return correct_pred

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