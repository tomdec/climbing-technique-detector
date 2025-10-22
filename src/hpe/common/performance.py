from typing import Callable, Any, List, Tuple, override
from cv2.typing import MatLike
from os.path import join
from numpy import ndarray, full, nan
from pandas import DataFrame

from src.hpe.common.typing import HpeEstimation
from src.hpe.common.metrics import calc_accuracy, calc_throughput, distance
from src.common.helpers import imread, raise_not_implemented_error
from src.hpe.common.helpers import list_image_label_pairs
from src.hpe.common.landmarks import MyLandmark, PredictedKeyPoint, PredictedKeyPoints, \
    build_yolo_labels, YoloLabels, get_most_central

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

def log_performance(data: DataFrame, name: str = "model"):

    throughput = calc_throughput(data)
    print(f"{name} has throughput of: {throughput:.2%}")

    accuracy = calc_accuracy(data)
    print(f"{name} has accuracy of: {accuracy:.2%}")