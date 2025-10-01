from typing import Dict, Callable, List
from cv2.typing import MatLike
from numpy import ndarray, full, nan
from os.path import join
from pandas import DataFrame, read_pickle

from src.common.helpers import imread as imread_as_rgb
from src.hpe.common.helpers import eucl_distance, list_image_label_pairs
from src.hpe.common.landmarks import MyLandmark, YoloLabels, build_yolo_labels, get_most_central
from src.hpe.common.performance import log_overall_performance
from src.hpe.mp.evaluate import predict_landmarks
from src.hpe.mp.landmarks import PredictedLandmarks, can_predict
from src.hpe.mp.model import build_holistic_model

def PCKh50(ytrue: YoloLabels, yhat: PredictedLandmarks) -> Dict[MyLandmark, bool | None]:
    """
    Return mapping of MyLandmark to booleans to indicate correct, or incorrect predictions, 
    as defined by the PCKh50 metric.
    None, if mediapipe is not able to predict this landmark.
    """
    limit = ytrue.get_head_bone_link() / 2
    
    correct_pred = {}

    for landmark in MyLandmark:
        
        if not can_predict(landmark):
            correct_pred[landmark] = None
            continue

        ytrue_value = ytrue.get_keypoint(landmark)
        yhat_value = yhat[landmark]

        if yhat_value is None:
            result = ytrue_value.is_missing()
        else:
            result = eucl_distance(ytrue_value.as_array(), yhat_value.as_array()) <= limit

        correct_pred[landmark] = result

    return correct_pred

def distance(ytrue: YoloLabels, yhat: PredictedLandmarks) -> ndarray:
    _num_landmarks = len(MyLandmark)
    _limit = ytrue.get_head_bone_link() / 2
    distances = full(_num_landmarks, nan, dtype=float)

    for landmark in MyLandmark:
        if not can_predict(landmark):
            continue

        index = landmark.value
        ytrue_value = ytrue.get_keypoint(landmark)
        yhat_value = yhat[landmark]

        if ytrue_value.is_missing() and yhat_value is None:
            distances[index] = 0 
        elif yhat_value is not None:
            distances[index] = eucl_distance(ytrue_value.as_array(), yhat_value.as_array()) / _limit
                
    return distances

def estimate_performance(data_root: str = "data", split: str = "test",
        dataset_name: str = "MediaPipe",
        image_mutators: List[Callable[[MatLike], MatLike]] = []):
    root_image_dir = join(data_root, "hpe", "img", split, "images")
    data_pairs = list_image_label_pairs(root_image_dir)
    performance_maps = list()

    with build_holistic_model() as model:
        for image_path, label_path in data_pairs:
            image = imread_as_rgb(image_path)

            for mutator in image_mutators:
                image = mutator(image)
            
            results = predict_landmarks(image, model)
            labels_list = build_yolo_labels(label_path)

            if len(labels_list) == 0:
                performance_maps.append(results.ensure_empty())
            elif len(labels_list) == 1:
                performance_maps.append(PCKh50(labels_list[0], results))
            else:
                performance_maps.append(PCKh50(get_most_central(labels_list), results))
    
    log_overall_performance(performance_maps, dataset_name)

def estimate_distances(data_root: str = "data", split: str = "test",
        dataset_name: str = "distances",
        image_mutators: List[Callable[[MatLike], MatLike]] = []) -> DataFrame:
    root_image_dir = join(data_root, "hpe", "img", split, "images")
    data_pairs = list_image_label_pairs(root_image_dir)

    num_images = len(data_pairs)
    num_landmarks = len(MyLandmark)
    distances = full((num_images, num_landmarks), nan, dtype=float)
    index = 0
    
    with build_holistic_model() as model:
        for image_path, label_path in data_pairs:
            image = imread_as_rgb(image_path)

            for mutator in image_mutators:
                image = mutator(image)

            results = predict_landmarks(image, model)
            labels_list = build_yolo_labels(label_path)

            if len(labels_list) == 0:
                #TODO: add check if predictions are also empty, then set to array of zeros
                continue
            elif len(labels_list) == 1:
                distances[index] = distance(labels_list[0], results)
            else:
                distances[index] = distance(get_most_central(labels_list), results)
                
            index += 1

    result_path = join(data_root, "hpe", "mediapipe", f"{dataset_name}.pkl") 
    df = DataFrame(distances, columns=[landmark.name for landmark in MyLandmark])
    df.to_pickle(result_path)
    print(f"Distances saved to '{result_path}'")

    return df

def read_distances(data_root: str = "data", dataset_name: str = "distances") -> DataFrame:
    result_path = join(data_root, "hpe", "mediapipe", f"{dataset_name}.pkl")
    df = read_pickle(result_path)
    return df
