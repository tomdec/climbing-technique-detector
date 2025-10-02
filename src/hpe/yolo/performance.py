from typing import Dict, List, Callable
from cv2.typing import MatLike
from numpy import full, nan, array, ndarray
from pandas import DataFrame, read_pickle
from os.path import join

from src.common.helpers import imread as imread_as_rgb
from src.hpe.common.landmarks import YoloLabels, MyLandmark, build_yolo_labels, get_most_central
from src.hpe.common.helpers import eucl_distance, list_image_label_pairs
from src.hpe.common.performance import log_overall_performance
from src.hpe.yolo.landmarks import YoloPredictedKeyPoints, can_predict
from src.hpe.yolo.evaluate import predict_landmarks
from src.hpe.yolo.model import build_pose_model

__num_landmarks = len(MyLandmark)

def PCKh50(ytrue: YoloLabels, 
        yhat: YoloPredictedKeyPoints,
        verbose: bool = False,
        limit_mod = 1) -> Dict[MyLandmark, bool | None]:
    
    limit = ytrue.get_head_bone_link() / 2 * limit_mod
    if verbose:
        print(f'PCKh50 limit: {limit}')

    correct_pred = {}

    for landmark in MyLandmark:
        if not can_predict(landmark):
            correct_pred[landmark] = None
            continue
        
        ytrue_value = ytrue.get_keypoint(landmark)
        
        result = None
        yhat_value = yhat[landmark]

        if ytrue_value.is_missing() and yhat_value is None:
            # True Negative
            result = True
        elif ytrue_value.is_missing() and yhat_value is not None:
            # False Positive
            result = False
        elif not ytrue_value.is_missing() and yhat_value is None:
            # False Negative
            result = False
        else:
            if verbose:
                print(f'y_true: {ytrue_value.as_array()}')
                print(f'y_hat: {yhat_value.as_array()}')
            result = eucl_distance(ytrue_value.as_array(), yhat_value.as_array()) <= limit
        
        correct_pred[landmark] = result

    return correct_pred
    
def distance(ytrue: YoloLabels, 
        yhat: YoloPredictedKeyPoints) -> ndarray: 
    _limit = ytrue.get_head_bone_link() / 2
    
    def calc_distance(landmark: MyLandmark) -> float:
        if not can_predict(landmark):
            return nan
        
        ytrue_value = ytrue.get_keypoint(landmark)
        yhat_value = yhat[landmark]

        if ytrue_value.is_missing() and yhat_value is None:
            # True Negative
            return 0
        elif ytrue_value.is_missing() and yhat_value is not None:
            # False Positive
            return nan # should be handle differently than 'cannot predict' case
        elif not ytrue_value.is_missing() and yhat_value is None:
            # False Negative
            return nan # should be handle differently than 'cannot predict' case
        else:
            return eucl_distance(ytrue_value.as_array(), yhat_value.as_array()) / _limit

    distances = list(map(calc_distance, MyLandmark))
    return array(distances)

def performance_map_for_false_negative() -> Dict[MyLandmark, bool | None]:
    performance = {}
    for landmark in MyLandmark:
            performance[landmark] = None
    return performance

def estimate_performance(data_root: str = "data", split: str = "test",
        name: str = "Yolo",
        image_mutators: List[Callable[[MatLike], MatLike]] = []):
    path = join(data_root, "hpe", "img", split, "images")
    data_pairs = list_image_label_pairs(path)
    performance_maps = list()
    
    model = build_pose_model()
    for image_path, label_path in data_pairs:
        image = imread_as_rgb(image_path)
        
        for mutator in image_mutators:
            image = mutator(image)

        results = predict_landmarks(image, model)
        labels_list = build_yolo_labels(label_path)

        if len(labels_list) == 0:
            performance_maps.append(results.ensure_empty())
        elif len(labels_list) == 1:
            if len(results) == 0:
                performance_maps.append(performance_map_for_false_negative())
            else:
                performance_maps.append(PCKh50(labels_list[0], results))
        else:
            performance_maps.append(PCKh50(get_most_central(labels_list), results))
    
    log_overall_performance(performance_maps, name)

def estimate_distances(data_root: str = "data", split: str = "test",
        dataset_name: str = "distances",
        image_mutators: List[Callable[[MatLike], MatLike]] = []) -> DataFrame:
    """
    Estimate the distances between HPE landmark labels and predicted HPE landmarks.
    Distances are saved to the file system as a .pkl file.

    Args:
        data_root (str, optional): Path to the data directory. Defaults to "data".
        split (str, optional): Split for which to calculate the distances. Defaults to "test".
        dataset_name: Name of the file the distances are stored in. Defaults to "distances".
        image_path_mutators (List[Callable[[str], str]], optional): Functions to call on the input image paths. Defaults to [].
        clean_up (List[Callable], optional): Clean up functions to execute after all estimations are done and the distances are saved to the file system. Defaults to [].

    Returns:
        DataFrame: Matrix with distances (normalized according to PCKh50) between label landmarks and predicted landmarks.
        Each row represents an image and the columns the landmarks.
    """
    path = join(data_root, "hpe", "img", split, "images")
    data_pairs = list_image_label_pairs(path)
    model = build_pose_model()
    
    _num_images = len(data_pairs)
    distances = full((_num_images, __num_landmarks), nan, dtype=float)
    index = 0

    for image_path, label_path in data_pairs:
        image = imread_as_rgb(image_path)
        
        for mutator in image_mutators:
            image = mutator(image)
        
        results = predict_landmarks(image, model)
        labels_list = build_yolo_labels(label_path)

        if len(labels_list) == 0:
            if results.no_person_detected():
                # True Negative
                distances[index] = full(__num_landmarks, 0, dtype=float)
            else:
                # False Postive
                continue
        elif len(labels_list) == 1:
            if results.no_person_detected():
                # False Negative
                continue
            else:    
                # 1 set of labels and predictions
                distances[index] = distance(labels_list[0], results)
        else:
            # Multiple labels (people in image), only evaluate most central.
            distances[index] = distance(get_most_central(labels_list), results)
        
        index += 1

    result_path = join(data_root, "hpe", "yolo", f"{dataset_name}.pkl")
    df = DataFrame(distances)
    df.columns = [landmark.name for landmark in MyLandmark]
    df.to_pickle(result_path)
    print(f"Distances saved to '{result_path}'")

    return df

def read_distances(data_root: str = "data",
        dataset_name: str = "distances") -> DataFrame:
    result_path = join(data_root, "hpe", "yolo", f"{dataset_name}.pkl")
    df = read_pickle(result_path)
    return df
