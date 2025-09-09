from typing import Dict, List, Callable
from numpy import array, full, nan, ndarray
from ultralytics.engine.results import Results, Keypoints
from pandas import DataFrame, read_pickle
from os.path import join

from src.hpe.common.labels import YoloLabels, MyLandmark, build_yolo_labels, get_most_central
from src.hpe.common.helpers import eucl_distance, list_image_label_pairs
from src.hpe.common.performance import log_overall_performance
from src.hpe.yolo.landmarks import get_pose_landmark
from src.hpe.yolo.evaluate import predict_landmarks
from src.hpe.yolo.model import build_pose_model

__num_landmarks = 28

def PCKh50(ytrue: YoloLabels, 
        yhat: Keypoints,
        object_index: int = 0,
        verbose: bool = False,
        limit_mod = 1) -> Dict[MyLandmark, bool | None]:
    
    limit = ytrue.get_head_bone_link() / 2 * limit_mod
    if verbose:
        print(f'PCKh50 limit: {limit}')

    correct_pred = {}

    for landmark in MyLandmark:
        ytrue_value = ytrue.get_keypoint(landmark)
        pose_landmark_index = get_pose_landmark(landmark)
        
        result = None
        if pose_landmark_index is not None:
            yhat_value = _get_pose_prediction(yhat, object_index, pose_landmark_index)

            if yhat_value is None:
                result = ytrue_value.is_missing()
            else:
                if verbose:
                    print(f'y true: {ytrue_value.as_array()}')
                    print(f'y hat: {yhat_value}')
                result = eucl_distance(ytrue_value.as_array(), yhat_value) <= limit
        
        correct_pred[landmark] = result

    return correct_pred
    
def distance(ytrue: YoloLabels, 
        yhat: Keypoints,
        object_index: int = 0): 
    _limit = ytrue.get_head_bone_link() / 2
    distances = full(__num_landmarks, nan, dtype=float)

    for landmark in MyLandmark:
        index = landmark.value
        ytrue_value = ytrue.get_keypoint(landmark)
        pose_landmark_index = get_pose_landmark(landmark)
        
        if pose_landmark_index is not None:
            yhat_value = _get_pose_prediction(yhat, object_index, pose_landmark_index)

            if ytrue_value.is_missing() and yhat_value is None:
                distances[index] = 0
            else:
                distances[index] = eucl_distance(ytrue_value.as_array(), yhat_value) / _limit

    return distances

def ensure_empty(yhat: Results) -> Dict[MyLandmark, bool | None]:
    correct_pred = {}
    
    if yhat.keypoints.xyn.shape[0] == 0:
        for landmark in MyLandmark:
            pose_landmark_index = get_pose_landmark(landmark)
            correct_pred[landmark] = True if pose_landmark_index is not None else None
    else:
        for landmark in MyLandmark:
            pose_landmark_index = get_pose_landmark(landmark)
            result = None
            
            if pose_landmark_index is not None:
                yhat_value = _get_pose_prediction(yhat.keypoints, 0, pose_landmark_index)
                result = yhat_value is None
            
            correct_pred[landmark] = result

    return correct_pred

def performance_map_for_false_negative() -> Dict[MyLandmark, bool | None]:
    performance = {}
    for landmark in MyLandmark:
            performance[landmark] = None
    return performance

def estimate_performance(data_root: str = "data", split: str = "test",
        name: str = "Yolo",
        image_path_mutators: List[Callable[[str], str]] = [],
        clean_up: List[Callable] = []):
    try:
        path = join(data_root, "hpe", "img", split, "images")
        data_pairs = list_image_label_pairs(path)
        performance_maps = list()
        model = build_pose_model()
        
        for image_path, label_path in data_pairs:

            for mutator in image_path_mutators:
                image_path = mutator(image_path)

            _, results, _ = predict_landmarks(image_path, model)
            labels_list = build_yolo_labels(label_path)

            if len(labels_list) == 0:
                performance_maps.append(ensure_empty(results))
            elif len(labels_list) == 1:
                if len(results) == 0:
                    performance_maps.append(performance_map_for_false_negative())
                else:
                    performance_maps.append(PCKh50(labels_list[0], 
                        results[0].keypoints, 0))
            else:
                performance_maps.append(PCKh50(get_most_central(labels_list), 
                    results[0].keypoints, 0))
        
        log_overall_performance(performance_maps, name)
    finally:
        for clean_func in clean_up:
            clean_func()

def estimate_distances(data_root: str = "data", split: str = "test",
        dataset_name: str = "distances",
        image_path_mutators: List[Callable[[str], str]] = [],
        clean_up: List[Callable] = []) -> DataFrame:
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
    try:
        path = join(data_root, "hpe", "img", split, "images")
        data_pairs = list_image_label_pairs(path)
        model = build_pose_model()
        
        _num_images = len(data_pairs)
        _num_landmarks = 28
        distances = full((_num_images, _num_landmarks), nan, dtype=float)
        index = 0

        for image_path, label_path in data_pairs:

            for mutator in image_path_mutators:
                image_path = mutator(image_path)
            
            _, results, _ = predict_landmarks(image_path, model)
            labels_list = build_yolo_labels(label_path)

            if len(labels_list) == 0:
                if len(results) == 0:
                    # True Negative
                    distances[index] = full(_num_landmarks, 0, dtype=float)
                else:
                    # False Postive
                    continue
            elif len(labels_list) == 1:
                if len(results) == 0:
                    # False Negative
                    continue
                else:    
                    # 1 set of labels and predictions
                    distances[index] = distance(labels_list[0], results[0].keypoints, 0)
            else:
                # Multiple labels (people in image), only evaluate most central.
                distances[index] = distance(get_most_central(labels_list), results[0].keypoints, 0)
            
            index += 1

        result_path = join(data_root, "hpe", "yolo", f"{dataset_name}.pkl")
        df = DataFrame(distances)
        df.columns = [landmark.name for landmark in MyLandmark]
        df.to_pickle(result_path)
        print(f"Distances saved to '{result_path}'")

        return df

    finally:
        for clean_func in clean_up:
            clean_func()

def read_distances(data_root: str = "data",
        dataset_name: str = "distances") -> DataFrame:
    result_path = join(data_root, "hpe", "yolo", f"{dataset_name}.pkl")
    df = read_pickle(result_path)
    return df

def _get_pose_prediction(results: Keypoints, object_index: int, key: int) -> ndarray | None:
    """
    Get normalized coordinates of a predicted landmark for a detected object. 
    None, if the landmark is not detected.

    Args:
        results (Keypoints): HPE keypoints predicted by a YOLO model 
        object_index (int): Index of a detected object in the results.
        key (int): Index of a landmark.

    Returns:
        (ndarray | None): Normalized coordinates (x, y) of the predicted landmark.
    """
    
    if results.xyn.shape.count(0) > 0:
        return None
    
    raw_tensor = results.xyn[object_index][key]
    if raw_tensor.device.type == 'cuda':
        return array(raw_tensor.cpu())
    return array(raw_tensor)
