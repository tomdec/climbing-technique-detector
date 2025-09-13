from ultralytics import YOLO
from ultralytics.engine.results import Results
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_BGR2RGB
from cv2.typing import MatLike
from typing import Tuple, List, Any
from numpy import ndarray, array

from src.labels import get_label_value_from_path
from src.hpe.yolo.model import build_pose_model
from src.hpe.yolo.landmarks import _pose_landmark_mapping

def predict_landmarks(image_path: str | MatLike, model: YOLO) -> Tuple[MatLike, Results, Tuple[int, int]]:
    results = model(image_path)
    image = cvtColor(results[0].orig_img, COLOR_BGR2RGB)

    return image, results[0], results[0].orig_shape

def __to_np_array(results: Results) -> ndarray:
    result_array = []
    for value in list(_pose_landmark_mapping.values()):
        if (results.keypoints.shape[1] > 0):
            result_array.append(float(results.keypoints.xyn[0][value][0])) # x normalized
            result_array.append(float(results.keypoints.xyn[0][value][1])) # y normalized
            result_array.append(float(results.keypoints.conf[0][value])) # visibility
        else:
            result_array.append(None)
            result_array.append(None)
            result_array.append(None)

    return array(result_array)

def to_feature_vector(image_path: str, model: YOLO) -> ndarray:
    _, results, _ = predict_landmarks(image_path, model)
    return __to_np_array(results)

def extract_features(image_paths: List[str]) -> List[List[Any]]:
    model = build_pose_model()
    
    def image_2_features(image_path: str) -> List[Any]:
        label = get_label_value_from_path(image_path)
        features = to_feature_vector(image_path, model)
        return [*features, label, image_path]
    
    return list(map(image_2_features, image_paths))