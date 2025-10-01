from ultralytics import YOLO
from cv2.typing import MatLike
from typing import List, Any
from numpy import ndarray

from src.labels import get_label_value_from_path
from src.hpe.yolo.model import build_pose_model
from src.hpe.yolo.landmarks import PredictedLandmarks

def predict_landmarks(image_path: str | MatLike, model: YOLO) -> PredictedLandmarks:
    results = model(image_path)
    return PredictedLandmarks(results[0])

def to_feature_vector(image_path: str, model: YOLO) -> ndarray:
    results = predict_landmarks(image_path, model)
    return results.to_array()

def extract_features(image_paths: List[str]) -> List[List[Any]]:
    model = build_pose_model()
    
    def image_2_features(image_path: str) -> List[Any]:
        label = get_label_value_from_path(image_path)
        features = to_feature_vector(image_path, model)
        return [*features, label, image_path]
    
    return list(map(image_2_features, image_paths))