from cv2.typing import MatLike
from mediapipe.python.solutions.holistic import Holistic
from typing import NamedTuple, Tuple, List, Any
from numpy import array, ndarray

from src.common.helpers import imread
from src.labels import get_label_value_from_path
from src.hpe.mp.model import build_holistic_model
from src.hpe.mp.landmarks import PredictedLandmarks

def predict_landmarks(image: MatLike, model: Holistic) -> PredictedLandmarks:
    """Predict landmarks with MediaPipe tool, and wrap results in custom class PredictedLandmarks.

    Args:
        image (MatLike): Image to predict landmarks from.
        model (Holistic): MediaPipe model that predicts the landmarks.

    Returns:
        PredictedLandmarks: Predicted landmark results.
    """
    #image_height, image_width, _ = image.shape
    results = model.process(image)
    return PredictedLandmarks(results)

def to_feature_vector(image: MatLike, model: Holistic) -> ndarray:
    results = predict_landmarks(image, model)
    return results.to_array()

def extract_features(image_paths: List[str]) -> List[List[Any]]:
    """
    Keep at one model per image!! Evaluating images with different sized wit the same model gives incorrect values.
    """
    def image_2_features(image_path: str) -> List[Any]:
        with build_holistic_model() as model:
            image = imread(image_path)
            label = get_label_value_from_path(image_path)
            features = to_feature_vector(image, model)
            return [*features, label, image_path]

    return list(map(image_2_features, image_paths))