from cv2.typing import MatLike
from mediapipe.python.solutions.holistic import Holistic
from typing import NamedTuple, Tuple
from numpy import array

from src.hpe.mp.landmarks import _used_pose_landmarks, _used_hand_landmarks

def predict_landmarks(image: MatLike, model: Holistic) -> Tuple[NamedTuple, Tuple[int, int]]:
    """
    Returns:
        (results, image_shape): 
        - results: NamedTuple object containing the HPE results from the Holistic model.
        - image_shape: Shape, (image_height, image_width), of the processed image. 
    """
    image_height, image_width, _ = image.shape
    
    results = model.process(image)                # Make prediction

    return results, (image_height, image_width)

def __to_np_array(results: NamedTuple):
    result_array = []
    
    for landmark in _used_pose_landmarks:
        if (results.pose_landmarks is None):
            result_array.append(None)
            result_array.append(None)
            result_array.append(None)
            result_array.append(None)
        else: 
            result_array.append(results.pose_landmarks.landmark[landmark].x)
            result_array.append(results.pose_landmarks.landmark[landmark].y)
            result_array.append(results.pose_landmarks.landmark[landmark].z)
            result_array.append(results.pose_landmarks.landmark[landmark].visibility)

    for landmark in _used_hand_landmarks:
        if (results.right_hand_landmarks is None):
            result_array.append(None)
            result_array.append(None)
            result_array.append(None)
        else: 
            result_array.append(results.right_hand_landmarks.landmark[landmark].x)
            result_array.append(results.right_hand_landmarks.landmark[landmark].y)
            result_array.append(results.right_hand_landmarks.landmark[landmark].z)

    for landmark in _used_hand_landmarks:
        if (results.left_hand_landmarks is None):
            result_array.append(None)
            result_array.append(None)
            result_array.append(None)
        else: 
            result_array.append(results.left_hand_landmarks.landmark[landmark].x)
            result_array.append(results.left_hand_landmarks.landmark[landmark].y)
            result_array.append(results.left_hand_landmarks.landmark[landmark].z)

    return array(result_array)

def to_feature_vector(model: Holistic, image: MatLike):
    results, _ = predict_landmarks(image, model)
    return __to_np_array(results)