from cv2 import putText, FONT_HERSHEY_PLAIN
from cv2.typing import MatLike
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.python.solutions.holistic import POSE_CONNECTIONS, HAND_CONNECTIONS
from typing import Mapping, List
from numpy import ndarray
from math import isnan, sqrt, pi
from pandas import Series

from src.common.draw import BLUE, GREEN
from src.hpe.common.typing import PredictedKeyPoint, KeypointDrawConfig
from src.hpe.mp.landmarks import MediaPipePredictedKeyPoints,\
    _pose_landmark_mapping, _left_hand_landmark_mapping, _right_hand_landmark_mapping,\
    used_pose_landmarks, unused_pose_landmarks, used_hand_landmarks, unused_hand_landmarks

PURPLE = (80, 44, 121)

def __get_my_pose_landmark_styles(height: int, width: int, 
        config: KeypointDrawConfig = KeypointDrawConfig()) -> Mapping[int, DrawingSpec]:
    result = {}
    radius = max(1, int(sqrt(config.relative_size * height * width / pi)))
    thickness = max(1, int(config.relative_thickness * radius))
    color = config.right_color

    for key in used_pose_landmarks:
        result[key] = DrawingSpec(color=color, thickness=thickness, circle_radius=radius)
    
    for key in unused_pose_landmarks:
        result[key] = DrawingSpec(color=(1,1,1), thickness=0, circle_radius=1)

    return result

def __get_my_hand_landmark_styles(height: int, width: int, 
        config: KeypointDrawConfig = KeypointDrawConfig()) -> Mapping[int, DrawingSpec]:
    result = {}
    radius = max(1, int(sqrt(config.relative_size * height * width / pi)))
    thickness = max(1, int(config.relative_thickness * radius))
    color = config.right_color

    for key in used_hand_landmarks:
        result[key] = DrawingSpec(color=color, thickness=thickness, circle_radius=radius)
    
    for key in unused_hand_landmarks:
        result[key] = DrawingSpec(color=(1,1,1), thickness=0, circle_radius=1)

    return result

def draw_my_landmarks(image: MatLike, results: MediaPipePredictedKeyPoints,
        config: KeypointDrawConfig = KeypointDrawConfig()) -> MatLike:
    annotated = image.copy()

    image_height, image_width, _ = annotated.shape
    
    radius = max(1, int(sqrt(config.relative_size * image_height * image_width / pi)))
    thickness = max(1, int(config.relative_thickness * radius))
    connection_thinkness = max(1, int(thickness / 2))
    connection_color = PURPLE
    connection_drawing_spec=DrawingSpec(color=connection_color, thickness=connection_thinkness)

    draw_landmarks(annotated, results.pose_landmarks, POSE_CONNECTIONS, 
        landmark_drawing_spec=__get_my_pose_landmark_styles(image_height, image_width, config),
        connection_drawing_spec=connection_drawing_spec)
    
    draw_landmarks(annotated, results.left_hand_landmarks, HAND_CONNECTIONS,
        landmark_drawing_spec=__get_my_hand_landmark_styles(image_height, image_width, config),
        connection_drawing_spec=connection_drawing_spec)
    
    draw_landmarks(annotated, results.right_hand_landmarks, HAND_CONNECTIONS,
        landmark_drawing_spec=__get_my_hand_landmark_styles(image_height, image_width, config),
        connection_drawing_spec=connection_drawing_spec)
    
    return annotated

def draw_features(img: MatLike, features: Series) -> MatLike:
    
    class LandmarkList:

        @staticmethod
        def from_features(features: Series, names: List[str]) -> 'LandmarkList':
            landmarks = [
                PredictedKeyPoint(
                    x=feature[0], 
                    y=feature[1], 
                    z=None, 
                    visibility=1,
                    name=name) 
                for (feature, name) in zip(features, names)]
            return LandmarkList(landmarks)

        def __init__(self, landmarks: List[PredictedKeyPoint]):
            self.landmark = landmarks

    pose_start_idx = 0
    right_hand_start_idx = 4*len(used_pose_landmarks)
    left_hand_start_idx = 4*len(used_pose_landmarks) + 3*len(used_hand_landmarks)
    pose = features.iloc[pose_start_idx:right_hand_start_idx].values.reshape(-1, 4)
    pose_names = [landmark.name for landmark in _pose_landmark_mapping.keys()]
    right_hand = features.iloc[right_hand_start_idx:left_hand_start_idx].values.reshape(-1, 3)
    right_hand_names = [landmark.name for landmark in _right_hand_landmark_mapping.keys()]
    left_hand = features.iloc[left_hand_start_idx:].values.reshape(-1, 3)
    left_hand_names = [landmark.name for landmark in _left_hand_landmark_mapping.keys()]
    
    pose = LandmarkList.from_features(pose, pose_names)
    right_hand = LandmarkList.from_features(right_hand, right_hand_names)
    left_hand = LandmarkList.from_features(left_hand, left_hand_names)

    annotated = img.copy()
    annotated = putText(annotated, 'L', (0, 100), FONT_HERSHEY_PLAIN, 10, BLUE, 10)
    annotated = putText(annotated, 'R', (img.shape[1]-100, 100), FONT_HERSHEY_PLAIN, 10,
        GREEN, 10)

    relative_size=0.0003
    relative_thickness=0.5
    for kp in [*pose.landmark, *right_hand.landmark, *left_hand.landmark]:
        if not isnan(kp.x) and not isnan(kp.y): 
            annotated = kp.draw(annotated, 
                config=KeypointDrawConfig(
                    relative_size=relative_size,
                    relative_thickness=relative_thickness
                ))

    return annotated