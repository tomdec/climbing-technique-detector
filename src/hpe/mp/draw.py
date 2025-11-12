from cv2.typing import MatLike
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.python.solutions.holistic import POSE_CONNECTIONS, HAND_CONNECTIONS
from typing import Mapping, List
from numpy import ndarray
from math import isnan

from src.hpe.common.landmarks import PredictedKeyPoint
from src.hpe.mp.landmarks import MediaPipePredictedKeyPoints,\
    used_pose_landmarks, unused_pose_landmarks, used_hand_landmarks, unused_hand_landmarks

_pose_landmark_style = {
    tuple(used_pose_landmarks): DrawingSpec(color=(80,22,10), thickness=3, circle_radius=5),
    tuple(unused_pose_landmarks): DrawingSpec(color=(1,1,1), thickness=1, circle_radius=1)
}

def get_my_pose_landmark_styles() -> Mapping[int, DrawingSpec]:
    result = {}
    for keys, value in _pose_landmark_style.items():
        for key in keys:
            result[key] = value
    return result

_hand_landmark_style = {
    tuple(used_hand_landmarks): DrawingSpec(color=(80,22,10), thickness=3, circle_radius=5),
    tuple(unused_hand_landmarks): DrawingSpec(color=(1,1,1), thickness=1, circle_radius=1)
}

def get_my_hand_landmark_styles() -> Mapping[int, DrawingSpec]:
    result = {}
    for keys, value in _hand_landmark_style.items():
        for key in keys:
            result[key] = value
    return result

def draw_my_landmarks(image: MatLike, results: MediaPipePredictedKeyPoints) -> MatLike:
    annotated = image.copy()
    
    draw_landmarks(annotated, results.pose_landmarks, POSE_CONNECTIONS, 
        get_my_pose_landmark_styles(), 
        DrawingSpec(color=(80,44,121), thickness=3, circle_radius=5))
    
    draw_landmarks(annotated, results.left_hand_landmarks, HAND_CONNECTIONS,
        get_my_hand_landmark_styles(),
        DrawingSpec(color=(80,44,121), thickness=3, circle_radius=5))
    
    draw_landmarks(annotated, results.right_hand_landmarks, HAND_CONNECTIONS,
        get_my_hand_landmark_styles(),
        DrawingSpec(color=(80,44,121), thickness=3, circle_radius=5))
    
    return annotated

def draw_features(img: MatLike, features: ndarray) -> MatLike:
    
    class LandmarkList:

        @staticmethod
        def from_features(features: ndarray) -> 'LandmarkList':
            landmarks = [PredictedKeyPoint(feature[0], feature[1], None, 1) for feature in features]
            return LandmarkList(landmarks)

        def __init__(self, landmarks: List[PredictedKeyPoint]):
            self.landmark = landmarks

    pose = features.iloc[0:4*len(used_pose_landmarks)].values.reshape(-1, 4)
    right_hand = features.iloc[4*len(used_pose_landmarks):4*len(used_pose_landmarks) + 3*len(used_hand_landmarks)].values.reshape(-1, 3)
    left_hand = features.iloc[4*len(used_pose_landmarks) + 3*len(used_hand_landmarks):].values.reshape(-1, 3)
    
    pose = LandmarkList.from_features(pose)
    right_hand = LandmarkList.from_features(right_hand)
    left_hand = LandmarkList.from_features(left_hand)

    annotated = img.copy()
    relative_size=0.0003
    relative_thickness=0.5
    for kp in [*pose.landmark, *right_hand.landmark, *left_hand.landmark]:
        if not isnan(kp.x) and not isnan(kp.y): 
            annotated = kp.draw(annotated, 
                relative_size=relative_size,
                relative_thickness=relative_thickness)

    return annotated