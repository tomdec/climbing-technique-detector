from mediapipe.python.solutions.holistic import PoseLandmark, HandLandmark
from typing import Mapping
from numpy import concatenate

from src.hpe.common.labels import MyLandmark

_used_pose_landmarks = list([
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.LEFT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_FOOT_INDEX
])

_unused_pose_landmarks = set(PoseLandmark).difference(_used_pose_landmarks)

_used_hand_landmarks = list([
    HandLandmark.THUMB_MCP,
    HandLandmark.THUMB_IP,
    HandLandmark.THUMB_TIP,
    HandLandmark.INDEX_FINGER_MCP,
    HandLandmark.PINKY_MCP
])

_unused_hand_landmarks = set(HandLandmark).difference(_used_hand_landmarks)

_pose_landmark_mapping: Mapping[MyLandmark, PoseLandmark] = {
    MyLandmark.HEAD: PoseLandmark.NOSE,
    MyLandmark.RIGHT_SHOULDER: PoseLandmark.RIGHT_SHOULDER,
    MyLandmark.LEFT_SHOULDER: PoseLandmark.LEFT_SHOULDER,
    MyLandmark.LEFT_ELBOW: PoseLandmark.LEFT_ELBOW,
    MyLandmark.LEFT_WRIST: PoseLandmark.LEFT_WRIST,
    MyLandmark.LEFT_HIP: PoseLandmark.LEFT_HIP,
    MyLandmark.RIGHT_HIP: PoseLandmark.RIGHT_HIP,
    MyLandmark.LEFT_KNEE: PoseLandmark.LEFT_KNEE,
    MyLandmark.RIGHT_KNEE: PoseLandmark.RIGHT_KNEE,
    MyLandmark.LEFT_HEEL: PoseLandmark.LEFT_HEEL,
    MyLandmark.LEFT_FOOT_TIP: PoseLandmark.LEFT_FOOT_INDEX,
    MyLandmark.RIGHT_HEEL: PoseLandmark.RIGHT_HEEL,
    MyLandmark.RIGHT_FOOT_TIP: PoseLandmark.RIGHT_FOOT_INDEX,
    MyLandmark.RIGHT_ELBOW: PoseLandmark.RIGHT_ELBOW,
    MyLandmark.RIGHT_WRIST: PoseLandmark.RIGHT_WRIST,
    MyLandmark.LEFT_ANKLE: PoseLandmark.LEFT_ANKLE,
    MyLandmark.RIGHT_ANKLE: PoseLandmark.RIGHT_ANKLE
}

def get_pose_landmark(key: MyLandmark) -> PoseLandmark | None:
    try:
        return _pose_landmark_mapping[key]
    except KeyError:
        return None
    
_left_hand_landmark_mapping: Mapping[MyLandmark, HandLandmark] = {
    MyLandmark.LEFT_INDEX: HandLandmark.INDEX_FINGER_MCP,
    MyLandmark.LEFT_THUMB_MCP: HandLandmark.THUMB_MCP,
    MyLandmark.LEFT_PINKY: HandLandmark.PINKY_MCP,
    MyLandmark.LEFT_THUMB_IP: HandLandmark.THUMB_IP,
    MyLandmark.LEFT_THUMB_TIP: HandLandmark.THUMB_TIP
}

def get_left_hand_landmark(key: MyLandmark) -> HandLandmark | None:
    try:
        return _left_hand_landmark_mapping[key]
    except KeyError:
        return None

_right_hand_landmark_mapping: Mapping[MyLandmark, HandLandmark] = {
    MyLandmark.RIGHT_INDEX: HandLandmark.INDEX_FINGER_MCP,
    MyLandmark.RIGHT_THUMB_MCP: HandLandmark.THUMB_MCP,
    MyLandmark.RIGHT_PINKY: HandLandmark.PINKY_MCP,
    MyLandmark.RIGHT_THUMB_IP: HandLandmark.THUMB_IP,
    MyLandmark.RIGHT_THUMB_TIP: HandLandmark.THUMB_TIP,
}

def get_right_hand_landmark(key: MyLandmark) -> HandLandmark | None:
    try:
        return _right_hand_landmark_mapping[key]
    except KeyError:
        return None
    
def get_feature_labels():
    def to_pose_features(landmark: PoseLandmark):
        name = landmark.name
        return [f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_visibility']

    def to_hand_features(landmark: HandLandmark, prefix: str):
        name = landmark.name
        return [f'{prefix}_{name}_x', f'{prefix}_{name}_y', f'{prefix}_{name}_z']
    
    pose_feature = concatenate([to_pose_features(x) for x in _used_pose_landmarks])
    right_hand_features = concatenate([to_hand_features(x, "RIGHT") for x in _used_hand_landmarks])
    left_hand_features = concatenate([to_hand_features(x, "LEFT") for x in _used_hand_landmarks])

    return concatenate([pose_feature, right_hand_features, left_hand_features])

