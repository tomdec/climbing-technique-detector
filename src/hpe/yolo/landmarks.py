from typing import Dict
from src.hpe.common.labels import MyLandmark

_pose_landmark_mapping: Dict[MyLandmark, int] = {
    MyLandmark.HEAD: 0,
    MyLandmark.RIGHT_SHOULDER: 6,
    MyLandmark.LEFT_SHOULDER: 5,
    MyLandmark.LEFT_ELBOW: 7,
    MyLandmark.LEFT_WRIST: 9,
    MyLandmark.LEFT_HIP: 11,
    MyLandmark.RIGHT_HIP: 12,
    MyLandmark.LEFT_KNEE: 13,
    MyLandmark.RIGHT_KNEE: 14,
    MyLandmark.RIGHT_ELBOW: 8,
    MyLandmark.RIGHT_WRIST: 10,
    MyLandmark.LEFT_ANKLE: 15,
    MyLandmark.RIGHT_ANKLE: 16
}

_flipped_pose_landmark_mapping: Dict[MyLandmark, int] = {
    MyLandmark.HEAD: 0,
    MyLandmark.RIGHT_SHOULDER: 5,
    MyLandmark.LEFT_SHOULDER: 6,
    MyLandmark.LEFT_ELBOW: 8,
    MyLandmark.LEFT_WRIST: 10,
    MyLandmark.LEFT_HIP: 12,
    MyLandmark.RIGHT_HIP: 11,
    MyLandmark.LEFT_KNEE: 14,
    MyLandmark.RIGHT_KNEE: 13,
    MyLandmark.RIGHT_ELBOW: 7,
    MyLandmark.RIGHT_WRIST: 9,
    MyLandmark.LEFT_ANKLE: 16,
    MyLandmark.RIGHT_ANKLE: 15
}

def get_pose_landmark(key: MyLandmark) -> int | None:
    try:
        return _pose_landmark_mapping[key]
    except KeyError:
        return None