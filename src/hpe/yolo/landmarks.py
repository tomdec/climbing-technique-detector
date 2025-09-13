from typing import Dict, List

from numpy import concatenate
from src.hpe.common.landmarks import MyLandmark

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
    MyLandmark.RIGHT_ANKLE: 16,
    MyLandmark.LEFT_EYE: 2,
    MyLandmark.RIGHT_EYE: 3,
    MyLandmark.LEFT_EAR: 4,
    MyLandmark.RIGHT_EAR: 5 
}

def get_pose_landmark(key: MyLandmark) -> int | None:
    try:
        return _pose_landmark_mapping[key]
    except KeyError:
        return None
    
def get_feature_labels() -> List[str]:
    def get_features(landmark: MyLandmark) -> List[str]:
        name = landmark.name
        return [f'{name}_x', f'{name}_y', f'{name}_visibility']

    used_landmarks = list(_pose_landmark_mapping.keys())
    return concatenate(list(map(get_features, used_landmarks)))