from typing import Dict, List
from ultralytics.engine.results import Results
from numpy import ndarray, array

from numpy import concatenate
from src.hpe.common.landmarks import MyLandmark

class PredictedLandmarks:

    @property
    def values(self) -> Results:
        return self._values

    def __init__(self, values: Results):
        self._values = values

    def is_missing(self) -> bool:
        return len(self._values) == 0

    def to_array(self) -> ndarray:
        result_array = []
        for value in list(_pose_landmark_mapping.values()):
            if (self._values.keypoints.shape[1] > 0):
                result_array.append(float(self._values.keypoints.xyn[0][value][0])) # x normalized
                result_array.append(float(self._values.keypoints.xyn[0][value][1])) # y normalized
                result_array.append(float(self._values.keypoints.conf[0][value])) # visibility
            else:
                result_array.append(None)
                result_array.append(None)
                result_array.append(None)

        return array(result_array)
    
    def ensure_empty(self) -> Dict[MyLandmark, bool | None]:
        correct_pred = {}
    
        if self._values.keypoints.xyn.shape[0] == 0: # no person detected
            for landmark in MyLandmark:
                pose_landmark_index = get_pose_landmark(landmark)
                correct_pred[landmark] = True if pose_landmark_index is not None else None
        else:
            for landmark in MyLandmark:
                pose_landmark_index = get_pose_landmark(landmark)
                result = None
                
                if pose_landmark_index is not None:
                    yhat_value = self.get_prediction(0, pose_landmark_index)
                    result = yhat_value is None
                
                correct_pred[landmark] = result

        return correct_pred
    
    def get_prediction(self, object_index: int, key: int) -> ndarray | None:
        """
        Get normalized coordinates of a predicted landmark for a detected object. 
        None, if the landmark is not detected.

        Args:
            object_index (int): Index of a detected object in the results.
            key (int): Index of a landmark.

        Returns:
            (ndarray | None): Normalized coordinates (x, y) of the predicted landmark.
        """
        keypoints = self._values.keypoints
        if keypoints.xyn.shape.count(0) > 0:
            return None
        
        raw_tensor = keypoints.xyn[object_index][key]
        if raw_tensor.device.type == 'cuda':
            return array(raw_tensor.cpu())
        return array(raw_tensor)

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