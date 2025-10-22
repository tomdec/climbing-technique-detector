from typing import Dict, List, override
from ultralytics.engine.results import Results
from numpy import ndarray, array

from numpy import concatenate
from src.hpe.common.landmarks import MyLandmark, PredictedKeyPoints, PredictedKeyPoint

class YoloPredictedKeyPoints(PredictedKeyPoints):

    @property
    def values(self) -> Results:
        return self._values

    @override
    def __init__(self, values: Results):
        self._values = values

    @override
    def __getitem__(self, index: MyLandmark) -> PredictedKeyPoint:
        """Get landmark prediction for given index.
        Returns an empty landmark when the landmark was not detected.
        
        For YOLO, when a landmark is not detected but a person is, the landmark will be
        at coordinates (0, 0).

        Args:
            index (MyLandmark): Landmark to get prediction for.

        Raises:
            Exception: When tool cannot predict given landmark.

        Returns:
            PredictedKeyPoint: Landmark prediction.
        """
        pose_landmark = get_pose_landmark(index)
        if pose_landmark is None:
            raise Exception(f"Cannot get prediction for {index}, likely unable to predict this landmark")
        
        if self.no_person_detected():
            return PredictedKeyPoint.empty()
        
        keypoints = self._values.keypoints
        coordinates = keypoints.xyn[0][pose_landmark]
        if coordinates.device.type == 'cuda':
            coordinates = coordinates.cpu()

        visibility = keypoints.conf[0][pose_landmark]
        if visibility.device.type == 'cuda':
            visibility = visibility.cpu()

        return PredictedKeyPoint(float(coordinates[0]), float(coordinates[1]), None, float(visibility))
    
    @override
    def no_person_detected(self) -> bool:
        return len(self._values) == 0
    
    @override
    def can_predict(self, landmark: MyLandmark):
        return can_predict(landmark)

    @override    
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

def can_predict(landmark: MyLandmark) -> bool:
    return landmark in _pose_landmark_mapping.keys()

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

def get_recognizable_landmarks():
    return len(_pose_landmark_mapping.keys())