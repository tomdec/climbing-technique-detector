from mediapipe.python.solutions.holistic import PoseLandmark, HandLandmark
from numpy import concatenate, ndarray, array
from typing import Dict, NamedTuple, Set, override, List, Any

from src.hpe.common.typing import MyLandmark, PredictedKeyPoint
from src.hpe.common.landmarks import PredictedKeyPoints

class MediaPipePredictedKeyPoints(PredictedKeyPoints):

    @staticmethod
    def __find_landmark(index: int, 
            landmarks: List[Any],
            name: str) -> PredictedKeyPoint:
        if landmarks is None:
            return PredictedKeyPoint.empty()
        
        values = landmarks.landmark[index]
        return PredictedKeyPoint(
            x=values.x, 
            y=values.y, 
            z=values.z, 
            visibility=values.visibility, 
            name=name)

    @property
    def pose_landmarks(self):
        return self._values.pose_landmarks
    
    @property
    def right_hand_landmarks(self):
        return self._values.right_hand_landmarks
    
    @property
    def left_hand_landmarks(self):
        return self._values.left_hand_landmarks

    @override
    def __init__(self, values: NamedTuple):
        self._values = values

    @override
    def __getitem__(self, my_landmark: MyLandmark) -> PredictedKeyPoint:
        """Get landmark prediction for given index.
        Returns an empty landmark when the landmark was not detected.
        
        For MediaPipe, when a landmark is not detected but a person, or body part, is, 
        the landmark will be out of bounds.

        Args:
            index (MyLandmark): Landmark to get prediction for.

        Raises:
            Exception: When tool cannot predict given landmark.

        Returns:
            PredictedKeyPoint: Landmark prediction.
        """
        if not self.can_predict(my_landmark):
            raise Exception(f"Cannot get prediction for {my_landmark}, likely unable to predict this landmark")
        
        pose_landmark = get_pose_landmark(my_landmark)
        if pose_landmark is not None:
            return self.__find_landmark(index=pose_landmark, 
                landmarks=self._values.pose_landmarks,
                name=my_landmark.name)
        
        right_hand_landmark = get_right_hand_landmark(my_landmark)
        if right_hand_landmark is not None:
            return self.__find_landmark(index=right_hand_landmark, 
                landmarks=self._values.right_hand_landmarks,
                name=my_landmark.name)
            
        left_hand_landmark = get_left_hand_landmark(my_landmark)
        if left_hand_landmark is not None:
            return self.__find_landmark(index=left_hand_landmark, 
                landmarks=self._values.left_hand_landmarks,
                name=my_landmark.name)

    @override
    def no_person_detected(self):
        return self.pose_landmarks is None and \
            self.right_hand_landmarks is None and \
            self.left_hand_landmarks is None

    @override
    def can_predict(self, landmark: MyLandmark):
        return can_predict(landmark)

    @override
    def to_array(self) -> ndarray:
        result_array = []
    
        for landmark in used_pose_landmarks:
            if (self._values.pose_landmarks is None):
                result_array.append(None)
                result_array.append(None)
                result_array.append(None)
                result_array.append(None)
            else: 
                result_array.append(self._values.pose_landmarks.landmark[landmark].x)
                result_array.append(self._values.pose_landmarks.landmark[landmark].y)
                result_array.append(self._values.pose_landmarks.landmark[landmark].z)
                result_array.append(self._values.pose_landmarks.landmark[landmark].visibility)

        for landmark in used_hand_landmarks:
            if (self._values.right_hand_landmarks is None):
                result_array.append(None)
                result_array.append(None)
                result_array.append(None)
            else: 
                result_array.append(self._values.right_hand_landmarks.landmark[landmark].x)
                result_array.append(self._values.right_hand_landmarks.landmark[landmark].y)
                result_array.append(self._values.right_hand_landmarks.landmark[landmark].z)

        for landmark in used_hand_landmarks:
            if (self._values.left_hand_landmarks is None):
                result_array.append(None)
                result_array.append(None)
                result_array.append(None)
            else: 
                result_array.append(self._values.left_hand_landmarks.landmark[landmark].x)
                result_array.append(self._values.left_hand_landmarks.landmark[landmark].y)
                result_array.append(self._values.left_hand_landmarks.landmark[landmark].z)

        return array(result_array)

_pose_landmark_mapping: Dict[MyLandmark, PoseLandmark] = {
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
    MyLandmark.RIGHT_ANKLE: PoseLandmark.RIGHT_ANKLE,
    MyLandmark.LEFT_EYE: PoseLandmark.LEFT_EYE,
    MyLandmark.RIGHT_EYE: PoseLandmark.RIGHT_EYE,
    MyLandmark.LEFT_EAR: PoseLandmark.LEFT_EAR,
    MyLandmark.RIGHT_EAR: PoseLandmark.RIGHT_EAR
}
used_pose_landmarks: List[PoseLandmark] = list(_pose_landmark_mapping.values())
unused_pose_landmarks: Set[PoseLandmark] = set(PoseLandmark).difference(used_pose_landmarks)

def get_pose_landmark(key: MyLandmark) -> PoseLandmark | None:
    try:
        return _pose_landmark_mapping[key]
    except KeyError:
        return None

_left_hand_landmark_mapping: Dict[MyLandmark, HandLandmark] = {
    MyLandmark.LEFT_INDEX: HandLandmark.INDEX_FINGER_MCP,
    MyLandmark.LEFT_THUMB_MCP: HandLandmark.THUMB_MCP,
    MyLandmark.LEFT_PINKY: HandLandmark.PINKY_MCP,
    MyLandmark.LEFT_THUMB_IP: HandLandmark.THUMB_IP,
    MyLandmark.LEFT_THUMB_TIP: HandLandmark.THUMB_TIP
}
used_hand_landmarks = list(_left_hand_landmark_mapping.values())
unused_hand_landmarks = set(HandLandmark).difference(used_hand_landmarks)

def get_left_hand_landmark(key: MyLandmark) -> HandLandmark | None:
    try:
        return _left_hand_landmark_mapping[key]
    except KeyError:
        return None

_right_hand_landmark_mapping: Dict[MyLandmark, HandLandmark] = {
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
    
    pose_feature = concatenate([to_pose_features(x) for x in used_pose_landmarks])
    right_hand_features = concatenate([to_hand_features(x, "RIGHT") for x in used_hand_landmarks])
    left_hand_features = concatenate([to_hand_features(x, "LEFT") for x in used_hand_landmarks])

    return concatenate([pose_feature, right_hand_features, left_hand_features])

def can_predict(landmark: MyLandmark):
    return landmark in _pose_landmark_mapping.keys() or \
        landmark in _right_hand_landmark_mapping.keys() or \
        landmark in _left_hand_landmark_mapping.keys()

def get_recognizable_landmarks():
    return len(_pose_landmark_mapping.keys()) + \
        len(_right_hand_landmark_mapping.keys()) + \
        len(_left_hand_landmark_mapping.keys())