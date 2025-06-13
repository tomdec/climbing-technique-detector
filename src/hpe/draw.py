from cv2.typing import MatLike
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.python.solutions.holistic import POSE_CONNECTIONS, HAND_CONNECTIONS
from typing import Mapping

from src.hpe.landmarks import _used_pose_landmarks, _unused_pose_landmarks, _used_hand_landmarks, _unused_hand_landmarks


_pose_landmark_style = {
	tuple(_used_pose_landmarks): DrawingSpec(color=(80,22,10), thickness=3, circle_radius=5),
	tuple(_unused_pose_landmarks): DrawingSpec(color=(1,1,1), thickness=1, circle_radius=1)
}

def get_my_pose_landmark_styles() -> Mapping[int, DrawingSpec]:
	result = {}
	for keys, value in _pose_landmark_style.items():
		for key in keys:
			result[key] = value
	return result

_hand_landmark_style = {
	tuple(_used_hand_landmarks): DrawingSpec(color=(80,22,10), thickness=3, circle_radius=5),
	tuple(_unused_hand_landmarks): DrawingSpec(color=(1,1,1), thickness=1, circle_radius=1)
}

def get_my_hand_landmark_styles() -> Mapping[int, DrawingSpec]:
	result = {}
	for keys, value in _hand_landmark_style.items():
		for key in keys:
			result[key] = value
	return result

def draw_my_landmarks(image: MatLike, results) -> MatLike:
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