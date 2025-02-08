from mediapipe.python.solutions.holistic import PoseLandmark, HandLandmark
from typing import Mapping
from enum import Enum
from os.path import dirname, join
from sys import path

this_dir = dirname( __file__ )
mymodule_dir = join( this_dir, '..' )
path.append( mymodule_dir )

class MyLandmark(Enum):
	HEAD = 0
	RIGHT_SHOULDER = 1
	LEFT_SHOULDER = 2
	NECK = 3
	LEFT_ELBOW = 4
	LEFT_WRIST = 5
	LEFT_INDEX = 6
	LEFT_THUMB_MCP = 7
	LEFT_PINKY = 8
	LEFT_THUMB_IP = 9
	LEFT_THUMB_TIP = 10
	LEFT_HIP = 11
	RIGHT_HIP = 12
	LEFT_KNEE = 13
	RIGHT_KNEE = 14
	LEFT_HEEL = 15
	LEFT_FOOT_TIP = 16
	RIGHT_HEEL = 17
	RIGHT_FOOT_TIP = 18
	RIGHT_ELBOW = 19
	RIGHT_WRIST = 20
	RIGHT_PINKY = 21
	RIGHT_INDEX = 22
	RIGHT_THUMB_TIP = 23
	RIGHT_THUMB_MCP = 24
	RIGHT_THUMB_IP = 25
	LEFT_ANKLE = 26
	RIGHT_ANKLE = 27

_used_pose_landmarks = set([
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

_used_hand_landmarks = set([
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