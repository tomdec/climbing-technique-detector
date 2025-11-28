from enum import Enum

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
    LEFT_EYE = 28
    RIGHT_EYE = 29
    LEFT_EAR = 30
    RIGHT_EAR = 31

CONNECTIONS = frozenset([
    (MyLandmark.HEAD, MyLandmark.NECK),
    (MyLandmark.HEAD, MyLandmark.LEFT_EYE),
    (MyLandmark.HEAD, MyLandmark.RIGHT_EYE),
    (MyLandmark.NECK, MyLandmark.RIGHT_SHOULDER),
    (MyLandmark.NECK, MyLandmark.LEFT_SHOULDER),
    (MyLandmark.RIGHT_SHOULDER, MyLandmark.LEFT_SHOULDER),
    (MyLandmark.RIGHT_SHOULDER, MyLandmark.RIGHT_ELBOW),
    (MyLandmark.RIGHT_SHOULDER, MyLandmark.RIGHT_HIP),
    (MyLandmark.LEFT_SHOULDER, MyLandmark.LEFT_ELBOW),
    (MyLandmark.LEFT_SHOULDER, MyLandmark.LEFT_HIP),
    (MyLandmark.LEFT_ELBOW, MyLandmark.LEFT_WRIST),
    (MyLandmark.LEFT_WRIST, MyLandmark.LEFT_INDEX),
    (MyLandmark.LEFT_WRIST, MyLandmark.LEFT_PINKY),
    (MyLandmark.LEFT_WRIST, MyLandmark.LEFT_THUMB_MCP),
    (MyLandmark.LEFT_INDEX, MyLandmark.LEFT_PINKY),
    (MyLandmark.LEFT_THUMB_MCP, MyLandmark.LEFT_THUMB_IP),
    (MyLandmark.LEFT_THUMB_IP, MyLandmark.LEFT_THUMB_TIP),
    (MyLandmark.LEFT_HIP, MyLandmark.RIGHT_HIP),
    (MyLandmark.LEFT_HIP, MyLandmark.LEFT_KNEE),
    (MyLandmark.RIGHT_HIP, MyLandmark.RIGHT_KNEE),
    (MyLandmark.LEFT_KNEE, MyLandmark.LEFT_ANKLE),
    (MyLandmark.RIGHT_KNEE, MyLandmark.RIGHT_ANKLE),
    (MyLandmark.LEFT_HEEL, MyLandmark.LEFT_ANKLE),
    (MyLandmark.LEFT_HEEL, MyLandmark.LEFT_FOOT_TIP),
    (MyLandmark.LEFT_FOOT_TIP, MyLandmark.LEFT_ANKLE),
    (MyLandmark.RIGHT_HEEL, MyLandmark.RIGHT_ANKLE),
    (MyLandmark.RIGHT_HEEL, MyLandmark.RIGHT_FOOT_TIP),
    (MyLandmark.RIGHT_FOOT_TIP, MyLandmark.RIGHT_ANKLE),
    (MyLandmark.RIGHT_ELBOW, MyLandmark.RIGHT_WRIST),
    (MyLandmark.RIGHT_WRIST, MyLandmark.RIGHT_INDEX),
    (MyLandmark.RIGHT_WRIST, MyLandmark.RIGHT_PINKY),
    (MyLandmark.RIGHT_WRIST, MyLandmark.RIGHT_THUMB_MCP),
    (MyLandmark.RIGHT_INDEX, MyLandmark.RIGHT_PINKY),
    (MyLandmark.RIGHT_THUMB_MCP, MyLandmark.RIGHT_THUMB_IP),
    (MyLandmark.RIGHT_THUMB_IP, MyLandmark.RIGHT_THUMB_TIP),
    (MyLandmark.LEFT_EYE, MyLandmark.LEFT_EAR),
    (MyLandmark.RIGHT_EYE, MyLandmark.RIGHT_EAR),
])