from enum import Enum
from numpy import array, ndarray
from math import sqrt, pi
from cv2.typing import MatLike, Scalar
from cv2 import circle, putText, FONT_HERSHEY_PLAIN
from typing import List, Dict

from src.common.helpers import raise_not_implemented_error
from src.hpe.common.helpers import LEFT_COLOR, RIGHT_COLOR, CENTER_COLOR
from src.hpe.common.helpers import eucl_distance

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
    
class Visibility(Enum):
    MISSING = 0
    OBSCURED = 1
    VISIBLE = 2

class BoundingBox:

    _x: float
    _y: float
    _x_width: float
    _y_width: float

    def __init__(self, x, y, x_width, y_width):
        """
        Args:
            x: x-coordinate of the center
            y: y-coordinate of the center
            x_width: width of the bounding box
            y_width: height of the bounding box
        """
        self._x = float(x)
        self._y = float(y)
        self._x_width = float(x_width)
        self._y_width = float(y_width)

    def __str__(self):
        return f"{{'x': {self._x}, 'y': {self._y}, 'x_width': {self._x_width}, 'y_width': {self._y_width}}}"
    
    def distance_to(self, point) -> float:
        return eucl_distance(array([self._x, self._y]), point)

class KeyPoint:

    @staticmethod
    def from_dict(values: dict) -> 'KeyPoint':
        return KeyPoint(
            x=values['x'], 
            y=values['y'],
            visibility=values['visibility'])

    _x: float
    _y: float
    _visibility: Visibility

    def __init__(self, x, y, visibility):
        self._x = float(x)
        self._y = float(y)
        self._visibility = visibility if type(visibility) is Visibility else Visibility(int(visibility))

    def __str__(self):
        return f"{self.as_dict()}"
    
    def draw(self, image: MatLike, label: str = "") -> MatLike:
        result = image.copy()
        if self.is_missing():
            return result

        image_height, image_width, _ = result.shape
        center = (int(self._x * image_width), int(self._y * image_height))
        
        result = circle(result, center, 25, (1,100,1), 10)
        result = putText(result, label, center, FONT_HERSHEY_PLAIN, 10, (150, 1, 1), 10)
        return result
    
    def is_missing(self) -> bool:
        return self._visibility == Visibility.MISSING

    def as_array(self) -> ndarray:
        return array([self._x, self._y])
    
    def as_dict(self) -> dict:
        return {
            'x': self._x,
            'y': self._y,
            'visibility': self._visibility
        }

class KeyPoints:
    _values: List[KeyPoint]

    def __init__(self, values):
        self._values = list()
        start_index = 0
        while(len(values) >= start_index + 3):
            stop_index = start_index + 3
            self._values.append(KeyPoint(*values[start_index:stop_index]))
            start_index = stop_index

    def __getitem__(self, key: MyLandmark | int):
        if (key is int):
            return self._values[key]
        else:
            return self._values[key.value]
        
    def __str__(self):
        return f"[{",\n".join([str(kp) for kp in self._values])}]"
    
    def draw(self, image: MatLike) -> MatLike:
        result = image.copy()
        for index, value in enumerate(self._values):
            if (value._visibility == Visibility.MISSING):
                continue
            result = value.draw(result, str(index))
        return result
    
class YoloLabels:

    _class_index : int
    _bounding_box: BoundingBox
    _key_points: KeyPoints

    def __init__(self, text: str):
        values = text.split(" ")
        self._class_index = int(values[0])
        self._bounding_box = BoundingBox(*values[1:5])
        self._key_points = KeyPoints(values[5:])

    def __str__(self):
        class_str = f"'class': {self._class_index}"
        box_str = f"'bounding_box': {self._bounding_box}"
        key_points_str =f"'key_points': {self._key_points}"

        return f"{{{class_str}, {box_str}, {key_points_str}}}"

    def draw(self, image: MatLike) -> MatLike:
        return self._key_points.draw(image)

    def get_head_bone_link(self) -> float:
        head = self._key_points[MyLandmark.HEAD]
        neck = self._key_points[MyLandmark.NECK]

        return eucl_distance(head.as_array(), neck.as_array())
    
    def get_keypoint(self, key: MyLandmark) -> KeyPoint:
        return self._key_points[key]
    
    def distance_to(self, point) -> float:
        return self._bounding_box.distance_to(point)

class PredictedKeyPoint:

    @staticmethod
    def from_dict(values: dict) -> 'PredictedKeyPoint':
        return PredictedKeyPoint(
            x=values['x'],
            y=values['y'],
            z=values['z'],
            visibility=values['visibility']
        )

    @staticmethod
    def empty() -> 'PredictedKeyPoint':
        """To be used when a landmark is missing because the person (or body part) is not detected"""
        return PredictedKeyPoint(0.0, 0.0, None, 1)

    @property
    def x(self) -> float:
        """Normalized x-coordinate of the landmark."""
        return self._x
    
    @property
    def y(self) -> float:
        """Normalized y-coordinate of the landmark."""
        return self._y
    
    @property
    def z(self) -> float | None:
        """Estimated z-coordinate (or depth) of the landmark. Uses scaling of x-axis."""
        return self._z
    
    @property
    def visibility(self) -> float:
        """Likelihood of the landmark being visible, in range [0.0, 1.0]."""
        return self._visibility
    
    @property
    def name(self) -> str:
        """
        Name of the predicted landmark
        """
        return self._name

    def __init__(self, 
            x: float, 
            y: float, 
            z: float | None, 
            visibility: float,
            name: str | None):
        self._x = x
        self._y = y
        self._z = z
        self._visibility = visibility
        self._name = name

    def __str__(self) -> str:
        return f"{self.as_dict()}"
    
    def get_color(self) -> Scalar:
        cp = self.get_coronal_projection()
        if cp == 'L': return LEFT_COLOR
        if cp == 'R': return RIGHT_COLOR
        return CENTER_COLOR

    def draw(self, 
            image: MatLike, 
            label: str = "",
            relative_size: float = 0.01,
            relative_thickness: float = 0.2) -> MatLike:
        result = image.copy()
        if self.is_missing():
            return result
        
        image_height, image_width, _ = result.shape
        center = (int(self._x * image_width), int(self._y * image_height))
        radius = max(1, int(sqrt(relative_size * image_height * image_width / pi)))
        thickness = max(1, int(relative_thickness * radius))
        color = self.get_color()
        result = circle(result, center, radius, color, thickness)
        result = putText(result, label, center, FONT_HERSHEY_PLAIN, 10, (150, 1, 1), 10)
        return result

    def is_missing(self) -> bool:
        is_origin = (self._x == 0.0) and (self._y == 0.0)
        is_out_of_bounds = (self._x < 0.0) or (1.0 < self._x) \
            or (self._y < 0.0) or (1.0 < self._y)
        
        return is_origin or is_out_of_bounds

    def get_coronal_projection(self) -> str:
        """
        Get where the landmark is projected on the coronal plane.
        Possible values: 'L' (left), 'R' (right), 'C' (center)
        This is based on the name, if there is no value for name, defaults to 'C'.
        """
        if self.name is None: return 'C'
        if self.name.startswith('LEFT'): return 'L'
        if self.name.startswith('RIGHT'): return 'R'
        return 'C'


    def as_array(self) -> ndarray:
        return array([self._x, self._y])
    
    def as_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'visibility': self.visibility
        }

class PredictedKeyPoints:

    def __init__(self):
        pass
    
    def __getitem__(self, index: MyLandmark) -> PredictedKeyPoint:
        """Get landmark prediction for given index.
        Returns an empty landmark when the landmark was not detected.

        Args:
            index (MyLandmark): Landmark to get prediction for.

        Raises:
            Exception: When tool cannot predict given landmark.

        Returns:
            PredictedKeyPoint: Landmark prediction.
        """
        raise_not_implemented_error(self.__class__.__name__, self.__getitem__.__name__)

    def no_person_detected(self):
        raise_not_implemented_error(self.__class__.__name__, self.no_person_detected.__name__)

    def can_predict(self, landmark: MyLandmark) -> bool:
        """Check if the tool can predict this landmark.

        Args:
            landmark (MyLandmark): Landmark to check.
        """
        raise_not_implemented_error(self.__class__.__name__, self.can_predict.__name__)

    def ensure_empty(self) -> Dict[MyLandmark, bool | None]:
        """
        In the absence of a true object to detect landmarks,
        return a mapping of MyLandmark to booleans:\n
            - true, meaning no prediction (TN), 
            - false, meaning an incorrect prediction (FP),
            - None, if mediapipe is not able to predict this landmark.
        """
        def no_prediction_for_landmark(landmark: MyLandmark) -> bool | None:
            if not self.can_predict(landmark):
                return None
            elif self.no_person_detected():
                return True
            else:
                yhat = self[landmark]
                return yhat is None

        results = {}
        for landmark in MyLandmark:
            results[landmark] = no_prediction_for_landmark(landmark)
        return results

    def to_array(self) -> ndarray:
        raise_not_implemented_error(self.__class__.__name__, self.to_array.__name__)

def build_yolo_labels(file_path: str) -> List[YoloLabels]:
    result = list()
    
    with open(file_path, 'r') as file:
        file_text = file.read()
    
    if file_text == '':
        return result

    file_lines = file_text.split("\n")
    for line in file_lines:
        result.append(YoloLabels(line))

    return result

def get_most_central(label_list: List[YoloLabels]) -> YoloLabels:
    image_center = array([0.5, 0.5])
    result = label_list[0]
    min_distance = label_list[0].distance_to(image_center)
    
    for index in range(1, len(label_list)):
        distance = label_list[index].distance_to(image_center)
        if distance < min_distance:
            min_distance = distance
            result = label_list[index]
    
    return result

def get_mylandmark_count():
    return len(MyLandmark)