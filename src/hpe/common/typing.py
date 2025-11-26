from enum import Enum
from math import pi, sqrt
from typing import Dict, override
from cv2 import FONT_HERSHEY_PLAIN, circle, putText
from cv2.typing import Scalar, MatLike
from numpy import array, ndarray

from src.hpe.common.helpers import eucl_distance, LEFT_COLOR, RIGHT_COLOR, CENTER_COLOR

class Visibility(Enum):
    MISSING = 0
    OBSCURED = 1
    VISIBLE = 2

class KeypointDrawConfig:

    def __init__(self,
            label: str = "",
            relative_size: float = 0.01,
            relative_thickness: float = 0.2,
            left_color: Scalar = LEFT_COLOR,
            right_color: Scalar = RIGHT_COLOR,
            center_color: Scalar = CENTER_COLOR):
        self.label = label
        self.relative_size = relative_size
        self.relative_thickness = relative_thickness
        self.left_color = left_color
        self.right_color = right_color
        self.center_color = center_color

class DrawableKeyPoint:
    
    @property
    def x(self) -> float:
        """Normalized x-coordinate of the landmark."""
        return self._x

    @property
    def y(self) -> float:
        """Normalized y-coordinate of the landmark."""
        return self._y
    
    @property
    def name(self) -> str:
        """
        Name of the predicted landmark
        """
        return self._name
    
    def __init__(self, x: float, y: float, name: str):
        self._x = x
        self._y = y
        self._name = name
    
    def __get_coronal_projection(self) -> str:
        """
        Get where the landmark is projected on the coronal plane.
        Possible values: 'L' (left), 'R' (right), 'C' (center)
        This is based on the name, if there is no value for name, defaults to 'C'.
        """
        if self.name is None: return 'C'
        if self.name.startswith('LEFT'): return 'L'
        if self.name.startswith('RIGHT'): return 'R'
        return 'C'

    def __get_color(self, config: KeypointDrawConfig) -> Scalar:
        cp = self.__get_coronal_projection()
        if cp == 'L': return config.left_color
        if cp == 'R': return config.right_color
        return config.center_color

    def is_missing(self) -> bool:
        is_origin = (self._x == 0.0) and (self._y == 0.0)
        is_out_of_bounds = (self._x < 0.0) or (1.0 < self._x) \
            or (self._y < 0.0) or (1.0 < self._y)

        return is_origin or is_out_of_bounds

    def draw(self,
            image: MatLike,
            label: str = "",
            config: KeypointDrawConfig = KeypointDrawConfig()) -> MatLike:
        result = image.copy()
        if self.is_missing():
            return result

        image_height, image_width, _ = result.shape
        center = (int(self._x * image_width), int(self._y * image_height))
        radius = max(1, int(sqrt(config.relative_size * image_height * image_width / pi)))
        thickness = max(1, int(config.relative_thickness * radius))
        color = self.__get_color(config)
        result = circle(result, center, radius, color, thickness)
        result = putText(result, label, center, FONT_HERSHEY_PLAIN, 10, (150, 1, 1), 10)
        return result


class LabelKeyPoint(DrawableKeyPoint):

    @staticmethod
    def from_dict(values: dict, name: str) -> 'LabelKeyPoint':
        return LabelKeyPoint(
            x=values['x'],
            y=values['y'],
            visibility=values['visibility'],
            name=name)

    _visibility: Visibility

    def __init__(self, x, y, visibility, name: str | None = None):
        DrawableKeyPoint.__init__(self, x=float(x), y=float(y), 
            name="" if name is None else name)
        self._visibility = visibility\
            if type(visibility) is Visibility\
            else Visibility(int(visibility))
        
    def __str__(self):
        return f"{self.as_dict()}"

    @override
    def is_missing(self) -> bool:
        return self._visibility == Visibility.MISSING

    def as_array(self) -> ndarray:
        return array([self._x, self._y])

    def as_dict(self) -> dict:
        return {
            'x': self._x,
            'y': self._y,
            'visibility': self._visibility,
            #'name': self._name
        }

class PredictedKeyPoint(DrawableKeyPoint):

    @staticmethod
    def from_dict(values: dict, name: str) -> 'PredictedKeyPoint':
        return PredictedKeyPoint(
            x=values['x'],
            y=values['y'],
            z=values['z'],
            visibility=values['visibility'],
            name=name
        )

    @staticmethod
    def empty() -> 'PredictedKeyPoint':
        """
        To be used when a landmark is missing because the person (or body part) is not detected
        """
        return PredictedKeyPoint(x=0.0, y=0.0, z=None, visibility=1, name="")

    @property
    def z(self) -> float | None:
        """Estimated z-coordinate (or depth) of the landmark. Uses scaling of x-axis."""
        return self._z

    @property
    def visibility(self) -> float:
        """Likelihood of the landmark being visible, in range [0.0, 1.0]."""
        return self._visibility

    def __init__(self,
            x: float,
            y: float,
            z: float | None,
            visibility: float,
            name: str):
        DrawableKeyPoint.__init__(self, x, y, name)
        self._z = z
        self._visibility = visibility

    def __str__(self) -> str:
        return f"{self.as_dict()}"

    def as_array(self) -> ndarray:
        return array([self._x, self._y])

    def as_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'visibility': self.visibility,
            #'name': self.name
        }

class HpeEstimation:

    @staticmethod
    def from_dict(values: dict) -> 'HpeEstimation':
        return HpeEstimation(
            name=values['name'],
            true_landmark= None 
                if values['true_landmark'] is None 
                else LabelKeyPoint.from_dict(
                    values['true_landmark'],
                    name=values['name']),
            predicted_landmark=None 
                if values['predicted_landmark'] is None 
                else PredictedKeyPoint.from_dict(
                    values=values['predicted_landmark'],
                    name=values['name']),
            head_bone_link=values['head_bone_link'],
            image_path=values['image_path'],
            can_predict=values['can_predict']
        )
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def true_landmark(self) -> LabelKeyPoint | None:
        return self._true_landmark

    @property
    def predicted_landmark(self) -> PredictedKeyPoint:
        return self._predicted_landmark

    @property
    def head_bone_link(self) -> float | None:
        return self._head_bone_link

    @property
    def image_path(self) -> str:
        return self._image_path

    @property
    def can_predict(self) -> bool:
        return self._can_predict

    def __init__(self, 
            name: str,
            true_landmark: LabelKeyPoint | None,
            predicted_landmark: PredictedKeyPoint,
            head_bone_link: float | None,
            image_path: str,
            can_predict: bool):
        self._name = name
        self._true_landmark = true_landmark
        self._predicted_landmark = predicted_landmark
        self._head_bone_link = head_bone_link
        self._image_path = image_path
        self._can_predict = can_predict

    def __str__(self) -> str:
        return f"{self.as_dict()}"

    def _get_prediction(self, conf_threshold: float, verbose: bool) -> PredictedKeyPoint | None:
        if self._predicted_landmark.is_missing():
            if verbose: print("No prediction because landmark is missing")
            return None
        elif self._predicted_landmark.visibility < conf_threshold:
            if verbose: print("No prediction because confidence is too low")
            return None
        else:
            if verbose: print("Prediction is made")
            return self._predicted_landmark

    def get_distance(self, verbose: bool = False) -> float:
        true = self._true_landmark.as_array()
        prediction = self._predicted_landmark.as_array()
        distance = eucl_distance(true, prediction)

        if verbose:
            print(f"True landmark: x = {true[0]}, y = {true[1]}")
            print(f"Predicted landmark: x = {prediction[0]}, y = {prediction[1]}")
            print(f"Euclidian distance: {distance}")

        return distance
    
    def get_relative_distance(self, verbose: bool = False) -> float:
        abs_dist = self.get_distance(verbose=verbose)
        return abs_dist / (self.head_bone_link / 2)

    def is_detected(self) -> bool:
        """Checks if a labelled landmark is detected or not.

        Returns:
            bool: True when a landmark is detected, False when the landmark was not labeled or
            not estimated.
        """
        return self.is_present() and \
            not self.predicted_landmark.is_missing()
    
    def is_present(self) -> bool:
        """Checks if the landmark is present.

        Returns:
            bool: True if the landmark is labeled, else False.
        """
        return self.true_landmark is not None and \
            not self.true_landmark.is_missing()

    def is_correct(self) -> bool:
        """Checks if the estimation is correct, according to the PCKh50 metric, 
        without considering a confidence threshold.

        Returns:
            bool: True if labeled and estimated landmark are close enough, else False.
        """
        return self.prediction_result() == "TP"

    def is_present_and_recognizable(self) -> bool:
        return self.can_predict and self.is_present()

    def prediction_result(self, conf_threshold: float = 0.0,
            verbose: bool = False) -> str:
        if not self._can_predict:
            return ""

        estimation = self._get_prediction(conf_threshold, verbose)
        if self._true_landmark is None and estimation is None:
            if verbose: print("TN")
            return "TN"
        elif self._true_landmark is not None and estimation is None:
            if verbose: print("FN")
            return "FN"
        elif self._true_landmark is None and estimation is not None:
            if verbose: print("FP")
            return "FP"
        else:
            limit = self._head_bone_link / 2
            distance = self.get_distance(verbose)
            correct = distance <= limit
            #TODO: maybe this needs to be different?
            #Should an incorrect landmark count as both:
            # - a FP, for the predicted landmark that is incorrect
            # - and FN, for the true landmark that is not detected?
            result = "TP" if correct else "FP"

            if verbose:
                print(f"limit: {limit}")
                print(result)

            return result

    def draw(self, image: MatLike,
            prediction_config: KeypointDrawConfig = KeypointDrawConfig()):
        
        pass

    def as_dict(self) -> dict:
        return {
            'name': self._name,
            'true_landmark': None if self._true_landmark is None else self._true_landmark.as_dict(),
            'predicted_landmark': self._predicted_landmark.as_dict(),
            'head_bone_link': self._head_bone_link,
            'image_path': self._image_path,
            'can_predict': self._can_predict
        }

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

PerformanceMap = Dict[MyLandmark, bool | None]
"""
Dictionary that maps each value of MyLandmark to either:
- True: the landmark was correctly detected
- False: the landmark was not (correctly) detected
- None: the tool cannot detect the landmark
"""        