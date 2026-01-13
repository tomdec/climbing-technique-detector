from cv2 import FONT_HERSHEY_PLAIN, circle, putText
from cv2.typing import Scalar, MatLike
from math import pi, sqrt

from src.hpe.common.typing.KeypointDrawConfig import KeypointDrawConfig

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

    def draw(self, image: MatLike, label: str = "", 
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