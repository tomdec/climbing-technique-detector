from cv2.typing import Scalar

from src.hpe.common.helpers import CENTER_COLOR, LEFT_COLOR, RIGHT_COLOR

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