from cv2.typing import Scalar

from src.common.draw import RED, BLUE, GREEN

class KeypointDrawConfig:

    def __init__(self,
            label: str = "",
            relative_size: float = 0.01,
            relative_thickness: float = 0.2,
            left_color: Scalar = BLUE,
            right_color: Scalar = GREEN,
            center_color: Scalar = RED):
        self.label = label
        self.relative_size = relative_size
        self.relative_thickness = relative_thickness
        self.left_color = left_color
        self.right_color = right_color
        self.center_color = center_color