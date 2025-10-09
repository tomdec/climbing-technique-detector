from cv2 import COLOR_RGB2BGR, cvtColor
from cv2.typing import MatLike

def convert_to_bgr(image: MatLike) -> MatLike:
    return cvtColor(image, COLOR_RGB2BGR)