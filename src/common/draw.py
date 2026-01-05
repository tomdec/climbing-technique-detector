from cv2 import FILLED, FONT_HERSHEY_SIMPLEX, LINE_AA, getTextSize, rectangle, putText
from cv2.typing import MatLike, Scalar
from typing import Tuple

from src.labels import value_to_name

# cv2 uses BGR colors!!
BLACK: Scalar = (0, 0, 0)
RED: Scalar = (1,1,100)
GREEN: Scalar = (1,100,1)
BLUE: Scalar = (100,1,1)

def write_label(frame: MatLike, label: int | str) -> MatLike:
    if type(label) is int:
        label = value_to_name(label)
    return write_text(frame, label)

def write_label_and_prediction(frame: MatLike, label: int | str, prediction: str) -> MatLike:
    if type(label) is int:
        label = value_to_name(label)

    frame = write_label(frame, label)

    prediction_color = GREEN if label == prediction else RED
    prediction_position = (20, 40)

    return write_text(frame, prediction, prediction_color, prediction_position)

def write_text(frame: MatLike, text: str, background_color: Scalar = BLUE, 
        position: Tuple[int, int] = (20, 20)) -> MatLike:
    result = frame.copy()

    pos = position
    bg_color = background_color
    font_face = FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = BLACK
    thickness = FILLED
    margin = 5
    txt_size = getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    rectangle(result, pos, (end_x, end_y), bg_color, thickness)
    putText(result, text, pos, font_face, scale, color, 1, LINE_AA)

    return result