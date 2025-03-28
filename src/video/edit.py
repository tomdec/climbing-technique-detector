
from cv2 import FILLED, FONT_HERSHEY_SIMPLEX, LINE_AA, getTextSize, rectangle, putText
from cv2.typing import MatLike

from src.labels import Technique 

def write_label(frame: MatLike, label: Technique) -> MatLike:
    return write_text(frame, label.name)

def write_text(frame: MatLike, text: str) -> MatLike:
    result = frame.copy()

    pos = (20, 20)
    bg_color = (255, 0, 0)
    font_face = FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0, 0, 0)
    thickness = FILLED
    margin = 5
    txt_size = getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    rectangle(result, pos, (end_x, end_y), bg_color, thickness)
    putText(result, text, pos, font_face, scale, color, 1, LINE_AA)

    return result