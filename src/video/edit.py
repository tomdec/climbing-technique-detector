
from cv2 import FILLED, FONT_HERSHEY_SIMPLEX, LINE_AA, getTextSize, rectangle, putText

from src.labels import Technique 

def write_label(frame, label: Technique):
    pos = (20, 20)
    bg_color = (255, 0, 0)
    font_face = FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0, 0, 0)
    thickness = FILLED
    margin = 5
    txt_size = getTextSize(label.name, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    rectangle(frame, pos, (end_x, end_y), bg_color, thickness)
    putText(frame, label.name, pos, font_face, scale, color, 1, LINE_AA)