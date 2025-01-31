from enum import Enum
from csv import reader as csv_reader
class Technique(Enum):
    INVALID = 0
    NONE = 1
    FOOT_SWAP = 2
    OUTSIDE_FLAG = 3
    BACK_FLAG = 4
    INSIDE_FLAG = 5
    BACK_STEP = 6
    DROP_KNEE = 7
    CROSS_MIDLINE = 8


def get_label(label_path, frame_number):
    with open(label_path, 'r', newline='') as csvfile:
        reader = csv_reader(csvfile)
        label = Technique.INVALID
        for row in reader:
            current_frame = int(row[0])
            if frame_number >= current_frame:
                label = Technique(int(row[1]))
            else:
                break
        return label