from enum import Enum
from csv import writer as csv_writer, reader as csv_reader
from os import listdir
from os.path import join

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
 
def validate_label(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv_reader(csvfile)
        last_frame = -1
        for idx, row in enumerate(reader):
            #print(f'{idx}: {row}')
            if len(row) != 2:
                print(f'Line {idx}: {row} - Expected a pair of values')
                continue
            if not row[0].isdigit() or not row[1].isdigit():
                print(f'Line {idx}: {row} - not a number')
                continue
            if int(row[1]) not in Technique:
                print(f'Line {idx}: {row} - second value not a valid technique')
                continue
            if last_frame >= int(row[0]):
                print(f'Line {idx}: {row} - expected strictly increasing frame numbers')
                continue
            last_frame = int(row[0])

def validate_all(root_dir):
    files = listdir(root_dir)
    for file in files:
        file_path = join(root_dir, file)
        print(f'Validating: {file}')
        validate_label(file_path)

def correct_fps(label_path, output_path):
    '''
    Example: 
        correct_fps("../data/labels/How to Flag - A Climbing Technique for Achieving Balance.csv", 
            "../data/labels/How to Flag - A Climbing Technique for Achieving Balance corrected.csv")
    '''
    
    actual = 29.97
    current = 23.976
    with open(label_path, 'r', newline='') as original:
        reader = csv_reader(original)
        
    
        with open(output_path, 'w', newline='') as new:
            writer = csv_writer(new)
            for row in reader:
                frame = int(row[0])
                label = int(row[1])
                writer.writerow([int(frame / current * actual), label])