from enum import Enum
from csv import writer as csv_writer, reader as csv_reader
from os import listdir
from os.path import join, split
from typing import Iterator
from pandas import read_csv

class Technique(Enum):
    INVALID = 0
    NONE = 1
    FOOT_SWAP = 2
    OUTSIDE_FLAG = 3
    BACK_FLAG = 4
    INSIDE_FLAG = 5
    #BACK_STEP = 6
    DROP_KNEE = 7
    CROSS_MIDLINE = 8

def name_to_value(name: str) -> int:
    return Technique[name].value

def iterate_valid_labels() -> Iterator[str]:
    return iter([label.name for label in Technique if label != Technique.INVALID])

def get_label(label_path: str, frame_number: int) -> Technique:
    with open(label_path, 'r', newline='') as csvfile:
        reader = csv_reader(csvfile)
        for row in reader:
            current_start = int(row[0])
            current_stop = int(row[1])
            if current_stop <= frame_number:
                continue
            elif current_start <= frame_number and frame_number < current_stop:
                return Technique(int(row[2]))
            else:
                return Technique.INVALID

def get_labels_as_dataframe(label_path):
    return read_csv(label_path, header=None, names=["start", "stop", "label"])

def validate_label(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv_reader(csvfile)
        last_stop = -1
        for idx, row in enumerate(reader):
            #print(f'{idx}: {row}')
            if len(row) != 3:
                print(f'Line {idx}: {row} - Expected three values')
                continue
            if not row[0].isdigit() or not row[1].isdigit() or not row[2].isdigit():
                print(f'Line {idx}: {row} - not a number')
                continue
            if int(row[2]) == 0:
                print(f'Line {idx}: {row} - unnecessary INVALID label found')
                continue
            if int(row[2]) not in Technique:
                print(f'Line {idx}: {row} - third value not a valid technique')
                continue
            if int(row[0]) >= int(row[1]):
                print(f'Line {idx}: {row} - stop must be higher than start')
                continue
            if last_stop > int(row[0]):
                print(f'Line {idx}: {row} - start must be higher or equal to previous stop')
                continue
            last_stop = int(row[1])

def validate_all(root_dir):
    files = listdir(root_dir)
    for file in files:
        file_path = join(root_dir, file)
        print(f'Validating: {file}')
        validate_label(file_path)
    print("Done validating")

def correct_fps(label_path, output_path):
    '''
    Example: 
        correct_fps("./data/labels/How to Flag - A Climbing Technique for Achieving Balance.csv", 
            "./data/labels/How to Flag - A Climbing Technique for Achieving Balance corrected.csv")
    '''
    actual = 29.97
    current = 23.976
    with open(label_path, 'r', newline='') as original:
        reader = csv_reader(original)
            
        with open(output_path, 'w', newline='') as new:
            writer = csv_writer(new)
            for row in reader:
                start = int(row[0])
                stop = int(row[1])
                label = int(row[2])
                writer.writerow([int(start / current * actual), int(stop / current * actual), label])

def get_label_from_path(path) -> Technique:
    head, tail = split(path)
    if head == '':
        raise Exception("Could not find Technique")
    
    if tail in [label.name for label in Technique]:
        return Technique[tail]
    
    return get_label_from_path(head)