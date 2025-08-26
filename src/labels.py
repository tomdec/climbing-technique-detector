from enum import Enum
from csv import writer as csv_writer, reader as csv_reader
from os import listdir, makedirs
from os.path import join, split, exists
from typing import Iterator
from pandas import DataFrame, read_csv

__LABELS_PATH = "data/labels/labels.yml"
__labels = None

if __labels is None:
    if not exists(__LABELS_PATH):
        print(f"Could not load labels, '{__LABELS_PATH}' was not found")
    else:
        from yaml import safe_load
        with open(__LABELS_PATH, 'r') as file:
            __labels = safe_load(file)
            print("loaded labels")

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
    return __labels['values'][name]

def value_to_name(value: int) -> str:
    return Technique(value).name

def iterate_valid_labels() -> Iterator[str]:
    return iter([label.name for label in Technique if label != Technique.INVALID])

def make_label_dirs(root: str):
    for name in iterate_valid_labels():
            label_dir = join(root, name)
            makedirs(label_dir, exist_ok=True)

def get_label_name(label_path: str, frame_number: int) -> str:
    with open(label_path, 'r', newline='') as csvfile:
        reader = csv_reader(csvfile)
        for row in reader:
            current_start = int(row[0])
            current_stop = int(row[1])
            if current_stop <= frame_number:
                continue
            elif current_start <= frame_number and frame_number < current_stop:
                return Technique(int(row[2])).name
            else:
                return Technique.INVALID.name

def get_label_from_path(path: str) -> Technique:
    head, tail = split(path)
    if head == '':
        raise Exception("Could not find Technique")
    
    if tail in [label.name for label in Technique]:
        return Technique[tail]
    
    return get_label_from_path(head)

def get_labels_as_dataframe(label_path) -> DataFrame:
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

def validate_all(labels_path):
    files = listdir(labels_path)
    for file in files:
        file_path = join(labels_path, file)
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