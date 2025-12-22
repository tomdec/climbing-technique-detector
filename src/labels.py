from enum import Enum
from csv import writer as csv_writer, reader as csv_reader
from os import listdir, makedirs
from os.path import join, exists
from typing import Iterator, List
from pandas import DataFrame, read_csv

__LABELS_PATH = "data/labels/labels.yml"
__labels = None
LabelsCSV = DataFrame

if __labels is None:
    if not exists(__LABELS_PATH):
        print(f"Could not load labels, '{__LABELS_PATH}' was not found")
    else:
        from yaml import safe_load
        with open(__LABELS_PATH, 'r') as file:
            __labels = safe_load(file)
            print("loaded labels")

def get_valid_label_count():
    return len(__labels['values']) - 1

def get_dataset_name() -> str:
    return __labels['name']

def name_to_value(name: str) -> int:
    return __labels['values'].index(name)

def value_to_name(value: int) -> str:
    if value < 0:
        raise IndexError("Expected values to be non-negative.")
    return __labels['values'][value]

def iterate_valid_labels() -> Iterator[str]:
    return iter([name for (value, name) in enumerate(__labels['values']) if value > 0])

def make_label_dirs(root: str):
    for name in iterate_valid_labels():
            label_dir = join(root, name)
            makedirs(label_dir, exist_ok=True)

def get_label_value_from_path(path: str) -> int:
    for (value, name) in enumerate(__labels['values']):
        if path.__contains__(f"/{name}/"):
            return value
        
    raise Exception(f"did not find label in path: {path}")

def get_labels_as_dataframe(label_path) -> LabelsCSV:
    return read_csv(label_path, header=None, names=["start", "stop", "label", "cvs_start"])

def get_label_by_frame_num(labels: LabelsCSV, frame_num: int) -> str:
    row = labels.query(f"start <= {frame_num} and {frame_num} < stop")
    if len(row) == 0:
        return value_to_name(0)
    return value_to_name(row.iloc[0]['label'])

def get_labels_from_video(video_path: str) -> str:
    return video_path.replace("/videos/", "/labels/").replace(".mp4", ".csv")

def find_valid_segments(label_path: str):
    labels = get_labels_as_dataframe(label_path)

    valid_segments = []
    current_segment = None
    for _, row in labels.iterrows():
        start = row["start"]
        stop = row["stop"]

        if current_segment is None:
            current_segment = [start, stop]
        elif current_segment[1] == start:
            current_segment = [current_segment[0], stop]
        else:
            valid_segments.append(current_segment)
            current_segment = [start, stop]

    print(f"Reduced {labels.index.size} individual segments to {len(valid_segments)} continuous " + 
        "valid segments.")
    return valid_segments

def validate_label(file_path) -> List[str]:
    errors = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv_reader(csvfile)
        last_stop = -1
        for idx, row in enumerate(reader):
            #print(f'{idx}: {row}')
            if len(row) != 4:
                error = f'Line {idx+1}: {row} - Expected four values'
                errors.append(error)
                print(error)
                continue
            if not row[0].isdigit() or not row[1].isdigit() or not row[2].isdigit()\
                    or not row[3].isdigit():
                error = f'Line {idx+1}: {row} - not a number'
                errors.append(error) 
                print(error)
                continue
            if int(row[2]) == 0:
                error = f'Line {idx+1}: {row} - unnecessary INVALID label found'
                errors.append(error) 
                print(error)
                continue
            if int(row[2]) < 0 or len(__labels['values']) <= int(row[2]):
                error = f'Line {idx+1}: {row} - third value not a valid label, according to the " +\
                    f"yaml file.'
                errors.append(error) 
                print(error)
                continue
            if int(row[0]) >= int(row[1]):
                error = f'Line {idx+1}: {row} - stop must be higher than start'
                errors.append(error) 
                print(error)
                continue
            if last_stop > int(row[0]):
                error = f'Line {idx+1}: {row} - start must be higher or equal to previous stop'
                errors.append(error) 
                print(error)
                continue
            if int(row[3]) not in [0, 1]:
                error = f'Line {idx+1}: {row} - fourth value represent a bool, must be 0 or 1.'
                errors.append(error) 
                print(error)
                continue
            last_stop = int(row[1])
    
    for error in errors:
        print(error)
    return errors

def validate_all(labels_path) -> List[str]:
    errors = []
    files = listdir(labels_path)
    for file in files:
        if file.endswith('.csv'):
            file_path = join(labels_path, file)
            print(f'Validating: {file}')
            file_errors = validate_label(file_path)
            errors = [*errors, *file_errors]
    print("Done validating")
    return errors