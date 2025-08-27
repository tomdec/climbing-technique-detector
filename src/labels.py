from enum import Enum
from csv import writer as csv_writer, reader as csv_reader
from os import listdir, makedirs
from os.path import join, exists
from typing import Iterator, List
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

def get_dataset_name() -> str:
    return __labels['name']

def name_to_value(name: str) -> int:
    return __labels['values'][name]

def value_to_name(value: int) -> str:
    for key, val in __labels['values'].items():
        if (val == value):
            return key
    raise ValueError(f"Value '{value}' not found in labels.")

def iterate_valid_labels() -> Iterator[str]:
    return iter([key for (key, value) in __labels['values'].items() if value > 0])

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
                return value_to_name(int(row[2]))
            else:
                return value_to_name(0)

def get_labels_as_dataframe(label_path) -> DataFrame:
    return read_csv(label_path, header=None, names=["start", "stop", "label"])

def validate_label(file_path) -> List[str]:
    errors = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv_reader(csvfile)
        last_stop = -1
        for idx, row in enumerate(reader):
            #print(f'{idx}: {row}')
            if len(row) != 3:
                error = f'Line {idx+1}: {row} - Expected three values'
                errors.append(error) 
                print(error)
                continue
            if not row[0].isdigit() or not row[1].isdigit() or not row[2].isdigit():
                error = f'Line {idx+1}: {row} - not a number'
                errors.append(error) 
                print(error)
                continue
            if int(row[2]) == 0:
                error = f'Line {idx+1}: {row} - unnecessary INVALID label found'
                errors.append(error) 
                print(error)
                continue
            if int(row[2]) not in __labels['values'].values():
                error = f'Line {idx+1}: {row} - third value not a valid label, according to the yaml file.'
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
            last_stop = int(row[1])
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

def __correct_fps(label_path, output_path):
    '''
    One-time-use code. Can be ignored unless needed
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

def __correct_all_labelling(labels_path):
    '''
    One-time-use code. Can be ignored unless needed
    '''
    files = [file for file in listdir(labels_path) if file.endswith('.csv')]
    for file_name in files:
        file_path = join(labels_path, file_name)
        __correct_labelling(file_path)

def __correct_labelling(file_path):
    '''
    One-time-use code. Can be ignored unless needed
    Still needs manual removal of trailing newline. Did not find how to tell pandas not to store that.
    '''
    label_df = get_labels_as_dataframe(file_path)
    label_df['label'] = label_df['label'].replace({7: 6, 8: 7})
    label_df.to_csv(file_path, header=False, index=False)