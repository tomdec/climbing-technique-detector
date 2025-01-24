import csv
from common import Technique
from os import listdir
from os.path import join

labels_dir = "./data/labels"

def validate_label(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            #print(f'{idx}: {row}')
            if len(row) != 2:
                print(f'Line {idx}: {row} - Expected a pair of numbers')
                continue
            if not row[0].isdigit() or not row[1].isdigit():
                print(f'Line {idx}: {row} - not a number')
                continue
            if int(row[1]) not in Technique:
                print(f'Line {idx}: {row} - second value not a valid technique')
                continue

def validate_all(root_dir):
    files = listdir(root_dir)
    for file in files:
        file_path = join(root_dir, file)
        print(f'Validating: {file}')
        validate_label(file_path)

if __name__ == '__main__':
    validate_all(labels_dir)