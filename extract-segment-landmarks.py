from argparse import ArgumentParser
from glob import glob
from os.path import join, exists
from pathlib import Path
from typing import MutableSet
from numpy import array, save, load

from src.labels import iterate_valid_labels

__data_root = "data"

if __name__ == "__main__":

    parser = ArgumentParser(prog="extract-hpe-landmarks",
        description="Interactively extracts landmarks for segments")
    parser.add_argument("label",
        choices=list(iterate_valid_labels()),
        help="Label to extract hpe segments for.")
    parser.add_argument("--inspect", 
        action="store_true",
        help="Inspect the extracted landmarks and accepted segments.")
    args = parser.parse_args()

    from src.sampling.landmarks import extract_segment_landmarks

    inspect_label = args.label
    segments = glob(join(__data_root, "samples", inspect_label, "*.*"), recursive=True)

    accepted_segments_path = join(__data_root, "df", "segments", inspect_label, "accepted.npy")
    if exists(accepted_segments_path):
        arr_loaded = load(accepted_segments_path)
        accepted_segments = set(arr_loaded)
    else:
        accepted_segments: MutableSet[str] = set()

    evaluated_segments = glob(join(__data_root, "df", "segments", inspect_label, "*.pkl"), recursive=True)
    evaluated_names = list(map(lambda seg: Path(seg).stem, evaluated_segments))

    for segment_path in segments:
        if Path(segment_path).stem in evaluated_names:
            print(f"Skipping '{segment_path}', already evaluated")
            continue
        
        print(f"Evaluating: {segment_path}")
        context, response = extract_segment_landmarks(segment_path)

        if response == 'q': break
        context.store_segment()
        if response == 'y': accepted_segments.add(segment_path)

    save(accepted_segments_path, array(list(accepted_segments)))