from argparse import ArgumentParser
from glob import glob
from os.path import join

__data_root = "data"

if __name__ == "__main__":

    parser = ArgumentParser(prog="extract-video-landmarks",
        description="Extracts landmarks from videos")
    args = parser.parse_args()

    from src.sampling.landmarks import extract_hpe_dataset

    video_paths = glob(join(__data_root, "videos", "**", "*.*"), recursive=True)
    for video_path in video_paths:
        extract_hpe_dataset(video_path=video_path)
