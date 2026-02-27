from argparse import ArgumentParser
from glob import glob
from os.path import join

__data_root = "data"

if __name__ == "__main__":

    parser = ArgumentParser(
        prog="extract-video-landmarks",
        description="Extracts landmarks from all videos entirely.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect the landmarks annotated on the video as they are extracted.",
    )
    args = parser.parse_args()
    inspect = args.inspect
    from src.sampling.landmarks import extract_hpe_dataset

    video_paths = glob(join(__data_root, "videos", "**", "*.*"), recursive=True)
    for video_path in video_paths:
        if not inspect:
            print(f"Extracting from: {video_path}")
        extract_hpe_dataset(video_path=video_path, inspect=inspect)
