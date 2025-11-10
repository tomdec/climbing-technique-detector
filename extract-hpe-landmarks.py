from argparse import ArgumentParser
from glob import glob
from os.path import join

__data_root = "data"

if __name__ == "__main__":

    parser = ArgumentParser(prog="extract-hpe-landmarks",
        description="Extracts landmarks from various sources")
    parser.add_argument("source", 
        choices=["videos", "segments", "images"],
        help="Where to extract the landmarks from.")
    args = parser.parse_args()

    if args.source == "videos":
        from src.sampling.landmarks import extract_hpe_dataset

        video_paths = glob(join(__data_root, "videos", "**", "*.*"), recursive=True)
        for video_path in video_paths:
            extract_hpe_dataset(video_path=video_path)

    else:
        print(f"{args.source} is not yet implemented")