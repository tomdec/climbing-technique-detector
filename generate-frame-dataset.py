from argparse import ArgumentParser

__data_root = "data"

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="generate-frame-dataset",
        description="Generate dataset of individual frames from the videos.",
    )
    args = parser.parse_args()

    from src.sampling.images import generate_frame_dataset

    generate_frame_dataset(__data_root)
