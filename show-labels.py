import cv2
from argparse import ArgumentParser
from os.path import exists

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="show-labels",
        description="Play video annotated with current label",
    )
    parser.add_argument("path", help="Path to video to play.")
    args = parser.parse_args()
    video_name = args.path
    if not exists(video_name):
        raise Exception(f"Video file {video_name} not found, cannot play.")

    from src.video.play.with_text import play_with_label

    try:
        play_with_label(video_name)
    finally:
        cv2.destroyAllWindows()
