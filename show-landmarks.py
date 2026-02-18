import cv2
from argparse import ArgumentParser
from os.path import exists


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="show-landmarks",
        description="Play video annotated with HPE landmarks from either MediaPipe or YOLO",
    )
    parser.add_argument("path", help="Path to video to play.")
    parser.add_argument(
        "--tool",
        choices=["mp", "yolo"],
        help="HPE tool to extract landmarks with. Default = mp",
        default="mp",
    )
    args = parser.parse_args()

    video_name = args.path
    if not exists(video_name):
        raise Exception(f"Video file {video_name} not found, cannot play.")
    tool = args.tool

    from src.hpe.mp.play import play_with_hpe as play_with_mp_hpe
    from src.hpe.yolo.play import play_with_hpe as play_with_yolo_hpe

    try:
        if tool == "mp":
            play_with_mp_hpe(video_name)
        else:
            play_with_yolo_hpe(video_name)
    finally:
        cv2.destroyAllWindows()
