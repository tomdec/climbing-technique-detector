from argparse import ArgumentParser
from os.path import exists

if __name__ == "__main__":

    parser = ArgumentParser(
        prog="play-video",
        description=f"""
            Play the given video with detailed controls and annotated with the current frame number.
            Controls during playback: 
            Spacebar: pause/play
            left arrow key: rewind video. 1s while playing, 1 frame while paused.
            right arrow key: forward video. 1s while playing, 1 frame while paused. 
            q key: quit video playback.
        """,
    )
    parser.add_argument("path", help="Path to video to play.")
    args = parser.parse_args()
    video_name = args.path
    if not exists(video_name):
        raise Exception(f"Video file {video_name} not found, cannot play.")

    import cv2

    from src.video.play.with_text import play_with_frame_num_detailed

    try:
        play_with_frame_num_detailed(video_name)
    finally:
        cv2.destroyAllWindows()
