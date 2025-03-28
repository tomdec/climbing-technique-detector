import cv2
from os import listdir
from sys import argv
from os.path import join

from src.video.play.with_text import play_with_frame_num

if __name__ == '__main__':
    videos_path = "./data/videos"
    videos = listdir(videos_path)
    video_idx = int(argv[1])

    try:
        play_with_frame_num(join(videos_path, videos[video_idx]))
    finally:
        cv2.destroyAllWindows()