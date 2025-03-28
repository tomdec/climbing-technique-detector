import cv2
from src.video.play.with_text import play_with_label

if __name__ == '__main__':
    try:
        play_with_label("./data/videos/How to Flag - A Climbing Technique for Achieving Balance.mp4")
    finally:
        cv2.destroyAllWindows()