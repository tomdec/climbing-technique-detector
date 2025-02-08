import cv2
from src.video.play import play_with_hpe

if __name__ == '__main__':
    try:
        play_with_hpe("./data/samples/NONE/How to Flag - A Climbing Technique for Achieving Balance__500.mp4")
    finally:
        cv2.destroyAllWindows()