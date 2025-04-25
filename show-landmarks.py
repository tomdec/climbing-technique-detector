import cv2
from video.play.with_hpe import play_with_hpe

if __name__ == '__main__':
    try:
        play_with_hpe("./data/samples/NONE/How to Flag - A Climbing Technique for Achieving Balance__500.mp4")
    finally:
        cv2.destroyAllWindows()