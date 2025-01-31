import cv2
import numpy as np
from common import Technique, get_label

def show_labels(video_path: str):
    label_path = video_path.replace("/videos/", "/labels/").replace(".mp4", ".csv")
    vid_capture = cv2.VideoCapture(video_path)
    frame_num = 0
    if not vid_capture.isOpened():
        print(f"Cannot open video file '{video_path}'")
        exit()
    
    while vid_capture.isOpened():
        #fig = plt.figure()
        ret, frame = vid_capture.read()
        label = get_label(label_path, frame_num)
        if ret == True:
            #plt.imshow(frame)
            #plt.show()
            cv2.imshow('name', frame)
            if cv2.waitKey(20) == ord('q'):
                break
        else:
            break
        frame_num += 1

    vid_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        show_labels("./data/videos/How to Flag - A Climbing Technique for Achieving Balance.mp4")
    finally:
        cv2.destroyAllWindows()