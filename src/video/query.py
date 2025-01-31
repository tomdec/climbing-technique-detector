from cv2 import VideoCapture

def get_frame(video_path: str, frame_nr: int):
    vid_capture = VideoCapture(video_path)

    if (not vid_capture.isOpened()):
        print("video not opened")
        return None

    for x in range(frame_nr):
        vid_capture.read()

    success, frame = vid_capture.read()
    if not success:
        print(f"Failed to read frame {frame_nr} from video {vid_capture}")
        return None
    
    vid_capture.release()
    return frame