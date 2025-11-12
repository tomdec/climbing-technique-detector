from typing import List, Callable, Any
from cv2 import VideoCapture, CAP_PROP_FPS, imshow, waitKey, destroyAllWindows,\
    namedWindow, moveWindow
from cv2.typing import MatLike

def play_video(video_path: str,
        context: Any = None,
        mutators: List[Callable[[MatLike, Any], MatLike]] = []):
    vid_capture = VideoCapture(video_path)
    frame_num = 0
    if not vid_capture.isOpened():
        print(f"Cannot open video file '{video_path}'")
        exit()
    
    fps = vid_capture.get(CAP_PROP_FPS)

    namedWindow(video_path)
    moveWindow(video_path, 100, 100)

    while vid_capture.isOpened():
        if context is None:
            context = frame_num
        ret, frame = vid_capture.read()
        if ret == False:
            print(f"Could not read frame nr {frame_num}")
            break
            
        for mutator in mutators:
            frame = mutator(frame, context)

        imshow(video_path, frame)
        if waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
        
        frame_num += 1

    vid_capture.release()
    destroyAllWindows()