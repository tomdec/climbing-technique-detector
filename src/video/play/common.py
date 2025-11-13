from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, WINDOW_NORMAL,\
    imshow, waitKey, destroyAllWindows, namedWindow, moveWindow, resizeWindow
from cv2.typing import MatLike
from screeninfo import get_monitors, Monitor
from typing import List, Callable, Any, Tuple

from src.common.helpers import get_center_coordinates

ImgShape = Tuple[int, int]

def place_on_monitor(window_name: str, m: Monitor, img_size: ImgShape):
    center_x, center_y = (m.x + int(m.width / 2), m.y + int(m.height / 2))
    image_width, image_height = img_size
    if image_width > m.width:
        image_width = m.width
        image_height = int(image_height * m.width / image_width)
    if image_height > m.height:
        image_width = int(image_width * m.height / image_height)
        image_height = m.height
    
    namedWindow(window_name, WINDOW_NORMAL)
    moveWindow(window_name, center_x - int(image_width / 2), center_y - int(image_height / 2))
    resizeWindow(window_name, int(image_width * 0.9), int(image_height * 0.9))

def play_video(video_path: str,
        context: Any = None,
        mutators: List[Callable[[MatLike, Any], MatLike]] = []):
    vid_capture = VideoCapture(video_path)
    try:
        frame_num = 0
        if not vid_capture.isOpened():
            print(f"Cannot open video file '{video_path}'")
            exit()
        
        fps = vid_capture.get(CAP_PROP_FPS)
        vid_width = vid_capture.get(CAP_PROP_FRAME_WIDTH)
        vid_height = vid_capture.get(CAP_PROP_FRAME_HEIGHT)

        m = get_monitors()[0]
        
        while vid_capture.isOpened():
            if context is None:
                context = frame_num
            ret, frame = vid_capture.read()
            if ret == False:
                print(f"Could not read frame nr {frame_num}")
                break
                
            for mutator in mutators:
                frame = mutator(frame, context)

            place_on_monitor("segment", m, (vid_width, vid_height))
            imshow("segment", frame)
            if waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
            
            frame_num += 1
    finally:
        vid_capture.release()
        destroyAllWindows()