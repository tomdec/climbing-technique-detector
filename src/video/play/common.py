from cv2 import INTER_LINEAR, VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, WINDOW_NORMAL,\
    imshow, waitKey, destroyAllWindows, namedWindow, moveWindow, resizeWindow, resize
from cv2.typing import MatLike
from screeninfo import get_monitors, Monitor
from typing import List, Callable, Any, Tuple

ImgShape = Tuple[int, int]

def scale_image_to_monitor(m_width, m_height, im_width, im_height) -> ImgShape:
    lower_boundary = 0.7

    #upscale
    if (im_width < m_width * lower_boundary) and (im_height < m_height * lower_boundary):
        if im_width > im_height:
            ratio = m_width  * lower_boundary / im_width
            im_width = int(m_width * lower_boundary)
            im_height = int(im_height * ratio)
        else:
            ratio = m_height  * lower_boundary / im_height
            im_width = int(im_width * ratio)
            im_height = int(m_height * lower_boundary)
    #downscale
    else:
        if im_width > m_width:
            ratio = m_width / im_width
            im_width = m_width
            im_height = int(im_height * ratio)
        if im_height > m_height:
            ratio = m_height / im_height
            im_width = int(im_width * ratio)
            im_height = m_height

    return im_width, im_height


def place_on_monitor(window_name: str, m: Monitor, img_size: ImgShape) -> ImgShape:
    center_x, center_y = (m.x + int(m.width / 2), m.y + int(m.height / 2))
    image_width, image_height = img_size
    
    image_width, image_height = scale_image_to_monitor(m.width, m.height, image_width, image_height)
    
    namedWindow(window_name, WINDOW_NORMAL)
    moveWindow(window_name, center_x - int(image_width / 2), center_y - int(image_height / 2))
    resizeWindow(window_name, int(image_width * 0.9), int(image_height * 0.9))
    return int(image_width * 0.9), int(image_height * 0.9)

def play_video(video_path: str,
        context: Any = None,
        mutators: List[Callable[[MatLike, Any], MatLike]] = [],
        playback_speed: float = 1):
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

            im_size = place_on_monitor("segment", m, (vid_width, vid_height))
            frame = resize(frame, im_size, interpolation=INTER_LINEAR)
            imshow("segment", frame)
            if waitKey(int(1000/(fps * playback_speed))) & 0xFF == ord('q'):
                break
            
            frame_num += 1
    finally:
        vid_capture.release()
        destroyAllWindows()