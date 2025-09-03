from cv2 import VideoCapture, CAP_PROP_FPS, imshow, waitKey, destroyAllWindows, cvtColor, COLOR_RGB2BGR, COLOR_BGR2RGB

from src.hpe.mp.model import build_holistic_model
from src.hpe.mp.evaluate import predict_landmarks
from src.hpe.mp.draw import draw_my_landmarks

def play_with_hpe(video_path: str):
    vid_capture = VideoCapture(video_path)
    frame_num = 0
    if not vid_capture.isOpened():
        print(f"Cannot open video file '{video_path}'")
        exit()
    
    fps = vid_capture.get(CAP_PROP_FPS)

    with build_holistic_model() as holistic:
        while vid_capture.isOpened():
            ret, frame = vid_capture.read()
            if ret == False:
                print(f"Could not read frame nr {frame_num}")
                break
                
            frame = cvtColor(frame, COLOR_RGB2BGR)         # COLOR CONVERSION BGR 2 RGB
            
            frame.flags.writeable = False                # Image is no longer writeable
            results, _ = predict_landmarks(frame, holistic)

            frame = cvtColor(frame, COLOR_BGR2RGB)         # COLOR CONVERSION BGR 2 RGB
            
            frame = draw_my_landmarks(frame, results)


            imshow(video_path, frame)
            if waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
            
            frame_num += 1

    vid_capture.release()
    destroyAllWindows()