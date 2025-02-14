from cv2 import VideoCapture, CAP_PROP_FPS, imshow, waitKey, destroyAllWindows

from src.labels import get_label
from src.video.edit import write_label
from src.hpe.model import build_holistic_model
from src.hpe.evaluate import predict_landmarks
from src.hpe.draw import draw_my_landmarks

def play_with_label(video_path: str):
    label_path = video_path.replace("/videos/", "/labels/").replace(".mp4", ".csv")
    vid_capture = VideoCapture(video_path)
    frame_num = 0
    if not vid_capture.isOpened():
        print(f"Cannot open video file '{video_path}'")
        exit()
    
    fps = vid_capture.get(CAP_PROP_FPS)

    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if ret == False:
            print(f"Could not read frame nr {frame_num}")
            break

        label = get_label(label_path, frame_num)
        write_label(frame, label)

        imshow(video_path, frame)
        if waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
        
        frame_num += 1

    vid_capture.release()
    destroyAllWindows()

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

            _, results, _ = predict_landmarks(frame, holistic)
            frame = draw_my_landmarks(frame, results)

            imshow(video_path, frame)
            if waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
            
            frame_num += 1

    vid_capture.release()
    destroyAllWindows()