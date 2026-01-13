from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES, imshow,\
    waitKey, destroyAllWindows
from cv2.typing import MatLike

from src.labels import get_label_by_frame_num, get_labels_from_video, get_labels_as_dataframe
from src.common.draw import write_label, write_text
from src.video.play.common import play_video

def play_with_label(video_path: str):
    label_path = get_labels_from_video(video_path)
    labels = get_labels_as_dataframe(label_path)

    def mutator(img: MatLike, frame_num: int):
        label = get_label_by_frame_num(labels, frame_num)
        img = write_label(img, label)
        return img

    play_video(video_path, mutators=[mutator])

def play_with_frame_num(video_path: str, start_frame: int = 0, stop_frame: int | None = None):
    
    def mutator(img: MatLike, frame_num: int):
        img = write_text(img, f"{frame_num}")
        return img

    play_video(video_path, mutators=[mutator], start_frame=start_frame, stop_frame=stop_frame)

# Might be platform dependent, check with: 
# https://stackoverflow.com/questions/14494101/using-other-keys-for-the-waitkey-function-of-opencv
__Q = ord('q')
__LeftKey = ord('Q')
__RightKey = ord('S')
__Space = ord(' ')

def play_with_frame_num_detailed(video_path: str):
    vid_capture = VideoCapture(video_path)
    if not vid_capture.isOpened():
        print(f"Cannot open video file '{video_path}'")
        exit()
    
    fps = int(vid_capture.get(CAP_PROP_FPS))
    frame_count = vid_capture.get(CAP_PROP_FRAME_COUNT)
    frame_num = 0
    current = frame_num
    paused = False

    print(f'starting playback. FPS: {fps}, Count: {frame_count}')
    while vid_capture.isOpened():
        if current != frame_num:
            ret, frame = vid_capture.read()
            current = frame_num 
            if ret == False:
                print(f"Could not read frame nr {frame_num}")
                break
            frame = write_text(frame, f"{frame_num} / {frame_count}")
            imshow(video_path, frame)
        
        key = waitKey(int(1000/fps)) & 0xFF
        if key == __Q:
            print('exiting playback')
            break
        if key == __LeftKey:
            if paused:
                frame_num = max(frame_num - 1, 0)
            else:
                frame_num = max(frame_num - fps, 0)
            vid_capture.set(CAP_PROP_POS_FRAMES, frame_num)
            print(f"Going back to frame {frame_num}")
            continue
        if key == __RightKey:
            if paused:
                frame_num = min(frame_num + 1, frame_count)
            else:
                frame_num = min(frame_num + fps, frame_count)
            vid_capture.set(CAP_PROP_POS_FRAMES, frame_num)
            print(f"Jumping to frame {frame_num}")
            continue
        if key == __Space:
            paused = not paused

        if not paused:
            frame_num += 1
    
    print('playback finished')
    vid_capture.release()
    destroyAllWindows()
