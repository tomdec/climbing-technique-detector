from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES, imshow, waitKey, destroyAllWindows

from src.labels import get_label
from src.video.edit import write_label, write_text

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
        frame = write_label(frame, label)

        imshow(video_path, frame)
        if waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
        
        frame_num += 1

    vid_capture.release()
    destroyAllWindows()

# Might be platform dependent, check with: https://stackoverflow.com/questions/14494101/using-other-keys-for-the-waitkey-function-of-opencv
__Q = ord('q')
__LeftKey = ord('Q')
__RightKey = ord('S')
__Space = ord(' ')

def play_with_frame_num(video_path: str):
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
