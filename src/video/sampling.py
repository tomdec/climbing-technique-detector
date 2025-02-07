from os.path import exists, join
from os import makedirs, listdir
from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, VideoWriter_fourcc, VideoWriter

from src.labels import get_labels_as_dataframe, Technique
from src.common import get_filename

def build_sample_dirs(rootpath):
    samples_dir = join(rootpath, "samples")
    if not exists(samples_dir):
        makedirs(samples_dir)

    for value in Technique:
        if value == Technique.INVALID:
            continue
        samples_label_path = join(samples_dir, value.name)
        if not exists(samples_label_path):
            makedirs(samples_label_path)

def generate_samples(video_path, 
                     path_to_samples,
                     run_build_sample_dirs = True):
    original_video = VideoCapture(video_path)
    if not original_video.isOpened():
        print(f"Cannot open video file '{video_path}'")
        exit()

    file_name = get_filename(video_path)
    fps = original_video.get(CAP_PROP_FPS)
    frame_width = original_video.get(CAP_PROP_FRAME_WIDTH)
    frame_height = original_video.get(CAP_PROP_FRAME_HEIGHT)
    frame_size = int(frame_width), int(frame_height)

    label_path = video_path.replace("/videos/", "/labels/").replace(".mp4", ".csv")
    labels = get_labels_as_dataframe(label_path)

    frame_num = 0
    row_num = 0
    write_frames = False

    if run_build_sample_dirs:
        build_sample_dirs(path_to_samples)

    while (original_video.isOpened()):
        success, image = original_video.read()
        if not success or image is None:
            print(f'Could not read frame nr {frame_num}')
            break
            
        if frame_num == labels.loc[row_num, "start"]:
            label_name = Technique(labels.loc[row_num, "label"]).name
            sample_path = f"{path_to_samples}/samples/{label_name}/{file_name}__{frame_num}.mp4"
            print(sample_path)
            sample = VideoWriter(sample_path, VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)
            write_frames = True
        
        if write_frames:
            sample.write(image)
            print(frame_num)
        
        if frame_num == labels.loc[row_num, "stop"]-1:
            sample.release()
            if row_num+1 == labels.shape[0]:
                break
            row_num += 1
            write_frames = False
            
        frame_num += 1
    
    original_video.release()
    sample.release()

def generate_all_samples(data_root):
    video_root = join(data_root, "videos")
    samples_root = join(data_root, "samples")

    build_sample_dirs(samples_root)

    videos = listdir(video_root)
    for video in videos:
        video_path = join(video_root, video)
        generate_samples(video_path, samples_root, False)

