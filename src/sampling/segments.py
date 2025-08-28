from os.path import exists, join
from os import listdir
from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_POS_FRAMES, VideoWriter_fourcc, VideoWriter

from src.labels import get_labels_as_dataframe, make_label_dirs, value_to_name
from src.common.helpers import get_filename

def build_sample_dirs(rootpath):
    samples_dir = join(rootpath, "samples")
    make_label_dirs(samples_dir)

def generate_from_labels(video_path, 
        path_to_samples,
        run_build_sample_dirs = True):
    original_video = VideoCapture(video_path)
    try: 
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

        if run_build_sample_dirs:
            build_sample_dirs(path_to_samples)
        
        for _, row in labels.iterrows():
            start = row["start"]
            label_name = value_to_name(row["label"])
            stop = row["stop"]
        
            sample_path = f"{path_to_samples}/samples/{label_name}/{file_name}__{start}.mp4"
            if (exists(sample_path)):
                print(f"Sample '{sample_path}' already exists, skipping.")
                continue
        
            sample = VideoWriter(sample_path, VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)
            try:
                original_video.set(CAP_PROP_POS_FRAMES, start)
                for frame_num in range(start, stop):
                    if (not original_video.isOpened()):
                        print(f"{video_path} was closed before all samples could be generated")
                        break
                    success, image = original_video.read()
                    if not success or image is None:
                        print(f'Could not read frame nr {frame_num}')
                        print(f'Sample {sample_path} is incomplete.')
                        break
                    
                    sample.write(image)
                print(f"Created segment '{sample_path}'")
            finally:
                sample.release()
    finally:
        original_video.release()

def generate_all_segments(data_root):
    video_root = join(data_root, "videos")

    build_sample_dirs(data_root)

    videos = listdir(video_root)
    for video in videos:
        video_path = join(video_root, video)
        generate_from_labels(video_path, data_root, False)

