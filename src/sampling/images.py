from os.path import exists, join
from os import makedirs, listdir
from random import randint, random
from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, imwrite
from matplotlib.pyplot import subplots, subplots_adjust, Axes
from numpy import mean

from labels import Technique
from common import get_filename, get_split_limits

def build_image_dirs(root):
    name = "techniques"
    dataset_dir = join(root, name)
    if not exists(dataset_dir):
        makedirs(dataset_dir)
    
    for segment in ["train", "test", "val"]:
        segment_dir = join(dataset_dir, segment)
        if not exists(segment_dir):
            makedirs(segment_dir)
        
        for value in Technique:
            if value == Technique.INVALID:
                continue
            label_dir = join(segment_dir, value.name)
            if not exists(label_dir):
                makedirs(label_dir)

def random_init_skip(max):
    max_int = int(max)
    return randint(0, max_int-1)

def data_slice_factory(data_split):
    train_limit, val_limit = get_split_limits(data_split)

    def factory():
        rand = random()
        if rand < train_limit: 
            return "train"
        elif rand < val_limit:
            return "val"
        else:
            return "test"
        
    return factory

def generate_image_dataset(video_path,
        dataset_root,
        label_name,
        data_split = (0.8, 0.0, 0.2)):
    '''
    data_split = (train, val, test)
    '''
    slice_factory = data_slice_factory(data_split)

    sample_video = VideoCapture(video_path)
    if not sample_video.isOpened():
        print(f"Cannot open video file '{video_path}'")
        exit()
    
    video_name = get_filename(video_path)
    total_frames = sample_video.get(CAP_PROP_FRAME_COUNT)
    frame_skip = 10
    frame_num = random_init_skip(min(frame_skip, total_frames))
    current_frame = 0

    while sample_video.isOpened() and (frame_num < total_frames):
        while current_frame < frame_num:
            success = sample_video.grab()
            if not success:
                print(f'Could not read frame nr {current_frame} of {total_frames}')
                break
            current_frame += 1

        success, image = sample_video.read()
        if not success or image is None:
            print(f'Could not read frame nr {current_frame} of {total_frames}')
            break

        slice = slice_factory()
        label_dir = join(dataset_root, "techniques", slice, label_name)
        
        file_name = f'{video_name}__{frame_num}.png'
        imwrite(join(label_dir, file_name), image)
        
        frame_num += frame_skip
        if frame_num > total_frames:
            print(f'Reached end of video')
            break

    sample_video.release()

def generate_image_dataset_from_samples(data_root,
        data_split = (0.8, 0.0, 0.2)):
    
    samples_root = join(data_root, "samples")
    img_root = join(data_root, "img")

    build_image_dirs(img_root)

    for label in listdir(samples_root):
        label_path = join(samples_root, label)
        for video in listdir(label_path):
            video_path = join(label_path, video)
            generate_image_dataset(video_path, img_root, label, data_split)

def plot_frame_count_distributions(samples_root_dir: str):
    frames_dict = {}
    for label in listdir(samples_root_dir):
        label_root = join(samples_root_dir, label)
        frames = []
        
        for sample in listdir(label_root):
            sample_path = join(label_root, sample)
            cap = VideoCapture(sample_path)
            frame_num = cap.get(CAP_PROP_FRAME_COUNT)
            fps = cap.get(CAP_PROP_FPS)
            frames.append(frame_num)
            #durations.append(round((frame_num / fps) * 1000)) #ms
            cap.release()

        frames_dict[label] = frames

    fig, axes = subplots(4, 2, figsize=(12,12))
    fig.suptitle("Frame counts from samples of each label")
    fig.text(0.55, 0.04, "Frame count", ha="center")
    fig.text(0.04, 0.5, "Sample count", va="center", rotation="vertical")
    fig.tight_layout()
    subplots_adjust(hspace=0.3, left=0.1, bottom=0.1, top=0.92)

    axes = axes.ravel()
    __plot_hist(axes[0], sum(frames_dict.values(), []), "TOTAL")
    for idx, label in enumerate(listdir(samples_root_dir)):
        __plot_hist(axes[idx+1], frames_dict[label], label)

def __plot_hist(axes: Axes, data, title:str):
    axes.hist(data, bins=20, range=(0, 600))
    axes.set_title(f"{title} (Count: {len(data)})")
    average = mean(data)
    axes.axvline(average, color="k", linestyle="dashed", linewidth=1)
    _, max_ylim = axes.get_ylim()
    axes.text(average*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(average))