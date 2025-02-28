from os.path import exists, join, split
from os import makedirs, listdir
from random import randint, random
from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT, imwrite

from src.labels import Technique
from src.common import get_filename

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

def get_label_from_path(path) -> Technique:
    head, tail = split(path)
    if head == '':
        raise Exception("Could not find Technique")
    
    if tail in [label.name for label in Technique]:
        return Technique[tail]
    
    return get_label_from_path(head)

def random_init_skip(max):
    return randint(0, max-1)

def data_slice_factory(data_split):
    train_limit = data_split[0]
    val_limit = data_split[0] + data_split[1]

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
        data_split = (0.8, 0.0, 0.2)):
    '''
    data_split = (train, val, test)
    '''
    label = get_label_from_path(video_path)
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
        label_dir = join(dataset_root, "techniques", slice, label.name)
        
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
            generate_image_dataset(video_path, img_root, data_split)
