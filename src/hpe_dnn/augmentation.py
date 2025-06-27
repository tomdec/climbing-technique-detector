from pandas import DataFrame, Series
from cv2 import imread
from itertools import zip_longest
from numpy import array
from albumentations import Compose, ShiftScaleRotate, HorizontalFlip, Erasing, Perspective, \
    RandomBrightnessContrast, KeypointParams

__transform_pipeline = Compose([
    #A.RandomCrop(width=300, height=300),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=0),
    HorizontalFlip(p=0.5),
    #A.Mosaic(grid_yx=(2, 2)),
    Erasing(p=0.4),
    Perspective(p=0.2),
    RandomBrightnessContrast(p=0.4)
], keypoint_params=KeypointParams(format('xyz'), remove_invisible=False))

def __is_coordinate(header: str) -> bool:
    return header.endswith('_x') or header.endswith('_y') or header.endswith('_z')

def __to_augmenting_array(input: Series, height, width):
    xyz = [input[header] for header in input.index if __is_coordinate(header)]
    xyz = array(xyz).reshape(-1, 3)
    xyz = xyz * array([width, height, width])
    
    vis = [input[header] for header in input.index if header.endswith('visibility')]
    
    return xyz, vis

def __to_df_row(input: Series, xyz, vis, height, width):
    scale = array([width, height, width])
    xyz_relative = xyz / scale
    zipped = list(zip_longest(xyz_relative, vis))
    appended = [[*coordinates, visibility] for coordinates, visibility in zipped]
    result_array = [element for element in array(appended).reshape(-1) if element != None]
    
    result_array.append(input["technique"])
    result_array.append(input["image_path"])

    return Series(data=result_array, index=input.index)

def augment_keypoints(series):
    img_path = series["image_path"]
    image = imread(img_path)
    height, width, _ = image.shape
    
    xyz, vis = __to_augmenting_array(series, height, width)
    transformed = __transform_pipeline(image=image, keypoints=xyz)

    return __to_df_row(series, transformed['keypoints'], vis, height, width)