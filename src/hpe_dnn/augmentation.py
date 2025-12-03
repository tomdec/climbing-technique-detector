from pandas import DataFrame, Series
from typing import Callable, Tuple, List, Any
from cv2.typing import MatLike
from numpy import ndarray, array
from itertools import zip_longest
from albumentations import Compose, ShiftScaleRotate, HorizontalFlip, Erasing, Perspective, \
    RandomBrightnessContrast, KeypointParams
from math import isnan, nan

from src.common.helpers import imread

CoordinateFeatures = ndarray[Series]
VisibilityFeatures = List[Series]
AugmentationFunc = Callable[[MatLike, CoordinateFeatures, VisibilityFeatures], 
                            Tuple[CoordinateFeatures, VisibilityFeatures]]

def _all_are(array, value):
    return all([x == value for x in array])

def _is_coordinate(header: str) -> bool:
    return header.endswith('_x') or header.endswith('_y') or header.endswith('_z')

def _2d_scaler(height: int, width: int) -> ndarray:
    return array([width, height])

def _3d_scaler(height: int, width: int) -> ndarray:
    return array([width, height, width])

class AugmentationPipeline:

    @staticmethod
    def for_dataframe(df: DataFrame):
        def is_z_coordinate(column_name: str) -> bool:
            return column_name.endswith("_z")
        
        if any(list(map(is_z_coordinate, df.columns))):
            return AugmentationPipeline(3)
        else:
            return AugmentationPipeline(2)

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def keypoint_format(self) -> str:
        return self._keypoint_format
    
    @property
    def keypoint_scaler(self) -> Callable[[int, int], ndarray]:
        return self._keypoint_scaler

    def __init__(self, dimensionality: int,
            augmentation_override: AugmentationFunc | None = None):
        self._dimensionality = dimensionality
        if self._dimensionality == 2:
            self._keypoint_format = "xy"
            self._keypoint_scaler = _2d_scaler

        elif self._dimensionality == 3:
            self._keypoint_format = "xyz"
            self._keypoint_scaler = _3d_scaler

        else:
            raise ValueError(f"Received unexpected value '{self._dimensionality}' for dimensionality.")

        self._augmentation_override = augmentation_override

    def __call__(self, series: Series) -> Series:
        return self.augment_keypoints(series)

    def augment_keypoints(self, series: Series) -> Series:
        img_path = series["image_path"]
        image = imread(img_path)
        height, width, _ = image.shape
        
        xyz, vis = self.__to_augmenting_array(series, height, width)
        
        if self._augmentation_override is None:
            xyz, vis = self.__default_augmentation(image, xyz, vis)
        else:
            xyz, vis = self._augmentation_override(image, xyz, vis)

        return self.__to_df_row(series, xyz, vis, height, width)
    
    def __to_augmenting_array(self, input: Series, 
            height: int, width: int) -> Tuple[CoordinateFeatures, VisibilityFeatures]:
        xyz = [input[header] for header in input.index if _is_coordinate(header)]
        xyz = array(xyz).reshape(-1, self.dimensionality)
        xyz = xyz * self.keypoint_scaler(height, width)
        
        vis = [input[header] for header in input.index if header.endswith('visibility')]
        
        return xyz, vis

    def __default_augmentation(self, image: MatLike, 
            coordinates: CoordinateFeatures, 
            visibility: VisibilityFeatures) -> Tuple[CoordinateFeatures, VisibilityFeatures]:

        transform_pipeline = Compose([
            #A.RandomCrop(width=300, height=300),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=0),
            HorizontalFlip(p=0.5),
            #A.Mosaic(grid_yx=(2, 2)),
            Erasing(p=0.4),
            Perspective(p=0.2),
            RandomBrightnessContrast(p=0.4)
        ], keypoint_params=KeypointParams(format(self.keypoint_format), remove_invisible=False))
        
        transformed = transform_pipeline(image=image, keypoints=coordinates)
        coordinates, visibility = self.__mark_removed_keypoints(transformed['image'], transformed['keypoints'], visibility)

        return coordinates, visibility

    def __mark_removed_keypoints(self, image: MatLike, 
            coordinates: CoordinateFeatures, 
            visibility: VisibilityFeatures) -> Tuple[CoordinateFeatures, VisibilityFeatures]:
        new_coordinates = []
        new_visibility = [ *visibility ]
        
        for idx, keypoint in enumerate(coordinates):
            color = self.__get_color_at_keypoint(image, keypoint)

            if _all_are(color, -2) or _all_are(color, 0): # out of bounds, or obscured
                new_coordinates.append([nan] * self.dimensionality)
                if idx < len(new_visibility):
                    new_visibility[idx] = 0
            else:
                new_coordinates.append(keypoint)

        return new_coordinates, new_visibility
    
    def __get_color_at_keypoint(self, image: MatLike, keypoint):
        x = keypoint[0]
        y = keypoint[1]
        
        if (isnan(x) or isnan(y)):
            return [-1, -1, -1] 
        
        if (x < 0 or y < 0 or image.shape[1] <= x or image.shape[0] <= y):
            return [-2, -2, -2]

        return image[int(y), int(x)]
    
    def __to_df_row(self, input: Series, 
            xyz: CoordinateFeatures, 
            vis: VisibilityFeatures, 
            height: int, 
            width: int) -> Series:
        
        xyz_relative = xyz / self.keypoint_scaler(height, width)
        
        zipped = list(zip_longest(xyz_relative, vis))
        appended = [[*coordinates, visibility] for coordinates, visibility in zipped]
        result_array = [element for element in array(appended).reshape(-1) if element != None]
        
        result_array = [input["label"], *result_array, input["image_path"]]

        return Series(data=result_array, index=input.index)
