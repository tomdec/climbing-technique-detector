# rnn augmentation

from pandas import DataFrame, Series
from typing import Callable, Tuple, List
from cv2 import VideoCapture, CAP_PROP_POS_FRAMES, cvtColor, COLOR_BGR2RGB
from cv2.typing import MatLike
from numpy import ndarray, array
from itertools import zip_longest
from albumentations import (
    Compose,
    Affine,
    HorizontalFlip,
    Perspective,
    KeypointParams,
)
from math import isnan, nan
from matplotlib import pyplot as plt
from PIL import Image

from src.labels import iterate_valid_labels

CoordinateFeatures = ndarray[Series]
VisibilityFeatures = List[Series]
AugmentationFunc = Callable[
    [MatLike, CoordinateFeatures, VisibilityFeatures],
    Tuple[CoordinateFeatures, VisibilityFeatures],
]


def _all_are(array, value):
    return all([x == value for x in array])


def _is_coordinate(header: str) -> bool:
    return header.endswith("_x") or header.endswith("_y") or header.endswith("_z")


class AugmentationPipeline:

    __DIMENSIONALITY = 3
    __KEYPOINT_FORMAT = "xyz"

    @property
    def dimensionality(self) -> int:
        return self.__DIMENSIONALITY

    @property
    def keypoint_format(self) -> str:
        return self.__KEYPOINT_FORMAT

    def __init__(self, augmentation_override: AugmentationFunc | None = None):
        self._seed: int | None = None
        self._augmentation_override = augmentation_override

        self._current_video: str | None = None
        self._video_capture: VideoCapture | None = None

    def __call__(self, series: Series) -> Series:
        return self.augment_keypoints(series)

    def __del__(self):
        self.__release_video()

    def get_keypoint_scaler(self, height: int, width: int) -> ndarray:
        return array([width, height, width])

    def augment_keypoints(self, series: Series) -> Series:
        image = self.__get_image(series)
        height, width, _ = image.shape

        xyz, vis = self.__to_augmenting_array(series, height, width)

        if self._augmentation_override is None:
            xyz, vis = self.__default_augmentation(image, xyz, vis)
        else:
            xyz, vis = self._augmentation_override(image, xyz, vis)

        return self.__to_df_row(series, xyz, vis, height, width)

    def set_seed(self, seed):
        self._seed = seed

    def __get_image(self, row: Series) -> MatLike:
        video_path = row["video"]
        if self._current_video != video_path:
            self.__release_video()
            self._video_capture = VideoCapture(video_path)
            self._current_video = video_path

        self._video_capture.set(CAP_PROP_POS_FRAMES, row["frame_num"])
        _, frame = self._video_capture.read()
        frame = cvtColor(frame, COLOR_BGR2RGB)
        return frame

    def __release_video(self):
        if self._video_capture is not None:
            self._video_capture.release()

    def __to_augmenting_array(
        self, input: Series, height: int, width: int
    ) -> Tuple[CoordinateFeatures, VisibilityFeatures]:
        xyz = [input[header] for header in input.index if _is_coordinate(header)]
        xyz = array(xyz).reshape(-1, self.dimensionality)
        xyz = xyz * self.get_keypoint_scaler(height, width)

        vis = [
            input[header]
            for header in input.index
            if f"{header}".endswith("visibility")
        ]

        return xyz, vis

    def __default_augmentation(
        self,
        image: MatLike,
        coordinates: CoordinateFeatures,
        visibility: VisibilityFeatures,
    ) -> Tuple[CoordinateFeatures, VisibilityFeatures]:

        transform_pipeline = Compose(
            [
                Affine(
                    translate_percent=(-0.1, 0.1),
                    scale=(0.9, 1.1),
                    rotate=0,
                ),
                # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0),
                HorizontalFlip(p=0.5),
                Perspective(p=0.3),
            ],
            keypoint_params=KeypointParams(
                format(self.keypoint_format), remove_invisible=False
            ),
            seed=self._seed,
        )

        transformed = transform_pipeline(image=image, keypoints=coordinates)
        coordinates, visibility = self.__mark_removed_keypoints(
            transformed["image"], transformed["keypoints"], visibility
        )

        return coordinates, visibility

    def __mark_removed_keypoints(
        self,
        image: MatLike,
        coordinates: CoordinateFeatures,
        visibility: VisibilityFeatures,
    ) -> Tuple[CoordinateFeatures, VisibilityFeatures]:
        new_coordinates = []
        new_visibility = [*visibility]

        for idx, keypoint in enumerate(coordinates):
            color = self.__get_color_at_keypoint(image, keypoint)

            if _all_are(color, -2) or _all_are(color, 0):  # out of bounds, or obscured
                new_coordinates.append([nan] * self.dimensionality)
                if idx < len(new_visibility):
                    new_visibility[idx] = 0
            else:
                new_coordinates.append(keypoint)

        return new_coordinates, new_visibility

    def __get_color_at_keypoint(self, image: MatLike, keypoint):
        x = keypoint[0]
        y = keypoint[1]

        if isnan(x) or isnan(y):
            return [-1, -1, -1]

        if x < 0 or y < 0 or image.shape[1] <= x or image.shape[0] <= y:
            return [-2, -2, -2]

        return image[int(y), int(x)]

    def __to_df_row(
        self,
        input: Series,
        xyz: CoordinateFeatures,
        vis: VisibilityFeatures,
        height: int,
        width: int,
    ) -> Series:

        xyz_relative = xyz / self.get_keypoint_scaler(height, width)

        zipped = list(zip_longest(xyz_relative, vis))
        appended = [[*coordinates, visibility] for coordinates, visibility in zipped]
        result_array = [
            element for element in array(appended).reshape(-1) if element != None
        ]

        label_names = list(iterate_valid_labels())
        label_names.sort()
        binarized_labels = input.filter(items=label_names)
        result_array = [
            *result_array,
            *binarized_labels,
            input["video"],
            input["frame_num"],
            input["group"],
        ]

        return Series(data=result_array, index=input.index)


def demo_augmentation(
    original: MatLike, augmentation: Compose, save_path: str | None = None
):
    transformed = augmentation(image=original)["image"]
    _, axarr = plt.subplots(1, 2)
    axarr[0].imshow(original)
    axarr[1].imshow(transformed)

    if save_path:
        transformed_pil = Image.fromarray(transformed)
        transformed_pil.save(save_path)
