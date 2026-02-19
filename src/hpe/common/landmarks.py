from numpy import array, ndarray
from cv2.typing import MatLike
from typing import List, Dict
from os import listdir
from os.path import join

from src.hpe.common.typing import (
    LabelKeyPoint,
    MyLandmark,
    PredictedKeyPoint,
    Visibility,
)
from src.common.helpers import raise_not_implemented_error
from src.hpe.common.helpers import eucl_distance


class BoundingBox:

    _x: float
    _y: float
    _x_width: float
    _y_width: float

    def __init__(self, x, y, x_width, y_width):
        """
        Args:
            x: x-coordinate of the center
            y: y-coordinate of the center
            x_width: width of the bounding box
            y_width: height of the bounding box
        """
        self._x = float(x)
        self._y = float(y)
        self._x_width = float(x_width)
        self._y_width = float(y_width)

    def __str__(self):
        return f"{self.as_dict()}"

    def as_dict(self) -> dict:
        return {
            "x": self._x,
            "y": self._y,
            "x_width": self._x_width,
            "y_width": self._y_width,
        }

    def distance_to(self, point) -> float:
        return eucl_distance(array([self._x, self._y]), point)


class KeyPoints:
    _values: List[LabelKeyPoint]

    def __init__(self, values):
        self._values = list()
        start_index = 0
        while len(values) >= start_index + 3:
            stop_index = start_index + 3
            self._values.append(LabelKeyPoint(*values[start_index:stop_index]))
            start_index = stop_index

    def __getitem__(self, key: MyLandmark | int):
        if key is int:
            return self._values[key]
        else:
            return self._values[key.value]

    def __str__(self):
        return f"[{",\n".join([str(kp) for kp in self._values])}]"

    def draw(self, image: MatLike) -> MatLike:
        result = image.copy()
        for index, value in enumerate(self._values):
            if value._visibility == Visibility.MISSING:
                continue
            result = value.draw(result, str(index))
        return result


class YoloLabels:

    _class_index: int
    _bounding_box: BoundingBox
    _key_points: KeyPoints

    def __init__(self, text: str):
        values = text.split(" ")
        self._class_index = int(values[0])
        self._bounding_box = BoundingBox(*values[1:5])
        self._key_points = KeyPoints(values[5:])

    def __str__(self):
        class_str = f"'class': {self._class_index}"
        box_str = f"'bounding_box': {self._bounding_box}"
        key_points_str = f"'key_points': {self._key_points}"

        return f"{{{class_str}, {box_str}, {key_points_str}}}"

    def draw(self, image: MatLike) -> MatLike:
        return self._key_points.draw(image)

    def get_head_bone_link(self) -> float:
        head = self._key_points[MyLandmark.HEAD]
        neck = self._key_points[MyLandmark.NECK]

        return eucl_distance(head.as_array(), neck.as_array())

    def get_keypoint(self, key: MyLandmark) -> LabelKeyPoint:
        return self._key_points[key]

    def distance_to(self, point) -> float:
        return self._bounding_box.distance_to(point)

    def count_landmarks(self) -> int:
        def count_landmark(landmark: MyLandmark) -> int:
            label = self.get_keypoint(landmark)
            return 0 if label.is_missing() else 1

        return sum(list(map(count_landmark, MyLandmark)))


class PredictedKeyPoints:

    def __init__(self):
        pass

    def __getitem__(self, index: MyLandmark) -> PredictedKeyPoint:
        """Get landmark prediction for given index.
        Returns an empty landmark when the landmark was not detected.

        Args:
            index (MyLandmark): Landmark to get prediction for.

        Raises:
            Exception: When tool cannot predict given landmark.

        Returns:
            PredictedKeyPoint: Landmark prediction.
        """
        raise_not_implemented_error(self.__class__.__name__, self.__getitem__.__name__)

    def no_person_detected(self):
        raise_not_implemented_error(
            self.__class__.__name__, self.no_person_detected.__name__
        )

    def can_predict(self, landmark: MyLandmark) -> bool:
        """Check if the tool can predict this landmark.

        Args:
            landmark (MyLandmark): Landmark to check.
        """
        raise_not_implemented_error(self.__class__.__name__, self.can_predict.__name__)

    def ensure_empty(self) -> Dict[MyLandmark, bool | None]:
        """
        In the absence of a true object to detect landmarks,
        return a mapping of MyLandmark to booleans:\n
            - true, meaning no prediction (TN),
            - false, meaning an incorrect prediction (FP),
            - None, if mediapipe is not able to predict this landmark.
        """

        def no_prediction_for_landmark(landmark: MyLandmark) -> bool | None:
            if not self.can_predict(landmark):
                return None
            elif self.no_person_detected():
                return True
            else:
                yhat = self[landmark]
                return yhat is None

        results = {}
        for landmark in MyLandmark:
            results[landmark] = no_prediction_for_landmark(landmark)
        return results

    def to_array(self) -> ndarray:
        raise_not_implemented_error(self.__class__.__name__, self.to_array.__name__)


def build_yolo_labels(file_path: str) -> List[YoloLabels]:
    result = list()

    with open(file_path, "r") as file:
        file_text = file.read()

    if file_text == "":
        return result

    file_lines = file_text.split("\n")
    for line in file_lines:
        result.append(YoloLabels(line))

    return result


def get_most_central(label_list: List[YoloLabels]) -> YoloLabels:
    image_center = array([0.5, 0.5])
    result = label_list[0]
    min_distance = label_list[0].distance_to(image_center)

    for index in range(1, len(label_list)):
        distance = label_list[index].distance_to(image_center)
        if distance < min_distance:
            min_distance = distance
            result = label_list[index]

    return result


def get_mylandmark_count():
    return len(MyLandmark)


def count_labelled_hpe_landmarks():
    label_root_dir = "data/hpe/img/test/labels"
    totals = 0

    for label_name in listdir(label_root_dir):
        label_path = join(label_root_dir, label_name)
        df = build_yolo_labels(label_path)

        if len(df) == 0:
            continue
        elif len(df) == 1:
            totals += df[0].count_landmarks()
        else:
            totals += get_most_central(df).count_landmarks()

    return totals
