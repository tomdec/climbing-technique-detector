from cv2.typing import MatLike

from src.hpe.common.helpers import eucl_distance
from src.hpe.common.typing.LabelKeyPoint import LabelKeyPoint
from src.hpe.common.typing.PredictedKeyPoint import PredictedKeyPoint 
from src.hpe.common.typing.KeypointDrawConfig import KeypointDrawConfig

class HpeEstimation:

    @staticmethod
    def from_dict(values: dict) -> 'HpeEstimation':
        return HpeEstimation(
            name=values['name'],
            true_landmark=None
                if values['true_landmark'] is None
                else LabelKeyPoint.from_dict(
                    values=values['true_landmark'],
                    name=values['name']),
            predicted_landmark=None
                if values['predicted_landmark'] is None
                else PredictedKeyPoint.from_dict(
                    values=values['predicted_landmark'],
                    name=values['name']),
            head_bone_link=values['head_bone_link'],
            image_path=values['image_path'],
            can_predict=values['can_predict']
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def true_landmark(self) -> LabelKeyPoint | None:
        return self._true_landmark

    @property
    def predicted_landmark(self) -> PredictedKeyPoint:
        return self._predicted_landmark

    @property
    def head_bone_link(self) -> float | None:
        return self._head_bone_link

    @property
    def image_path(self) -> str:
        return self._image_path

    @property
    def can_predict(self) -> bool:
        return self._can_predict

    def __init__(self,
            name: str,
            true_landmark: LabelKeyPoint | None,
            predicted_landmark: PredictedKeyPoint,
            head_bone_link: float | None,
            image_path: str,
            can_predict: bool):
        self._name = name
        self._true_landmark = true_landmark
        self._predicted_landmark = predicted_landmark
        self._head_bone_link = head_bone_link
        self._image_path = image_path
        self._can_predict = can_predict

    def __str__(self) -> str:
        return f"{self.as_dict()}"

    def _get_prediction(self, conf_threshold: float, verbose: bool) -> PredictedKeyPoint | None:
        if self._predicted_landmark.is_missing():
            if verbose: print("No prediction because landmark is missing")
            return None
        elif self._predicted_landmark.visibility < conf_threshold:
            if verbose: print("No prediction because confidence is too low")
            return None
        else:
            if verbose: print("Prediction is made")
            return self._predicted_landmark

    def get_distance(self, verbose: bool = False) -> float:
        true = self._true_landmark.as_array()
        prediction = self._predicted_landmark.as_array()
        distance = eucl_distance(true, prediction)

        if verbose:
            print(f"True landmark: x = {true[0]}, y = {true[1]}")
            print(f"Predicted landmark: x = {prediction[0]}, y = {prediction[1]}")
            print(f"Euclidian distance: {distance}")

        return distance

    def get_relative_distance(self, verbose: bool = False) -> float:
        abs_dist = self.get_distance(verbose=verbose)
        return abs_dist / (self.head_bone_link / 2)

    def is_detected(self) -> bool:
        """Checks if a labelled landmark is detected or not.

        Returns:
            bool: True when a landmark is detected, False when the landmark was not labeled or
            not estimated.
        """
        return self.is_present() and \
            not self.predicted_landmark.is_missing()

    def is_present(self) -> bool:
        """Checks if the landmark is present.

        Returns:
            bool: True if the landmark is labeled, else False.
        """
        return self.true_landmark is not None and \
            not self.true_landmark.is_missing()

    def is_correct(self) -> bool:
        """Checks if the estimation is correct, according to the PCKh50 metric, 
        without considering a confidence threshold.

        Returns:
            bool: True if labeled and estimated landmark are close enough, else False.
        """
        return self.prediction_result() == "TP"

    def is_present_and_recognizable(self) -> bool:
        return self.can_predict and self.is_present()

    def prediction_result(self, conf_threshold: float = 0.0,
            verbose: bool = False) -> str:
        if not self._can_predict:
            return ""

        estimation = self._get_prediction(conf_threshold, verbose)
        if self._true_landmark is None and estimation is None:
            if verbose: print("TN")
            return "TN"
        elif self._true_landmark is not None and estimation is None:
            if verbose: print("FN")
            return "FN"
        elif self._true_landmark is None and estimation is not None:
            if verbose: print("FP")
            return "FP"
        else:
            limit = self._head_bone_link / 2
            distance = self.get_distance(verbose)
            correct = distance <= limit
            #TODO: maybe this needs to be different?
            #Should an incorrect landmark count as both:
            # - a FP, for the predicted landmark that is incorrect
            # - and FN, for the true landmark that is not detected?
            result = "TP" if correct else "FP"

            if verbose:
                print(f"limit: {limit}")
                print(result)

            return result

    def draw(self, image: MatLike,
            prediction_config: KeypointDrawConfig = KeypointDrawConfig()):

        pass

    def as_dict(self) -> dict:
        return {
            'name': self._name,
            'true_landmark': None 
                if self._true_landmark is None 
                else self._true_landmark.as_dict(),
            'predicted_landmark': self._predicted_landmark.as_dict(),
            'head_bone_link': self._head_bone_link,
            'image_path': self._image_path,
            'can_predict': self._can_predict
        }