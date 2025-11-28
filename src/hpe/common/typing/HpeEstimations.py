from cv2 import line
from cv2.typing import MatLike
from typing import List, Dict, Tuple

from src.common.helpers import imread
from src.hpe.common.typing.MyLandmark import MyLandmark, CONNECTIONS
from src.hpe.common.typing.HpeEstimation import HpeEstimation
from src.hpe.common.typing.KeypointDrawConfig import KeypointDrawConfig

class HpeEstimations:

    _estimations: Dict[MyLandmark, HpeEstimation]

    def __init__(self, estimations: List[HpeEstimation]):
        
        def to_kvp(estimation: HpeEstimation) -> Tuple[MyLandmark, HpeEstimation]:
            return (MyLandmark[estimation.name], estimation)
        
        self._estimations = dict(map(to_kvp, estimations))

    def draw(self, label_config: KeypointDrawConfig,
            prediction_config: KeypointDrawConfig) -> MatLike:
        image_path = list(self._estimations.values())[0].image_path
        image = imread(image_path)
        
        for estimation in self._estimations.values():
            if estimation.image_path != image_path:
                raise Exception(f"Trying to draw estimations from image {estimation.image_path}" +
                    f"on image {image_path}")
            
            image = estimation.draw(image, label_config, prediction_config)


        image_height, image_width, _ = image.shape
        for connection in CONNECTIONS:
            from_estimation = self._estimations[connection[0]]
            to_estimation = self._estimations[connection[1]]
            
            #label
            if from_estimation.true_landmark is not None and\
                    to_estimation.true_landmark is not None and\
                    not from_estimation.true_landmark.is_missing() and\
                    not to_estimation.true_landmark.is_missing():
                from_point = from_estimation.true_landmark
                to_point = to_estimation.true_landmark

                start = (int(from_point.x * image_width), int(from_point.y * image_height))
                end = (int(to_point.x * image_width), int(to_point.y * image_height))
                image = line(image, start, end, label_config.center_color, 5)

            #prediction
            if not from_estimation.predicted_landmark.is_missing() and\
                    not to_estimation.predicted_landmark.is_missing():
                from_point = from_estimation.predicted_landmark
                to_point = to_estimation.predicted_landmark

                start = (int(from_point.x * image_width), int(from_point.y * image_height))
                end = (int(to_point.x * image_width), int(to_point.y * image_height))
                image = line(image, start, end, prediction_config.center_color, 5)

        return image