from typing import List
from cv2.typing import MatLike

from src.common.helpers import imread
from src.hpe.common.typing import HpeEstimation

def draw_estimations(estimations: List[HpeEstimation]) -> MatLike:
    image_path = estimations[0].image_path
    image = imread(image_path)
    
    for estimation in estimations:
        if estimation.image_path != image_path:
            raise Exception(f"Trying to draw estimations from image {estimation.image_path} on image {image_path}")
        
        if estimation.true_landmark is not None:
            image = estimation.true_landmark.draw(image)
        
        image = estimation.predicted_landmark.draw(image)

    return image