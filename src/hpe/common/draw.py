from typing import List
from cv2.typing import MatLike

from src.hpe.common.typing import HpeEstimation, KeypointDrawConfig, HpeEstimations

def draw_estimations(estimations: List[HpeEstimation],
        label_draw_config: KeypointDrawConfig = KeypointDrawConfig(),
        predictions_draw_config: KeypointDrawConfig = KeypointDrawConfig()) -> MatLike:
    
    est_list = HpeEstimations(estimations)
    return est_list.draw(label_draw_config, predictions_draw_config)