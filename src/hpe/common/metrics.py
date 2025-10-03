from typing import Dict
from numpy import ndarray, array, nan

from src.hpe.common.helpers import eucl_distance
from src.hpe.common.landmarks import MyLandmark, PredictedKeyPoints, YoloLabels

PerformanceMap = Dict[MyLandmark, bool | None]

def PCKh50(ytrue: YoloLabels, yhat: PredictedKeyPoints) -> PerformanceMap:
    limit = ytrue.get_head_bone_link() / 2
    
    def map_performance(landmark: MyLandmark) -> bool | None:
        if not yhat.can_predict(landmark):
            return None
        
        ytrue_value = ytrue.get_keypoint(landmark)
        yhat_value = yhat[landmark]

        if ytrue_value.is_missing() and yhat_value.is_missing():
            # True Negative
            return True
        elif ytrue_value.is_missing() and not yhat_value.is_missing():
            # False Positive
            return False
        elif not ytrue_value.is_missing() and yhat_value.is_missing():
            # False Negative
            return False
        else:
            return eucl_distance(ytrue_value.as_array(), yhat_value.as_array()) <= limit
        
    performance = dict(map(lambda x: (x, map_performance(x)), MyLandmark))
    return performance

def distance(ytrue: YoloLabels, yhat: PredictedKeyPoints) -> ndarray:
    limit = ytrue.get_head_bone_link() / 2

    def map_distance(landmark: MyLandmark) -> float:
        if not yhat.can_predict(landmark):
            return nan
        
        ytrue_value = ytrue.get_keypoint(landmark)
        yhat_value = yhat[landmark]

        if ytrue_value.is_missing() and yhat_value.is_missing():
            # True Negative
            return 0
        elif ytrue_value.is_missing() and not yhat_value.is_missing():
            # False Positive
            return nan # should be handle differently than 'cannot predict' case
        elif not ytrue_value.is_missing() and yhat_value.is_missing():
            # False Negative
            return nan # should be handle differently than 'cannot predict' case
        else:
            return eucl_distance(ytrue_value.as_array(), yhat_value.as_array()) / limit

    distances = list(map(map_distance, MyLandmark))
    return array(distances)