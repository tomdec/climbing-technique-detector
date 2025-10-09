from typing import Dict, Tuple, List
from numpy import ndarray, array, nan, arange
from pandas import DataFrame, Series

from src.common.helpers import safe_index
from src.hpe.common.helpers import eucl_distance
from src.hpe.common.landmarks import MyLandmark, PredictedKeyPoints, YoloLabels
from src.hpe.common.typing import HpeEstimation

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

def _precision(totals: DataFrame, verbose: bool = False) -> float:
    tp = safe_index(totals, "TP")
    fp = safe_index(totals, "FP")

    if verbose:
        print(f"TP: {tp}")
        print(f"FP: {fp}")

    if tp + fp == 0:
        return 0.0
    
    result = tp  / (tp + fp)
    if verbose:
        print(f"{tp} / ({tp} + {fp}) = {result}")

    return result

def _recall(totals: DataFrame, verbose: bool = False) -> float:
    tp = safe_index(totals, "TP")
    fn = safe_index(totals, "FN")

    if verbose:
        print(f"TP: {tp}")
        print(f"FN: {fn}")

    if tp + fn == 0:
        return 0
    
    result = tp  / (tp + fn)
    if verbose:
        print(f"{tp} / ({tp} + {fn}) = {result}")
    return result

PrecisionList = List[float]
RecallList = List[float]
def calc_precision_and_recall(estimations: DataFrame) -> Tuple[PrecisionList, RecallList]:
    conf_increment = 0.01
    confidences = arange(0, 1, conf_increment)
    recall_x = [0] * len(confidences)
    precision_y = [0] * len(confidences)
    
    for idx, conf in enumerate(confidences):
        
        def prediction_result(estimation: HpeEstimation) -> str:
            return estimation.prediction_result(conf)

        prediction_results = estimations.map(prediction_result)
        counts = prediction_results.apply(Series.value_counts).fillna(0)
        totals = counts.apply(sum, axis=1)
        current_recall = _recall(totals)
        current_precision = _precision(totals)
        
        recall_x[idx] = current_recall
        precision_y[idx] = current_precision
    
    return precision_y, recall_x

def calc_average_precision(precision_list: List[float], recall_list: List[float],
        verbose: bool = False):
    average_precision = 0
    previous_recall = 0

    for precision, recall in zip(precision_list, recall_list):
        average_precision = average_precision + abs(recall - previous_recall) * precision
        previous_recall = recall
    
    if verbose: print(f"Average precision is: {average_precision}")

    return average_precision