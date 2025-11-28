from numpy import ndarray, array, nan, arange
from pandas import DataFrame, Series

from src.hpe.common.typing import MyLandmark
from src.common.helpers import safe_index
from src.hpe.common.helpers import eucl_distance
from src.hpe.common.landmarks import PredictedKeyPoints, YoloLabels
from src.hpe.common.typing import HpeEstimation, PerformanceMap

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
    if totals[""] > 0:
        return nan
    
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
    if totals[""] > 0:
        return nan
    
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

def calc_precision_and_recall(estimations: DataFrame) -> DataFrame:
    """Calculate the precision and recall metrics for a range of confidence thresholds.

    Args:
        estimations (DataFrame): DataFrame containing HpeEstimation objects.
        With the rows (index) being the sample images and the columns the detected classes.

    Returns:
        DataFrame: A new DataFrame containing dictionaries like:
        ```
        {
            "p": 1.0,   # precision value
            "r": 0.0    # recall value
        }
        ```
        With each row (index) representing a confidence thresholds and each column still the 
        detected classes, with "CONFIDENCE" appended for the actual confidence theshold values.
    """
    conf_increment = 0.01
    confidences = arange(0, 1 + conf_increment, conf_increment)
    result = DataFrame()
    
    def pnr_dict(totals: DataFrame, verbose: bool=False) -> dict:
        return {
            'p': _precision(totals, verbose),
            'r': _recall(totals, verbose)
        }
    
    for idx, conf in enumerate(confidences):
        
        def prediction_result(estimation: HpeEstimation) -> str:
            return estimation.prediction_result(conf)

        prediction_results = estimations.map(prediction_result)
        counts = prediction_results.apply(Series.value_counts).fillna(0)
        
        landmark_precisions = counts.apply(pnr_dict, axis=0)
        landmark_precisions.at["CONFIDENCE"] = conf
        result[idx] = landmark_precisions

    return result.transpose()

def calc_average_precision(column: Series, verbose: bool = False) -> float:
    average_precision = 0
    previous_recall = 0

    if verbose: print("0")
    
    for cell in list(reversed(column))[1:]: #skip pnr point @ 1 confidence threshold, should not make a difference since p is always (?) 0
        precision = cell['p']
        recall = cell['r']

        average_precision = average_precision + abs(recall - previous_recall) * precision

        if verbose and (recall != previous_recall): 
            print(f"  + |({recall} - {previous_recall})| * {precision} = {average_precision}")

        previous_recall = recall
    
    if verbose: print(f"Average precision is: {average_precision}")

    return average_precision

def calc_average_precisions(pnr: DataFrame, verbose: bool = False) -> DataFrame:
    pnr = pnr.drop("CONFIDENCE",axis=1)
    return pnr.apply(func=(lambda x: calc_average_precision(column=x, verbose=verbose)), axis=0)

def calc_mean_average_precision(pnr: DataFrame, verbose: bool = False) -> float:
    return calc_average_precisions(pnr, verbose).mean(skipna=True)

def calc_throughput(estimations: DataFrame) -> float:
    estimations = estimations.drop(columns=["NECK"])
    detected = sum(sum(estimations.map(HpeEstimation.is_detected).values))
    present = sum(sum(estimations.map(HpeEstimation.is_present).values))
    
    return detected / present

def calc_accuracy(estimations: DataFrame) -> float:
    estimations = estimations.drop(columns=["NECK"])
    correct = sum(sum(estimations.map(HpeEstimation.is_correct).values))
    present_and_recognizable = sum(sum(estimations.map(HpeEstimation.is_present_and_recognizable).values))
    
    return correct / present_and_recognizable