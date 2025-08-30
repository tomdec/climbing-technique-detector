from typing import Callable, Any, List, Dict
from numpy import array

from src.hpe.common.labels import MyLandmark
from src.hpe.common.labels import get_mylandmark_count
from src.hpe.common.labels import YoloLabels
from src.hpe.common.helpers import eucl_distance

def PCKh50_general(ytrue: YoloLabels, get_prediction: Callable[[MyLandmark], Any | None]) -> List[bool]:
    limit = ytrue.get_head_bone_link() / 2
    
    correct_pred = list()

    for landmark in MyLandmark:
        yhat_value = get_prediction(landmark)
        if yhat_value is None:
            continue
        ytrue_value = ytrue._key_points[landmark]
        
        is_correct = eucl_distance(ytrue_value.as_array(), array([yhat_value.x, yhat_value.y])) <= limit
        correct_pred.append(is_correct)
    
    return correct_pred

def count_values(map: Dict[MyLandmark, bool | None], value: bool | None):
    count = 0
    for landmark in MyLandmark:
        if map[landmark] == value:
            count += 1

    return count

def performance(map):
    correct_num = count_values(map, True)
    cant_detect = count_values(map, None)

    return correct_num, get_mylandmark_count() - cant_detect

def log_overall_performance(maps,
        model_name: str = "The model"):
    total_correct = 0
    max_possible = 0
    _, detected = performance(maps[0])
    
    for map in maps:
        correct, possible = performance(map)
        total_correct += correct
        max_possible += possible

    print(f'Percentage of landmarks detected correctly: {total_correct / max_possible}%')
    print(f'{model_name} can only detect {detected} landmarks of our {get_mylandmark_count()} labels')