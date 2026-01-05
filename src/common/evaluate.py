from typing import List
from pandas import DataFrame

def get_majority_vote(prediction: str, window: List[str], window_size: int = 5) -> str:
    window.append(prediction)
    if len(window) > window_size:
        window.pop(0)
    return max(set(window), key=window.count)

def print_results(results: DataFrame):
    original_acc = sum(results["original"] == results["labels"]) / len(results.index)
    processed_acc = sum(results["processed"] == results["labels"]) / len(results.index)
    avg_total_speed_cpu = sum(results["total_speed_cpu"]) / len(results.index)
    avg_total_speed_seq = sum(results["total_speed_seq"]) / len(results.index)

    print(f"Original accuracy: {original_acc}")
    print(f"Processed accuracy: {processed_acc}")
    print(f"Average CPU time: {avg_total_speed_cpu} s")
    print(f"Average sequential time: {avg_total_speed_seq} s")