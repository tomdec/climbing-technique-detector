from typing import List
from pandas import DataFrame
from glob import glob
from pandas import concat
from os.path import join

from src.common.helpers import read_dataframe

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

def combine_model_type_results(model_type_root: str) -> DataFrame:
    paths = glob(model_type_root + "/**/*.*", recursive=True)
    frames = []
    for df_path in paths:
        df = read_dataframe(df_path)
        frames.append(df.copy())
    df = concat(frames)
    return df

def print_all_results(evaluation_root: str):
    model_type_root = join(evaluation_root, "sota")
    df = combine_model_type_results(model_type_root)
    print("Report of SOTA results:")
    print_results(df)

    model_type_root = join(evaluation_root, "dnn")
    df = combine_model_type_results(model_type_root)
    print()
    print("Report of HPE DNN results:")
    print_results(df)
    avg_hpe = sum(df["hpe_speed_seq"]) / len(df.index)
    avg_inference = sum(df["inference_speed_seq"]) / len(df.index)
    ratio_hpe = avg_hpe / (avg_hpe + avg_inference)
    ratio_inference = 1 - ratio_hpe
    print(f"Ratio between HPE extraction and DNN inference: {ratio_hpe:.1%}/{ratio_inference:.1%}")

    