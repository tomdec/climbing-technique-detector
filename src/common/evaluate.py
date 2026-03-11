from typing import List
from pandas import DataFrame
from glob import glob
from pandas import concat
from os.path import join, exists
from pathlib import Path

from src.labels import get_labels_from_video, find_valid_segments
from src.common.helpers import read_dataframe


def get_majority_vote(prediction: str, window: List[str], window_size: int = 5) -> str:
    window.append(prediction)
    if len(window) > window_size:
        window.pop(0)
    return max(set(window), key=window.count)


def print_results(results: DataFrame):
    total = len(results.index)
    if "original" in results.columns:
        original_acc = sum(results["original"] == results["labels"]) / total
        print(f"Original accuracy: {original_acc}")

    processed_acc = sum(results["processed"] == results["labels"]) / total
    print(f"Processed accuracy: {processed_acc}")

    avg_total_speed_cpu = sum(results["total_speed_cpu"]) / total
    print(f"Average CPU time: {avg_total_speed_cpu} s")

    avg_total_speed_seq = sum(results["total_speed_seq"]) / total
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
    if exists(model_type_root):
        df = combine_model_type_results(model_type_root)
        print("Report of SOTA results:")
        print_results(df)

    model_type_root = join(evaluation_root, "dnn")
    if exists(model_type_root):
        df = combine_model_type_results(model_type_root)
        print()
        print("Report of HPE DNN results:")
        print_results(df)
        avg_hpe = sum(df["hpe_speed_seq"]) / len(df.index)
        avg_inference = sum(df["inference_speed_seq"]) / len(df.index)
        print(f"Average HPE sequential time: {avg_hpe} s")
        print(f"Average inference sequential time: {avg_inference} s")
        ratio_hpe = avg_hpe / (avg_hpe + avg_inference)
        ratio_inference = 1 - ratio_hpe
        print(
            f"Ratio between HPE extraction and DNN inference: {ratio_hpe:.1%}/{ratio_inference:.1%}"
        )

    model_type_root = join(evaluation_root, "rnn")
    if exists(model_type_root):
        df = combine_model_type_results(model_type_root)
        print()
        print("Report of RNN results:")
        print_results(df)
        avg_hpe = sum(df["hpe_speed_seq"]) / len(df.index)
        avg_inference = sum(df["inference_speed_seq"]) / len(df.index)
        print(f"Average HPE sequential time: {avg_hpe} s")
        print(f"Average inference sequential time: {avg_inference} s")
        ratio_hpe = avg_hpe / (avg_hpe + avg_inference)
        ratio_inference = 1 - ratio_hpe
        print(
            f"Ratio between HPE extraction and RNN inference: {ratio_hpe:.1%}/{ratio_inference:.1%}"
        )


def get_evaluation_results_path(video_path: str, model_type: str) -> str:
    filename = Path(video_path).stem
    return f"data/df/evaluation_results/{model_type}/{filename}.pkl"


def get_results_per_cvs(model_type: str) -> DataFrame:
    video_paths = glob("data/videos/*")
    original_acc = []
    processed_acc = []

    for video_path in video_paths:
        results_path = get_evaluation_results_path(video_path, model_type)
        label_path = get_labels_from_video(video_path)
        results = read_dataframe(results_path)
        valid_segments = find_valid_segments(label_path)

        for start, stop in valid_segments:
            segment = results.query(f"{start} <= frame and frame < {stop}")
            count = len(segment.index)

            if "original" in segment.columns:
                original_acc.append(
                    sum(segment["original"] == segment["labels"]) / count
                )
            processed_acc.append(sum(segment["processed"] == segment["labels"]) / count)

    if len(original_acc) > 0:
        return DataFrame(
            data=zip(original_acc, processed_acc), columns=["original", "processed"]
        )
    else:
        return DataFrame(data=processed_acc, columns=["processed"])
