from typing import List
from pandas import DataFrame
from glob import glob
from pandas import concat
from os.path import join, exists
from pathlib import Path

from src.labels import (
    get_labels_from_video,
    find_valid_segments,
    iterate_valid_labels,
    get_valid_label_count,
)
from src.common.helpers import read_dataframe


def get_majority_vote(prediction: str, window: List[str], window_size: int = 5) -> str:
    window.append(prediction)
    if len(window) > window_size:
        window.pop(0)
    return max(set(window), key=window.count)


def print_common_results(results: DataFrame):
    total = len(results.index)

    processed_acc = sum(results["processed"] == results["labels"]) / total
    print(f"Processed accuracy: {processed_acc}")

    avg_total_speed_cpu = sum(results["total_speed_cpu"]) / total
    print(f"Average CPU time: {avg_total_speed_cpu} s")

    avg_total_speed_seq = sum(results["total_speed_seq"]) / total
    print(f"Average sequential time: {avg_total_speed_seq} s")


def print_original_results(results: DataFrame):
    total = len(results.index)
    original_acc = sum(results["original"] == results["labels"]) / total
    print(f"Original accuracy: {original_acc}")


def print_hpe_inference_comparison(results: DataFrame, type_name: str):
    avg_hpe = sum(results["hpe_speed_seq"]) / len(results.index)
    avg_inference = sum(results["inference_speed_seq"]) / len(results.index)
    print(f"Average HPE sequential time: {avg_hpe} s")
    print(f"Average inference sequential time: {avg_inference} s")
    ratio_hpe = avg_hpe / (avg_hpe + avg_inference)
    ratio_inference = 1 - ratio_hpe
    print(
        f"Ratio between HPE extraction and {type_name} inference:",
        f"{ratio_hpe:.1%}/{ratio_inference:.1%}",
    )


def combine_model_type_results(model_type_root: str) -> DataFrame:
    paths = glob(model_type_root + "/**/*.*", recursive=True)
    frames = []
    for df_path in paths:
        df = read_dataframe(df_path)
        frames.append(df.copy())
    df = concat(frames)
    return df


def print_all_results(evaluation_root: str):
    """
    Deprecated. Use print_results function from each specific model type you need.
    TODO: replace all references
    """
    model_type_root = join(evaluation_root, "sota")
    if exists(model_type_root):
        df = combine_model_type_results(model_type_root)
        print("Report of SOTA results:")
        print_common_results(df)

    model_type_root = join(evaluation_root, "dnn")
    if exists(model_type_root):
        df = combine_model_type_results(model_type_root)
        print()
        print("Report of HPE DNN results:")
        print_common_results(df)
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
        print_common_results(df)
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


def print_precision_recall_per_class(model_type_root: str):
    df = combine_model_type_results(model_type_root)
    total = len(df.index)
    label_count = df["labels"].value_counts()
    if "INVALID" in label_count.index:
        label_count = label_count.drop("INVALID")
    guess_precisions = label_count / total
    guess_recall = 1 / get_valid_label_count()

    tp_c = df.query("labels == processed")["labels"].value_counts()
    fp_c = df.query("labels != processed")["processed"].value_counts()
    fn_c = df.query("labels != processed")["labels"].value_counts()
    if "INVALID" in fn_c.index:
        fn_c = fn_c.drop("INVALID")

    precision = {
        label: tp_c[label] / (tp_c[label] + fp_c[label])
        for label in iterate_valid_labels()
    }
    recall = {
        label: tp_c[label] / (tp_c[label] + fn_c[label])
        for label in iterate_valid_labels()
    }

    for label in iterate_valid_labels():
        pr_guess = guess_precisions[label]
        re_guess = guess_recall

        pr_model = precision[label]
        re_model = recall[label]
        if pr_guess < pr_model:
            pr_msg_template = (
                "Model precision ({:.1%}) is greater than guessing precision ({:.1%})."
            )
        else:
            pr_msg_template = (
                "Model precision ({:.1%}) is lower than or equal to guessing "
                "precision ({:.1%})."
            )

        if re_guess < re_model:
            re_msg_template = (
                "Model recall ({:.1%}) is greater than guessing recall ({:.1%})."
            )
        else:
            re_msg_template = (
                "Model recall ({:.1%}) is lower than or equal to guessing recall "
                "({:.1%})."
            )

        print(f"{label}:")
        print(pr_msg_template.format(pr_model, pr_guess))
        print(re_msg_template.format(re_model, re_guess))
        print()
