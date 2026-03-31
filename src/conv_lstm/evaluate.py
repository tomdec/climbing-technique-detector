import tensorflow as tf
from cv2 import VideoCapture, CAP_PROP_POS_FRAMES, cvtColor, COLOR_BGR2RGB, resize
from cv2.typing import MatLike
from sklearn.preprocessing import LabelBinarizer
from pandas import DataFrame
from numpy import array
from time import process_time, perf_counter
from pathlib import Path
from typing import List
from os.path import join, exists

from src.common.evaluate import combine_model_type_results
from src.common.helpers import save_dataframe
from src.sampling.images import FRAME_SIZE
from src.labels import (
    iterate_valid_labels,
    get_labels_from_video,
    get_label_by_frame_num,
    find_valid_segments,
    get_labels_as_dataframe,
)
from src.conv_lstm.data import WindowGenerator
from src.conv_lstm.model import ConvLstm


def _get_prediction(
    frames: List[MatLike], model: ConvLstm, binarizer: LabelBinarizer
) -> str:
    data = tf.stack([frames])
    output = array(model.model(data))
    prediction = binarizer.inverse_transform(output[0])[0]
    return prediction


def collect_evaluation_performance(
    video_path: str, model: ConvLstm, wg: WindowGenerator
) -> DataFrame:
    label_path = get_labels_from_video(video_path)
    valids = find_valid_segments(label_path)
    labels = get_labels_as_dataframe(label_path)

    input_width = model.model_initialize_args.input_width

    binarizer = LabelBinarizer()
    binarizer.fit(list(iterate_valid_labels()))

    frame = []
    labels_arr = []
    processed = []
    inference_speed_cpu = []
    inference_speed_seq = []

    vid_capture = VideoCapture(video_path)
    try:
        for start_frame, stop_frame in valids:

            frame_num = start_frame
            vid_capture.set(CAP_PROP_POS_FRAMES, frame_num)

            frames = []

            while vid_capture.isOpened() and frame_num < stop_frame:
                _, image = vid_capture.read()
                image = cvtColor(image, COLOR_BGR2RGB)
                image = resize(image, (FRAME_SIZE, FRAME_SIZE))
                frames.append(image)
                if len(frames) > input_width:
                    frames = frames[-input_width:]

                frame.append(frame_num)
                label = get_label_by_frame_num(labels, frame_num)
                labels_arr.append(label)

                t0_cpu = process_time()
                t0_seq = perf_counter()

                pred = _get_prediction(frames, model, binarizer)

                t1_cpu = process_time()
                t1_seq = perf_counter()

                processed.append(pred)

                inference_speed_cpu.append(t1_cpu - t0_cpu)
                inference_speed_seq.append(t1_seq - t0_seq)

                frame_num += 1
    finally:
        vid_capture.release()

    results = DataFrame(
        data=zip(
            frame,
            labels_arr,
            processed,
            inference_speed_cpu,
            inference_speed_seq,
        ),
        columns=[
            "frame",
            "labels",
            "processed",
            "inference_speed_cpu",
            "inference_speed_seq",
        ],
    )
    filename = Path(video_path).stem
    save_dataframe(f"data/df/evaluation_results/conv_lstm/{filename}.pkl", results)
    return results


def print_results(results: DataFrame):
    total = len(results.index)

    processed_acc = sum(results["processed"] == results["labels"]) / total
    print(f"Processed accuracy: {processed_acc}")

    avg_total_speed_cpu = sum(results["inference_speed_cpu"]) / total
    print(f"Average CPU time: {avg_total_speed_cpu} s")

    avg_total_speed_seq = sum(results["inference_speed_seq"]) / total
    print(f"Average sequential time: {avg_total_speed_seq} s")


def print_all_results(evaluation_root: str):
    model_type_root = join(evaluation_root, "conv_lstm")
    if exists(model_type_root):
        df = combine_model_type_results(model_type_root)
        print()
        print("Report of CONV_LSTM results:")
        print_results(df)
