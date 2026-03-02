import tensorflow as tf
from cv2 import VideoCapture, CAP_PROP_POS_FRAMES, cvtColor, COLOR_BGR2RGB
from cv2.typing import MatLike
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from mediapipe.python.solutions.holistic import Holistic
from pandas import DataFrame
from numpy import array, ndarray
from time import process_time, perf_counter
from pathlib import Path
from typing import List
from numpy import float32
from pandas import concat

from src.common.evaluate import get_majority_vote
from src.common.helpers import save_dataframe
from src.labels import (
    iterate_valid_labels,
    get_labels_from_video,
    get_label_by_frame_num,
    find_valid_segments,
    get_labels_as_dataframe,
)
from src.hpe.mp.landmarks import get_feature_labels
from src.hpe.mp.evaluate import to_feature_vector
from src.hpe.mp.model import build_holistic_model
from src.rnn.architecture import RnnArch
from src.rnn.data import WindowGenerator
from src.rnn.model import Rnn, RnnConstructorArgs, RnnModelInitializeArgs


def _get_input_features(
    landmarks: DataFrame | None,
    image: MatLike,
    hpe_tool: Holistic,
    wg: WindowGenerator,
    input_width: int,
) -> DataFrame:
    new_landmarks = to_feature_vector(image, hpe_tool)
    new_landmarks = DataFrame(
        [new_landmarks], columns=get_feature_labels(), dtype=float32
    )
    new_landmarks = wg.get_processed_data(new_landmarks)

    if landmarks is None:
        landmarks = new_landmarks
    else:
        landmarks = concat([landmarks, new_landmarks], axis=0, ignore_index=True)
    landmarks = landmarks.tail(input_width)
    return landmarks


def _get_prediction(landmarks: DataFrame, model: Rnn, binarizer: LabelBinarizer) -> str:
    data = array(landmarks, dtype=float32)
    data = tf.stack([data])
    output = array(model.model(data))
    prediction = binarizer.inverse_transform(output[0])[0]
    return prediction


def collect_evaluation_performance(
    video_path: str, model: Rnn, wg: WindowGenerator
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
    hpe_speed_cpu = []
    inference_speed_cpu = []
    total_speed_cpu = []
    hpe_speed_seq = []
    inference_speed_seq = []
    total_speed_seq = []

    with build_holistic_model(static_image_model=False) as hpe_tool:
        vid_capture = VideoCapture(video_path)
        try:
            # for valid_segment in valids:
            valid_segment = valids[0]

            start_frame = valid_segment[0]
            stop_frame = valid_segment[1]

            frame_num = start_frame
            vid_capture.set(CAP_PROP_POS_FRAMES, frame_num)

            landmarks = None

            while vid_capture.isOpened() and frame_num < stop_frame:
                _, image = vid_capture.read()
                image = cvtColor(image, COLOR_BGR2RGB)

                frame.append(frame_num)

                label = get_label_by_frame_num(labels, frame_num)
                labels_arr.append(label)

                t0_cpu = process_time()
                t0_seq = perf_counter()
                landmarks = _get_input_features(
                    landmarks, image, hpe_tool, wg, input_width
                )
                t1_cpu = process_time()
                t1_seq = perf_counter()
                pred = _get_prediction(landmarks, model, binarizer)
                t2_cpu = process_time()
                t2_seq = perf_counter()

                processed.append(pred)

                hpe_speed_cpu.append(t1_cpu - t0_cpu)
                inference_speed_cpu.append(t2_cpu - t1_cpu)
                total_speed_cpu.append(t2_cpu - t0_cpu)

                hpe_speed_seq.append(t1_seq - t0_seq)
                inference_speed_seq.append(t2_seq - t1_seq)
                total_speed_seq.append(t2_seq - t0_seq)

                frame_num += 1
        finally:
            vid_capture.release()

    results = DataFrame(
        data=zip(
            frame,
            labels_arr,
            processed,
            hpe_speed_cpu,
            inference_speed_cpu,
            total_speed_cpu,
            hpe_speed_seq,
            inference_speed_seq,
            total_speed_seq,
        ),
        columns=[
            "frame",
            "labels",
            "processed",
            "hpe_speed_cpu",
            "inference_speed_cpu",
            "total_speed_cpu",
            "hpe_speed_seq",
            "inference_speed_seq",
            "total_speed_seq",
        ],
    )
    filename = Path(video_path).stem
    save_dataframe(f"data/df/evaluation_results/rnn/{filename}.pkl", results)
    return results
