
import tensorflow as tf
from cv2 import VideoCapture, CAP_PROP_POS_FRAMES
from cv2.typing import MatLike
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from mediapipe.python.solutions.holistic import Holistic
from pandas import DataFrame
from numpy import array, ndarray
from time import process_time, perf_counter
from pathlib import Path
from typing import List

from src.common.evaluate import get_majority_vote
from src.common.helpers import save_dataframe
from src.labels import iterate_valid_labels, get_labels_from_video, get_label_by_frame_num,\
    find_valid_segments, get_labels_as_dataframe
from src.hpe.mp.landmarks import get_feature_labels
from src.hpe.mp.evaluate import to_feature_vector
from src.hpe.mp.model import build_holistic_model
from src.hpe_dnn.model import HpeDnn

def _get_input_features(image: MatLike, hpe_tool: Holistic, imputer: SimpleImputer) -> dict:
    landmarks = to_feature_vector(image, hpe_tool)
    landmarks = DataFrame([landmarks], columns=get_feature_labels())
    if landmarks.isnull().any().any():
        landmarks = DataFrame(imputer.fit_transform(landmarks), columns=landmarks.columns)
    landmarks = {key: value.to_numpy()[:,tf.newaxis] for key, value in landmarks.items()}
    return landmarks

def _get_prediction(landmarks: dict, model: HpeDnn, binarizer: LabelBinarizer) -> str:
    output = array(model.model(landmarks))
    prediction = binarizer.inverse_transform(output)[0]
    return prediction

def evaluate(image: MatLike, model: HpeDnn, hpe_tool: Holistic, 
        imputer: SimpleImputer | None = None, binarizer: LabelBinarizer | None = None) -> str:
    
    if imputer is None:
        imputer = SimpleImputer(missing_values=None, strategy='constant', fill_value=0, 
            keep_empty_features=True)
    if binarizer is None:
        binarizer = LabelBinarizer()
        binarizer.fit(list(iterate_valid_labels()))

    landmarks = _get_input_features(image, hpe_tool, imputer)
    return _get_prediction(landmarks, model, binarizer)

def evaluate_with_majority_voting(image: MatLike, model: HpeDnn, hpe_tool: Holistic, 
        window: List[str], imputer: SimpleImputer | None = None, 
        binarizer: LabelBinarizer | None = None) -> str:
    prediction = evaluate(image, model, hpe_tool, imputer, binarizer)
    return get_majority_vote(prediction, window)

def collect_evaluation_performance(video_path: str, model: HpeDnn) -> DataFrame:
    label_path = get_labels_from_video(video_path)
    valids = find_valid_segments(label_path)
    labels = get_labels_as_dataframe(label_path)
    
    imputer = SimpleImputer(missing_values=None, strategy='constant', fill_value=0, 
        keep_empty_features=True)

    binarizer = LabelBinarizer()
    binarizer.fit(list(iterate_valid_labels()))

    frame = []
    labels_arr = []
    original = []
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
            for valid_segment in valids:
                start_frame = valid_segment[0]
                stop_frame = valid_segment[1]

                frame_num = start_frame
                vid_capture.set(CAP_PROP_POS_FRAMES, frame_num)
                
                window = []
                window_size = 5
                
                while vid_capture.isOpened() and frame_num < stop_frame:
                    _, image = vid_capture.read()

                    frame.append(frame_num)

                    label = get_label_by_frame_num(labels, frame_num)
                    labels_arr.append(label)
                    
                    t0_cpu = process_time()
                    t0_seq = perf_counter()
                    landmarks = _get_input_features(image, hpe_tool, imputer)
                    t1_cpu = process_time()
                    t1_seq = perf_counter()
                    prediction = _get_prediction(landmarks, model, binarizer)
                    prediction = get_majority_vote(prediction, window, window_size)                    
                    t2_cpu = process_time()
                    t2_seq = perf_counter()
                    processed.append(prediction)
                    original.append(window[-1])
                    
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
        data=zip(frame, labels_arr, original, processed, hpe_speed_cpu, inference_speed_cpu, 
            total_speed_cpu, hpe_speed_seq, inference_speed_seq, total_speed_seq), 
        columns=["frame", "labels", "original", "processed", "hpe_speed_cpu", "inference_speed_cpu", 
            "total_speed_cpu", "hpe_speed_seq", "inference_speed_seq", "total_speed_seq"])
    filename = Path(video_path).stem
    save_dataframe(f"data/df/evaluation_results/dnn/{filename}.pkl", results)
    return results