
from cv2 import VideoCapture, CAP_PROP_POS_FRAMES
from cv2.typing import MatLike
from pandas import DataFrame
from time import process_time, perf_counter
from pathlib import Path
from typing import List

from src.common.evaluate import get_majority_vote
from src.common.helpers import save_dataframe
from src.labels import get_labels_from_video, get_label_by_frame_num, find_valid_segments,\
    get_labels_as_dataframe
from src.sota.model import SOTA

def evaluate(image: MatLike, model: SOTA) -> str:
    output = model.model.predict(image)[0]
    pred_name = output.names[output.probs.top1]
    return pred_name

def evaluate_with_majority_voting(image: MatLike, model: SOTA, window: List[str], 
        window_size: int = 5) -> str:
    prediction = evaluate(image, model)
    return get_majority_vote(prediction, window, window_size)

def collect_evaluation_performance(video_path: str, model: SOTA) -> DataFrame:
    label_path = get_labels_from_video(video_path)
    valids = find_valid_segments(label_path)
    labels = get_labels_as_dataframe(label_path)
    
    frame = []
    labels_arr = []
    original = []
    processed = []
    total_speed_cpu = []
    total_speed_seq = []

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
                prediction = evaluate_with_majority_voting(image, model, window, window_size)
                t1_cpu = process_time()
                t1_seq = perf_counter()

                processed.append(prediction)
                original.append(window[-1])
                
                total_speed_cpu.append(t1_cpu - t0_cpu)
                total_speed_seq.append(t1_seq - t0_seq)

                frame_num += 1
    finally:
        vid_capture.release()

    results = DataFrame(
        data=zip(frame, labels_arr, original, processed, total_speed_cpu, total_speed_seq), 
        columns=["frame", "labels", "original", "processed", "total_speed_cpu", "total_speed_seq"])
    filename = Path(video_path).stem
    save_dataframe(f"data/df/evaluation_results/sota/{filename}.pkl", results)
    return results