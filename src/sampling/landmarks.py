from glob import glob
from typing import Any, List, Tuple
from cv2 import COLOR_BGR2RGB, cvtColor
from cv2.typing import MatLike
from mediapipe.python.solutions.holistic import Holistic
from numpy import ndarray, array, save, load
from pandas import DataFrame
from re import search
from os.path import join, exists
from os import makedirs

from src.common.helpers import read_dataframe, save_dataframe
from src.hpe.mp.draw import draw_my_landmarks, draw_features
from src.hpe.mp.evaluate import predict_landmarks
from src.hpe.mp.landmarks import get_feature_labels
from src.hpe.mp.model import build_holistic_model
from src.labels import (
    get_labels_as_dataframe,
    get_label_by_frame_num,
    LabelsCSV,
    get_labels_from_video,
)
from src.video.play.common import play_video, iterate_video


def get_landmark_df_path(video_path: str) -> str:
    return video_path.replace("/videos/", "/df/videos/").replace(".mp4", ".pkl")


# VIDEO
class HpeExtractionContext:

    def __init__(self, model: Holistic, labels: LabelsCSV):
        self.model = model
        self.labels = labels
        self.results: List[List[Any]] = []
        self.current_index = 0

    def get_current_label(self) -> str:
        return get_label_by_frame_num(self.labels, self.current_index)

    def append_features(self, features: ndarray):
        label = self.get_current_label()
        self.results.append([self.current_index, label, *features])

    def increment_index(self):
        self.current_index += 1


def hpe_extractor_mutator(img: MatLike, context: HpeExtractionContext) -> MatLike:
    img_rgb = cvtColor(img, COLOR_BGR2RGB)
    landmarks = predict_landmarks(img_rgb, context.model)
    context.append_features(landmarks.to_array())
    context.increment_index()
    return draw_my_landmarks(img, landmarks)


def extract_hpe_dataset(video_path: str, inspect: bool = False) -> DataFrame:
    df_output = get_landmark_df_path(video_path)
    label_path = get_labels_from_video(video_path)

    labels = get_labels_as_dataframe(label_path)

    with build_holistic_model(static_image_model=False) as model:
        context = HpeExtractionContext(model=model, labels=labels)
        if inspect:
            play_video(
                video_path=video_path, context=context, mutators=[hpe_extractor_mutator]
            )
        else:
            iterate_video(
                video_path=video_path, context=context, mutators=[hpe_extractor_mutator]
            )

    columns = ["frame_num", "label", *get_feature_labels()]
    df = DataFrame(data=context.results, columns=columns)
    save_dataframe(df_output, df)

    return df


# SEGMENT
class SegmentContext:

    def __init__(self, segment_path: str):
        m = search(r"data\/samples\/(\w*)\/(.+)__(\d+)\.mp4", segment_path)
        self.label = m.group(1)
        self.name = m.group(2)
        self.start_frame = int(m.group(3))
        self.frame = self.start_frame
        self.df: DataFrame = read_dataframe(f"data/df/videos/{self.name}.pkl")

    def get_features(self) -> ndarray:
        return self.df.iloc[self.frame][2:]

    def increment_frame(self):
        self.frame += 1

    def reset(self):
        self.frame = self.start_frame

    def store_segment(self):
        segment: DataFrame = self.df.iloc[self.start_frame : self.frame]
        segment = segment.reset_index(drop=True)
        path = join(
            "data", "df", "segments", self.label, f"{self.name}__{self.start_frame}.pkl"
        )
        save_dataframe(path, segment)


def ask_input():
    print(
        "What would you like to do?:",
        "r: replay segment",
        "s: replay at half speed (accumulative)",
        "y: accept landmarks of segment",
        "n: don't use landmarks for training",
        "q: quit iteration",
        sep="\n",
    )
    response = input("answer:")
    if response in ["r", "s", "y", "n", "q"]:
        return response
    else:
        print("incorrect input")
        return ask_input()


def extract_segment_landmarks(
    segment_path: str, context: SegmentContext | None = None, playback_speed: float = 1
) -> Tuple[SegmentContext, str]:

    if context is None:
        context = SegmentContext(segment_path)
    else:
        context.reset()

    def mutator(img, context: SegmentContext):
        features = context.get_features()
        img = draw_features(img, features)
        context.increment_frame()
        return img

    play_video(
        segment_path, context=context, mutators=[mutator], playback_speed=playback_speed
    )

    response = ask_input()
    if response == "r":  # replay
        return extract_segment_landmarks(segment_path, context, playback_speed)
    if response == "s":  # slow replay
        return extract_segment_landmarks(segment_path, context, 0.5 * playback_speed)
    else:
        return context, response


def remove_non_existing_segments(
    accepted_segments: List[str], segments: List[str]
) -> Tuple[List[str], bool]:
    result = [
        accepted_segment
        for accepted_segment in accepted_segments
        if accepted_segment in segments
    ]
    changed = len(result) != len(accepted_segments)
    if changed:
        removed = [
            accepted_segment
            for accepted_segment in accepted_segments
            if accepted_segment not in segments
        ]
        print(f"Removed {len(removed)} segments:")
        print(removed)
    return result, changed


def validate_accepted_segments(label: str):
    segments = glob(join("data", "samples", label, "*.*"), recursive=True)
    evaluated_segments = glob(
        join("data", "df", "segments", label, "*.pkl"), recursive=True
    )

    accepted_segments_path = join("data", "df", "segments", label, "accepted.npy")
    if exists(accepted_segments_path):
        accepted_segments = load(accepted_segments_path)
        accepted_segments, changed = remove_non_existing_segments(
            accepted_segments, segments
        )
        if changed:
            save(accepted_segments_path, array(accepted_segments))
    else:
        accepted_segments = []

    if len(segments) == len(evaluated_segments):
        print("Landmarks are extracted for all segments")
    else:
        print(
            f"Only extracted landmarks for {len(evaluated_segments)} out of {len(segments)} segments"
        )

    percentage = len(accepted_segments) / len(segments)
    print(
        f"Accepted the landmarks of {len(accepted_segments)} out of {len(segments)} segments ({percentage:.2%})"
    )
