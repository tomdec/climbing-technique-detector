from typing import Any, List
from cv2 import COLOR_BGR2RGB, cvtColor
from cv2.typing import MatLike
from mediapipe.python.solutions.holistic import Holistic
from numpy import ndarray
from pandas import DataFrame
from re import search
from os.path import join
from os import makedirs

from src.common.helpers import read_dataframe
from src.hpe.mp.draw import draw_my_landmarks, draw_features
from src.hpe.mp.evaluate import predict_landmarks
from src.hpe.mp.landmarks import get_feature_labels
from src.hpe.mp.model import build_holistic_model
from src.labels import get_labels_as_dataframe, get_label_by_frame_num, LabelsCSV
from src.video.play.common import play_video

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

def extract_hpe_dataset(video_path: str) -> DataFrame:
    df_output = video_path.replace("data/videos", "data/df/videos").replace(".mp4", ".pkl")
    label_path = video_path.replace("data/videos", "data/labels").replace(".mp4", ".csv")
    
    labels = get_labels_as_dataframe(label_path)

    with build_holistic_model(static_image_model=False) as model:
        context = HpeExtractionContext(model=model, labels=labels)
        play_video(video_path=video_path,
            context = context,
            mutators=[hpe_extractor_mutator])
        
    columns = ["frame_num", "label", *get_feature_labels()]
    df = DataFrame(data=context.results, columns=columns)
    df.to_pickle(df_output)

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
        segment: DataFrame = self.df.iloc[self.start_frame:self.frame]
        segment = segment.reset_index(drop=True)
        path = join("data", "df", "segments", self.label)
        makedirs(path, exist_ok=True)
        segment.to_pickle(join(path, f"{self.name}__{self.start_frame}.pkl"))

def ask_input():
    print("What would you like to do?:",
        "r: replay segment",
        "y: accept landmarks of segment",
        "n: don't use landmarks for training",
        "q: quit iteration",
        sep='\n')
    response = input("answer:")    
    if response in ['r', 'y', 'n', 'q']:
        return response
    else:
        print("incorrect input")
        return ask_input()

def extract_segment_landmarks(segment_path: str, 
        context: SegmentContext | None = None):
    
    if context is None:
        context = SegmentContext(segment_path)
    else:
        context.reset()
    
    def mutator(img, context: SegmentContext):
        features = context.get_features()
        img = draw_features(img, features)
        context.increment_frame()
        return img

    play_video(segment_path, context=context, mutators=[mutator])

    response = ask_input()
    if response == 'r':
        return extract_segment_landmarks(segment_path, context)
    else:
        return context, response