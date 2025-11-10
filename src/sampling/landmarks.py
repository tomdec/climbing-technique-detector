from typing import Any, List
from cv2 import COLOR_BGR2RGB, cvtColor
from cv2.typing import MatLike
from mediapipe.python.solutions.holistic import Holistic
from numpy import ndarray
from pandas import DataFrame

from src.hpe.mp.draw import draw_my_landmarks
from src.hpe.mp.evaluate import predict_landmarks
from src.hpe.mp.landmarks import get_feature_labels
from src.hpe.mp.model import build_holistic_model
from src.labels import get_labels_as_dataframe, get_label_by_frame_num, LabelsDF
from src.video.play.common import play_video

class HpeExtractionContext:

    def __init__(self, model: Holistic, labels: LabelsDF):
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