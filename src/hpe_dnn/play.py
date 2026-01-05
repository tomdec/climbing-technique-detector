from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from cv2.typing import MatLike

from src.labels import get_labels_from_video, get_labels_as_dataframe, iterate_valid_labels,\
    get_label_by_frame_num
from src.common.draw import write_label_and_prediction
from src.hpe_dnn.model import HpeDnn
from src.hpe_dnn.evaluate import evaluate_with_majority_voting
from src.hpe.mp.model import build_holistic_model
from src.video.play.common import play_video

def play_with_label_and_prediction(video_path: str, model: HpeDnn, start_frame: int = 0,
        stop_frame: int | None = None):
    label_path = get_labels_from_video(video_path)
    labels = get_labels_as_dataframe(label_path)
    
    imputer = SimpleImputer(missing_values=None, strategy='constant', fill_value=0, 
        keep_empty_features=True)

    binarizer = LabelBinarizer()
    binarizer.fit(list(iterate_valid_labels()))

    with build_holistic_model(static_image_model=False) as hpe_tool:
        
        def mutator(image: MatLike, current_frame: int) -> MatLike:
            label = get_label_by_frame_num(labels, current_frame)
            prediction = evaluate_with_majority_voting(image, model, hpe_tool, [], imputer, binarizer)
            image = write_label_and_prediction(image, label, prediction)
            return image

        play_video(video_path, mutators=[mutator], start_frame=start_frame, stop_frame=stop_frame)