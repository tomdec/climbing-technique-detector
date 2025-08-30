from ultralytics import YOLO
from ultralytics.engine.results import Results
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_BGR2RGB
from cv2.typing import MatLike
from typing import Tuple

def predict_landmarks(image_path: str, model: YOLO) -> Tuple[MatLike, Results, Tuple[int, int]]:
    results = model(image_path)
    image = cvtColor(results[0].orig_img, COLOR_BGR2RGB)

    return image, results[0], results[0].orig_shape