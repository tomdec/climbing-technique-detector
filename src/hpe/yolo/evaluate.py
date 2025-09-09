from ultralytics import YOLO
from ultralytics.engine.results import Results
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_BGR2RGB
from cv2.typing import MatLike
from typing import Tuple, List, Any

def evaluate_images(image_paths: List[str]) -> List[List[Any]]:

    def image_2_features(image_path: str) -> List[Any]:
        #TODO: ?
        pass
    
    return list(map(image_2_features, image_paths))

def predict_landmarks(image_path: str | MatLike, model: YOLO) -> Tuple[MatLike, Results, Tuple[int, int]]:
    results = model(image_path)
    image = cvtColor(results[0].orig_img, COLOR_BGR2RGB)

    return image, results[0], results[0].orig_shape
