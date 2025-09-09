from cv2.typing import MatLike
from mediapipe.python.solutions.holistic import Holistic

from src.video.play.common import play_video
from src.hpe.mp.model import build_holistic_model
from src.hpe.mp.evaluate import predict_landmarks
from src.hpe.mp.draw import draw_my_landmarks

def predict_and_draw(image: MatLike, model: Holistic) -> MatLike:
    image.flags.writeable = False                # Image is no longer writeable
    
    results, _ = predict_landmarks(image, model)
    image = draw_my_landmarks(image, results)

    return image

def play_with_hpe(video_path: str):
    with build_holistic_model() as model:
        play_video(video_path, 
            context=model,       
            mutators=[predict_and_draw])