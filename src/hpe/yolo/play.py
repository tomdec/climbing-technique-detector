from ultralytics import YOLO
from cv2.typing import MatLike

from src.video.play.common import play_video
from src.hpe.yolo.model import build_pose_model
from src.hpe.yolo.evaluate import predict_landmarks
from src.hpe.yolo.draw import draw_my_landmarks

def predict_and_draw(image: MatLike, model: YOLO) -> MatLike:
    image.flags.writeable = False                # Image is no longer writeable
    
    results = predict_landmarks(image, model)
    image = draw_my_landmarks(image, results)

    return image

def play_with_hpe(video_path: str):
    model = build_pose_model()
    play_video(video_path, 
        context=model,
        mutators=[predict_and_draw])