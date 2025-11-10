from mediapipe.python.solutions.holistic import Holistic

def build_holistic_model(
        static_image_model: bool = True,
        min_detection_confidence=0.0):
    return Holistic(
        static_image_mode=static_image_model,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence)