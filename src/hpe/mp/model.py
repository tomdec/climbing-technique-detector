from mediapipe.python.solutions.holistic import Holistic

def build_holistic_model():
    return Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False)