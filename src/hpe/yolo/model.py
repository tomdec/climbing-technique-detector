from ultralytics import YOLO

__model_name = "yolo11m-pose.pt"

def build_pose_model() -> YOLO:
    return YOLO(__model_name)