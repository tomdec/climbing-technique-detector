from ultralytics import YOLO
from os.path import join

def get_fresh_model():
    return YOLO("yolo11m-cls.yaml")

def get_trained_model(data_root_path):
    best_weights_path = join(data_root_path, "runs", "classify", "train", "weights", "best.pt")
    return YOLO(best_weights_path)

def train_fresh_model(data_root_path):
    model = get_fresh_model()
    project_path = join(data_root_path, "runs", "sota")
    dataset_path = join(data_root_path, "img", "techniques")

    results = model.train(data=dataset_path, 
                          epochs=20, 
                          imgsz=640,
                          project=project_path)
    print(results)

def train_best_model(data_root_path):
    model = get_trained_model(data_root_path)
    project_path = join(data_root_path, "runs", "sota")
    dataset_path = join(data_root_path, "img", "techniques")
    
    results = model.train(data=dataset_path, 
                          epochs=20, 
                          imgsz=640,
                          project=project_path)
    print(results)

#TODO: try https://github.com/rigvedrs/YOLO-V11-CAM for activation heatmaps
