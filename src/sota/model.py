from ultralytics import YOLO
from os.path import join
from os import listdir

def get_fresh_model():
    return YOLO("yolo11m-cls.yaml")

def __get_last_train(train_list):
    return train_list[-1]

def get_trained_model(data_root_path):
    model_root_path = join(data_root_path, "runs", "sota")
    train_list = listdir(model_root_path)
    newest_train = __get_last_train(train_list)
    print(f"Starting from best model weights from train run: '{newest_train}'")
    best_weights_path = join(model_root_path, newest_train, "weights", "best.pt")
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

def train_best_model(data_root_path,
        optimizer: str = "auto",
        lr0: float = 0.01):
    model = get_trained_model(data_root_path)
    project_path = join(data_root_path, "runs", "sota")
    dataset_path = join(data_root_path, "img", "techniques")
    
    results = model.train(data=dataset_path, 
                          epochs=20, 
                          imgsz=640,
                          project=project_path,
                          optimizer=optimizer,
                          lr0=lr0)
    print(results)

#TODO: try https://github.com/rigvedrs/YOLO-V11-CAM for activation heatmaps
