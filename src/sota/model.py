from ultralytics import YOLO
from os.path import join, exists
from os import listdir
from typing import Optional

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

class SOTA:

    data_root_path: str
    name: str
    dataset_name: str 
    model: Optional[YOLO]

    def __init__(self, data_root_path: str, name: str, 
            dataset_name: str = "techniques"):
    
        if (name == ""):
            raise Exception(f"'{name}' is not a valid name")
        
        self.data_root_path = data_root_path
        self.name = name
        self.dataset_name = dataset_name

    def initialize_model(self, name = ""):
        if (exists(self.__get_model_dir())):
            weights_path = self.__get_best_weights_path()
            self.__load_model(weights_path)
        else:
            self.__fresh_model(name)

    def train_model(self, optimizer: str = "auto", lr0: float = 0.01):
        if (self.model is None):
            raise Exception("Cannot train before model is initialized")
        
        project_path = join(self.data_root_path, "runs", "sota", self.name)
        dataset_path = join(self.data_root_path, "img", self.dataset_name)
        results = self.model.train(data=dataset_path, 
            epochs=20,
            imgsz=640,
            project=project_path,
            optimizer=optimizer,
            lr0=lr0)
        
        print(results)

    def __get_model_dir(self):
        return join(self.data_root_path, "runs", "sota", self.name)

    def __get_best_weights_path(self):
        model_dir = self.__get_model_dir()
        train_list = listdir(model_dir)
        return join(model_dir, train_list[-1], "weights", "best.pt")

    def __fresh_model(self, name):
        if (name == ""):
            name = self.name + ".yaml"

        print(f"loading a fresh model '{name}'")
        self.model = YOLO(name)
    
    def __load_model(self, best_weights_path):
        print(f"loading the model '{self.name}' with the weights at '{best_weights_path}'")
        self.model = YOLO(best_weights_path)
    

#TODO: try https://github.com/rigvedrs/YOLO-V11-CAM for activation heatmaps


