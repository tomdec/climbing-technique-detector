from ultralytics import YOLO
from os.path import join, exists
from os import listdir
from typing import Optional

from src.sota.balancing import WeightedTrainer

class SOTA:

    data_root_path: str
    name: str
    dataset_name: str 
    model: Optional[YOLO]

    def __init__(self, data_root_path: str, name: str, 
            dataset_name: str = "techniques"):
    
        if (name == ""):
            raise Exception(f"name cannot be an empty string")
        
        self.data_root_path = data_root_path
        self.name = name
        self.dataset_name = dataset_name

    def initialize_model(self, name = ""):
        if (exists(self.__get_model_dir())):
            weights_path = self.__get_best_weights_path()
            self.__load_model(weights_path)
        else:
            self.__fresh_model(name)

    def train_model(self, optimizer: str = "auto", lr0: float = 0.01, epochs=20, 
            balanced=False):
        if (self.model is None):
            raise Exception("Cannot train before model is initialized")
        
        trainer = WeightedTrainer if balanced else None

        dataset_path = self.__get_dataset_dir()
        project_path = self.__get_project_dir()
        results = self.model.train(trainer=trainer,
            data=dataset_path, 
            epochs=epochs,
            imgsz=640,
            project=project_path,
            optimizer=optimizer,
            lr0=lr0)
        
        print(results)

    def __get_model_dir(self):
        return join(self.data_root_path, "runs", "sota", self.name)

    def __get_dataset_dir(self):
        return join(self.data_root_path, "img", self.dataset_name)

    def __get_project_dir(self):
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

    def test_model(self):
        if (self.model is None):
            raise Exception("Cannot test before model is initialized")
        
        print(f"Make sure to swap val and test splits for {self.dataset_name}, otherwise validation data will be used.")
        
        self.model.val()
    
# results = model.predict(img, verbose = False)
# result = results[0]
# idx = result.probs.top1
# conf = result.probs.top1conf.item()
# label = model.names[idx]

#TODO: try https://github.com/rigvedrs/YOLO-V11-CAM for activation heatmaps


