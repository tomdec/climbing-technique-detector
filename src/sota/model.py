from ultralytics import YOLO
from os.path import join, exists
from os import listdir, rename
from typing import Optional
from wandb import finish, init
from wandb.integration.ultralytics import add_wandb_callback
from json import dump, load

from src.sota.balancing import WeightedTrainer

class SOTA:

    data_root_path: str
    name: str
    dataset_name: str 
    model: Optional[YOLO] = None

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

    def execute_train_runs(self, model, runs=1, optimizer: str = "auto", lr0: float = 0.01, 
            epochs=20, balanced=False):
        
        for run in range(runs):
            print(f"starting run #{run}")
            self.initialize_model(name=model)
            self.train_model(optimizer=optimizer, lr0=lr0, epochs=epochs, balanced=balanced)

    def train_model(self, optimizer: str = "auto", lr0: float = 0.01, epochs=20, 
            balanced=False):
        if (self.model is None):
            raise Exception("Cannot train before model is initialized")
        
        trainer = WeightedTrainer if balanced else None

        dataset_path = self.__get_dataset_dir()
        project_path = self.__get_project_dir()

        config = {
            'name': self.name,
            'dataset_name': self.dataset_name,
            'optimizer': optimizer,
            'lr0': lr0,
            'balanced': balanced,
            'augmented': True,
            'run': self.__get_next_train_run()
        }
        init(project="detect-climbing-technique", job_type="train", group="sota", name=self.name, 
            config=config, dir=self.data_root_path)
        add_wandb_callback(self.model, enable_model_checkpointing=True)

        results = self.model.train(trainer=trainer,
            data=dataset_path, 
            epochs=epochs,
            imgsz=640,
            project=project_path,
            optimizer=optimizer,
            lr0=lr0)
        
        finish()
        
        print(results)

    def __get_model_dir(self):
        return join(self.data_root_path, "runs", "sota", self.name)

    def __get_dataset_dir(self):
        return join(self.data_root_path, "img", self.dataset_name)

    def __get_next_train_run(self):
        model_dir = self.__get_model_dir()
        if not exists(model_dir):
            return "train1"
        
        train_runs = [dir for dir in listdir(model_dir) if "train" in dir]
        return f"train{len(train_runs)+1}"

    def __get_project_dir(self):
        return join(self.data_root_path, "runs", "sota", self.name)

    def __get_best_weights_path(self):
        model_dir = self.__get_model_dir()
        train_list = [dir for dir in listdir(model_dir) if "train" in dir]
        return join(model_dir, train_list[-1], "weights", "best.pt")

    def __fresh_model(self, name):
        if (name == ""):
            name = self.name + ".yaml"

        print(f"loading a fresh model '{name}'")
        self.model = YOLO(name)
    
    def __load_model(self, best_weights_path):
        print(f"loading the model '{self.name}' with the weights at '{best_weights_path}'")
        self.model = YOLO(best_weights_path)

    def get_test_metrics(self):
        with open(join(self.__get_project_dir(), "test", "metrics.json"), "r") as file:
            return load(file)

    def test_model(self, write_to_wandb = True):
        self.initialize_model()
        
        dataset_path = self.__get_dataset_dir()
        project_path = self.__get_project_dir()

        rename(join(dataset_path, "val"), join(dataset_path, "val_temp"))
        rename(join(dataset_path, "test"), join(dataset_path, "val"))
        #try:
        if write_to_wandb:
            config = {
                'name': self.name,
                'dataset_name': self.dataset_name,
                'balanced': False,
                'augmented': False,
                'run': 'test'
            }
            init(project="detect-climbing-technique", job_type="eval", group="sota", name=self.name, 
                config=config, dir=self.data_root_path)
            add_wandb_callback(self.model, enable_model_checkpointing=True)
        
        metrics = self.model.val(project=project_path, name="test")

        with open(join(project_path, "test", "metrics.json"), "w") as file:
            dump(metrics.results_dict, file)

        if write_to_wandb:
            finish()

        rename(join(dataset_path, "val"), join(dataset_path, "test"))
        rename(join(dataset_path, "val_temp"), join(dataset_path, "val"))

        return metrics
        
        # except Exception as ex:
        #     print(f"stopped with error: {ex.message}")
        # finally:
            # rename(join(dataset_path, "val"), join(dataset_path, "test"))
            # rename(join(dataset_path, "val_temp"), join(dataset_path, "val"))
    
# results = model.predict(img, verbose = False)
# result = results[0]
# idx = result.probs.top1
# conf = result.probs.top1conf.item()
# label = model.names[idx]

#TODO: try https://github.com/rigvedrs/YOLO-V11-CAM for activation heatmaps


