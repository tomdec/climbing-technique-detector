from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

class WeightedDataset(ClassificationDataset):

    def __init__(self, root, args, augment=False, prefix=""):
        super(WeightedDataset, self).__init__(root, args, augment, prefix)

        print("Inject reporting lines")

        self.train_mode = "train" in self.prefix
        pass

    def count_instances(self):
        pass        

    
    def __getitem__(self, index):
        print(f"getting item at: {index}")
        return super(WeightedDataset, self).__getitem__(index)
    
class WeightedTrainer(ClassificationTrainer):
    
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return WeightedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)