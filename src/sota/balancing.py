from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer
from typing import List

class WeightedDataset(ClassificationDataset):

    counts: List[int]

    def __init__(self, root, args, augment=False, prefix=""):
        super(WeightedDataset, self).__init__(root, args, augment, prefix)

        print("Inject reporting lines")

        self.train_mode = "train" in self.prefix

        self.count_instances()
        pass

    def count_instances(self):
        self.counts = {}
        for cls in self.base.classes:
            self.counts[cls] = 0

        pass

    
    def __getitem__(self, index):
        return super(WeightedDataset, self).__getitem__(index)
    
class WeightedTrainer(ClassificationTrainer):
    
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return WeightedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)