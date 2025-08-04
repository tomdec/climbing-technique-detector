from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

from src.common.balancing import BalancedSampler

class WeightedDataset(ClassificationDataset):

    train_mode: bool
    _sampler: BalancedSampler

    def __init__(self, root, args, augment=False, prefix=""):
        super(WeightedDataset, self).__init__(root, args, augment, prefix)

        self.train_mode = "train" in self.prefix
        
        self._sampler = BalancedSampler(list(self.base.classes))
        
        #Use path to image as sample, match by parsing path to search for label name
        self._sampler.count_classes(self.samples, lambda sample, cls: f'/{cls}/' in sample[0])

    def __getitem__(self, index):
        if self.train_mode:
            index = self._sampler.next_balanced()
            
        return super(WeightedDataset, self).__getitem__(index)
    
    def report_balancing(self):
        print("Original distribution:")
        print(dict(zip(self.classes, self.class_counts)))

    
class WeightedTrainer(ClassificationTrainer):
    
    def build_dataset(self, img_path: str, mode: str = "train"):
        return WeightedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)