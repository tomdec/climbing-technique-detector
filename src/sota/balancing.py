from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer
from typing import List
from numpy import sum, where, zeros, int32, random

class WeightedDataset(ClassificationDataset):

    train_mode: bool
    classes: List[str]
    class_counts: List[int]
    sample_classes: List[int]
    class_weights: List[float]
    sample_weights: List[float]
    sample_probabilities: List[float]

    balanced_class_counts: List[int]

    def __init__(self, root, args, augment=False, prefix=""):
        super(WeightedDataset, self).__init__(root, args, augment, prefix)

        self.train_mode = "train" in self.prefix

        self.classes = self.base.classes
        self.count_instances()
        self.calculate_weights()
        self.calculate_probabilities()

    def count_instances(self):
        self.class_counts = zeros(len(self.classes), dtype=int32)
        self.balanced_class_counts = zeros(len(self.classes), dtype=int32)
        self.sample_classes = []

        for sample in self.samples:
            for idx, cls in enumerate(list(self.classes)):
                if f'/{cls}/' in sample[0]:
                    self.class_counts[idx] += 1
                    self.sample_classes.append(idx)
                    continue
        
        print(dict(zip(self.classes, self.class_counts)))

        #Correction to avoid DividedByZero errors.
        self.class_counts = where(self.class_counts == 0, 1, self.class_counts)
    
    def calculate_weights(self):
        '''
        Sets fields: 
        - class_weights:\n
            The weight for each class.
            Inverse of distribution since we need higher values for classes that are underrepresented
        - sample_weights:\n
            Weight for this particular sample. Is equal to the weight of its true class.
        '''

        self.class_weights = sum(self.class_counts) / self.class_counts 
        self.sample_weights = self.class_weights[self.sample_classes]

    def calculate_probabilities(self):
        '''
        Sets field:
        - sample_probabilities:\n
            This is the normalized sample_weights array, so that the sum of all elements is 1.
        '''
        self.sample_probabilities = self.sample_weights / sum(self.sample_weights)

    def __getitem__(self, index):
        if self.train_mode:
            index = random.choice(len(self.sample_probabilities), p=self.sample_probabilities)

            new_class_idx = self.sample_classes[index]
            self.balanced_class_counts[new_class_idx] += 1
            #TODO: write mappings to local file
            
        return super(WeightedDataset, self).__getitem__(index)
    
    def report_balancing(self):
        print("Original distribution:")
        print(dict(zip(self.classes, self.class_counts)))

        print("Balanced distribution")
        print(dict(zip(self.classes, self.balanced_class_counts)))

    
class WeightedTrainer(ClassificationTrainer):
    
    def build_dataset(self, img_path: str, mode: str = "train"):
        return WeightedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)