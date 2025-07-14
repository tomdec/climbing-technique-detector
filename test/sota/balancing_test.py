from src.sota.balancing import WeightedDataset
from numpy import random

def test_sampling_of_WeightedDataset():
    img_path = "data/img/techniques"
    random.seed(10)
    techniques = []
    
    dataset = WeightedDataset(img_path=img_path, mode="train")
    for idx in range(10):
        row = dataset[idx]
        techniques.append(row)

    assert False
