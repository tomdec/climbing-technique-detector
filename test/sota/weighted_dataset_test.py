from ultralytics import YOLO
from os.path import exists
from shutil import rmtree
from glob import glob
import pytest

from src.sota.balancing import WeightedTrainer

test_dataset_path = "test/sota/data"
test_project_path = "test/sota/run"
   
def no_test_files() -> bool:
    if not exists(test_dataset_path):
        return True
    
    files = glob(test_dataset_path + "/**/*.*", recursive=True)
    return len(files) == 0

@pytest.mark.skipif(
    no_test_files(), reason="requires test data to run test"
)
def test_sota_balancing():

    model = YOLO("yolo11n-cls")
    trainer = WeightedTrainer

    try:

        metrics = model.train(data=test_dataset_path, 
            trainer=trainer,
            epochs=1,
            project=test_project_path)

        assert metrics is not None

    finally: 
        if exists(test_project_path):
            rmtree(test_project_path)