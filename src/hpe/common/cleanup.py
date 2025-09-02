from shutil import rmtree
from typing import Callable
from os.path import join, exists

def remove_temp_factory(data_root: str = "data") -> Callable:

    def remove_temp():    
        for split in ["train", "test", "valid"]:
            path = join(data_root, "hpe", "img", split, "temp")
            if exists(path):
                rmtree(path)

    return remove_temp