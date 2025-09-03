from numpy.linalg import norm
from numpy.typing import ArrayLike
from random import choice
from os import listdir
from os.path import join
from typing import List, Tuple

def eucl_distance(start: ArrayLike, stop: ArrayLike) -> float:
    return norm(start - stop)

def sample(root_dir: str) -> str:
    """Returns a random file name from the root directory"""
    file_name = choice(listdir(root_dir))
    file_path = join(root_dir, file_name)
    
    return file_path

def get_label_file(image_path: str) -> str:
    return image_path.replace('images', 'labels').replace('.jpg', '.txt')

def list_image_label_pairs(root_dir: str) -> List[Tuple[str, str]]:
    image_names = listdir(root_dir)
    
    def make_label_pair(image_name):
        image_path = join(root_dir, image_name)
        return (image_path, get_label_file(image_path))

    return list(map(make_label_pair, image_names))