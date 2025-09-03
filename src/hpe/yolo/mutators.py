from os import makedirs
from os.path import dirname
from cv2 import imread as imread_as_bgr, imwrite, cvtColor, COLOR_BGR2RGB
from src.common.helpers import imread
import matplotlib.pyplot as plt

def adapt_to_bgr(image_path: str) -> str:
    bgr_path = image_path.replace("/images/", "/temp/")

    bgr_image = imread_as_bgr(image_path)
    
    bgr_image = cvtColor(bgr_image, COLOR_BGR2RGB)
    makedirs(dirname(bgr_path), exist_ok=True)
    imwrite(bgr_path, bgr_image)

    return bgr_path