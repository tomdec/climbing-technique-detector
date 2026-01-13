from matplotlib import pyplot as plt
from PIL import Image
from PIL.ImageFile import ImageFile

from torchvision.transforms import Compose

def demo_augmentation(original: ImageFile, augmentation: Compose, save_path: str | None = None):
    transformed = augmentation(original)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(original)
    axarr[1].imshow(transformed)
    if save_path:
        transformed.save(save_path)