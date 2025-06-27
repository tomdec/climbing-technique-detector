from cv2 import circle
from math import isnan
from matplotlib import pyplot as plt

__KEYPOINT_COLOR = (0, 255, 0)  # Green

def draw_augmented_keypoints(image, keypoints):
    diameter = 5
    image = image.copy()

    for x, y, _ in keypoints:
        if (x is not None and y is not None and not isnan(x) and not isnan(y)):
            circle(image, (int(x), int(y)), diameter, __KEYPOINT_COLOR, -1)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(image)