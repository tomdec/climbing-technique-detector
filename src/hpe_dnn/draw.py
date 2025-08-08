from cv2 import circle, imread
from cv2.typing import MatLike
from math import isnan
from matplotlib import pyplot as plt
from pandas import Series

from src.hpe.model import build_holistic_model
from src.hpe.evaluate import predict_landmarks
from src.hpe.draw import draw_my_landmarks

__KEYPOINT_COLOR = (0, 255, 0)  # Green
__KEYPOINT_DIAMETER = 5

def __draw_coord(image: MatLike, x: int, y: int) -> MatLike:
    result = image.copy()
    if (x is not None and y is not None and not isnan(x) and not isnan(y)):
            circle(result, (int(x), int(y)), __KEYPOINT_DIAMETER, __KEYPOINT_COLOR, -1)

    return result

def draw_augmented_keypoints(image, keypoints):
    image = image.copy()

    for x, y, _ in keypoints:
        image = __draw_coord(image, x, y)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(image)

def predict_and_draw_landmarks(row: Series):
    image = imread(row['image_path'])
    with build_holistic_model() as model:
        results, shape = predict_landmarks(image, model)
        image = draw_my_landmarks(image, results)
    print(row.values)
    plt.imshow(image)

def draw_df_dataset_row(row: Series):
    image = imread(row['image_path'])
    height, width, _ = image.shape
    
    indexes = [index for index in row.index if index.endswith('_x') or index.endswith('_y')]
    coords = row[indexes].values.reshape(-1, 2)

    for x, y in coords:
        image = __draw_coord(image, int(x * width), int(y * height))
    
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(image)