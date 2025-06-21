from pandas import DataFrame
from cv2 import imread

def augment(series):
    print(series.index)
    print(type(series))

    # img_path = input["image_path"]
    # image = imread(img_path)
    # heigth, width, _ = image.shape
    # print(image.shape)

    return series