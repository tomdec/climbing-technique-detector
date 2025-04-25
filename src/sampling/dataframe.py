from cv2 import imread
from pandas import DataFrame
from os.path import join, isdir, exists
from os import listdir, mkdir

from src.labels import Technique
from src.hpe.model import build_holistic_model
from src.hpe.evaluate import to_feature_vector
from src.hpe.landmarks import get_feature_labels

def generate_hpe_feature_df(data_path) -> DataFrame:

    feature_names = get_feature_labels()
    column_names = [*feature_names, "technique"]
    matrix = []

    img_path = join(data_path, "img", "techniques")
 
    with build_holistic_model() as model:

        for data_split in listdir(img_path):
            data_split_path = join(img_path, data_split)
            if (isdir(data_split_path)):
                for label in listdir(data_split_path):
                    label_path = join(data_split_path, label)
                    for image_name in listdir(label_path):
                        image_file_path = join(label_path, image_name)
                        
                        image = imread(image_file_path)
                        features = to_feature_vector(model, image)
                        matrix.append([*features, Technique[label].value])

    df = DataFrame(data=matrix, columns=column_names)
    
    df_path = join(data_path, "df")
    if (not exists(df_path)):
        mkdir(df_path)

    df.to_pickle(join(data_path, "df", "hpe-dnn-data.pkl"))
    return df