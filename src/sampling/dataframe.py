from cv2 import imread
from pandas import DataFrame
from os.path import join, isdir, exists
from os import listdir, makedirs

from src.labels import Technique
from src.hpe.model import build_holistic_model
from src.hpe.evaluate import to_feature_vector
from src.hpe.landmarks import get_feature_labels

def generate_hpe_feature_df(data_path,
        dataset_name = "techniques"):

    feature_names = get_feature_labels()
    column_names = [*feature_names, "technique", "image_path"]
    
    img_path = join(data_path, "img", dataset_name)
    df_path = join(data_path, "df", dataset_name)
    if (not exists(df_path)):
        makedirs(df_path)

    for data_split in listdir(img_path):
        matrix = []
        data_split_path = join(img_path, data_split)
        if (isdir(data_split_path)):
            for label in listdir(data_split_path):
                label_path = join(data_split_path, label)
                for image_name in listdir(label_path):
                    image_file_path = join(label_path, image_name)
                    image = imread(image_file_path)
                    
                    with build_holistic_model() as model:
                        features = to_feature_vector(model, image)
                        matrix.append([*features, Technique[label].value, image_file_path])
        
        df = DataFrame(data=matrix, columns=column_names)
        df.to_pickle(join(df_path, f"{data_split}.pkl"))