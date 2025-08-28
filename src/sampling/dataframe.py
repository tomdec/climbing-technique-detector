from cv2 import imread
from pandas import DataFrame, concat
from os.path import join, isdir, exists
from os import listdir, makedirs, mkdir
from numpy import zeros

from src.labels import name_to_value
from src.common.helpers import read_dataframe
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
                        matrix.append([*features, name_to_value(label), image_file_path])
        
        df = DataFrame(data=matrix, columns=column_names)
        df.to_pickle(join(df_path, f"{data_split}.pkl"))

def generate_unity_df(data_root_path, combine_for_kfold=False):
    
    feature_names = get_feature_labels()
    column_names = [*feature_names, "technique", "image_path"]

    img_path = join(data_root_path, "img", "techniques")
    df_path = join(data_root_path, "df", "unity")
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
                    encoded_label = name_to_value(label)
                    features = zeros(len(column_names) - 2)
                    features[encoded_label] = 1
                    matrix.append([*features, encoded_label, image_file_path])

        df = DataFrame(data=matrix, columns=column_names)
        df.to_pickle(join(df_path, f"{data_split}.pkl"))

    if combine_for_kfold:
        combine_dataset(data_root_path, "unity")

def combine_dataset(data_root, dataset_name):
    og_dataset_path = join(data_root, "df", dataset_name)
    train = read_dataframe(join(og_dataset_path, "train.pkl"))
    test = read_dataframe(join(og_dataset_path, "test.pkl"))
    val = read_dataframe(join(og_dataset_path, "val.pkl"))

    all = concat([train, test, val], ignore_index=True)

    kf_dataset_path = join(data_root, "df", dataset_name + '_kf')
    if not exists(kf_dataset_path):
        mkdir(kf_dataset_path)
    
    all.to_pickle(join(kf_dataset_path, "all.pkl"))