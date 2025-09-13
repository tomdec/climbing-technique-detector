from sys import argv

from src.sampling.dataframe import combine_dataset, generate_hpe_feature_df
from src.labels import get_dataset_name
from src.hpe.mp.landmarks import get_feature_labels as get_mp_features
from src.hpe.mp.evaluate import extract_features as extract_with_mp
from src.hpe.yolo.landmarks import get_feature_labels as get_yolo_features
from src.hpe.yolo.evaluate import extract_features as extract_with_yolo


__data_root = "data"

if __name__ == '__main__':

    arguments = argv[1:]
    img_dataset_name = get_dataset_name()

    if "--yolo" in arguments:
        features_names = get_yolo_features()
        evaluate_func = extract_with_yolo
        df_dataset_name = img_dataset_name + "_yolo"
    else:
        features_names = get_mp_features()
        evaluate_func = extract_with_mp
        df_dataset_name = img_dataset_name + "_mp"
    
    if "-c" in arguments:
        combine_dataset(__data_root, df_dataset_name)
    else:
        generate_hpe_feature_df(data_path=__data_root, 
            feature_names=features_names,
            evaluate_func=evaluate_func,
            img_dataset_name=img_dataset_name,
            df_dataset_name=df_dataset_name)