from sys import argv
from src.sampling.dataframe import combine_dataset, generate_hpe_feature_df
from src.labels import get_dataset_name

__data_root = "./data"

if __name__ == '__main__':

    arguments = argv[1:]
    dataset_name = get_dataset_name()

    if "-c" in arguments:
        combine_dataset(__data_root, dataset_name)
    else:
        generate_hpe_feature_df(__data_root, dataset_name)