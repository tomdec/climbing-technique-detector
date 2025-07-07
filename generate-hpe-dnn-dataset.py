from src.sampling.dataframe import generate_hpe_feature_df
from sys import argv

__data_root = "./data"

if __name__ == '__main__':

    arguments = argv[1:]
    balanced = "-b" in arguments
    
    if balanced:
        print("Generating balanced dataset")

    dataset_name = "techniques_balanced" if balanced else "techniques"
    generate_hpe_feature_df(__data_root, dataset_name)