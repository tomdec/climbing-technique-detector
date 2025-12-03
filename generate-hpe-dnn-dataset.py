from argparse import ArgumentParser
from src.labels import get_dataset_name

__data_root = "data"

if __name__ == '__main__':
    parser = ArgumentParser(prog="generate-hpe-dnn-dataset",
        description="Generate HPE landmark dataset based on the image dataset.")
    parser.add_argument("name",
        default=get_dataset_name(),
        help="Name of image dataset to generate hpe dataset from.")
    parser.add_argument("-c", "--combine", 
        action="store_true",
        help="Combine an existing dataset for k-fold validation.",
        required=False)
    args = parser.parse_args()

    from src.sampling.dataframe import combine_dataset, generate_hpe_feature_df
    
    dataset_name = args.name
    
    if args.combine:
        print(f"Combining {dataset_name} data splits.")
        combine_dataset(__data_root, dataset_name)
    else:
        generate_hpe_feature_df(__data_root, dataset_name)