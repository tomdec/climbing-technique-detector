from argparse import ArgumentParser

__data_root = "data"

if __name__ == '__main__':
    parser = ArgumentParser(prog="generate-hpe-dnn-dataset",
        description="Generate HPE landmark dataset based on the image dataset.")
    parser.add_argument("-c", "--combine", 
        action="store_true",
        help="Combine an existing dataset for k-fold validation.")
    args = parser.parse_args()

    from src.sampling.dataframe import combine_dataset, generate_hpe_feature_df
    from src.labels import get_dataset_name

    dataset_name = get_dataset_name()
    
    if args.combine:
        print(f"Combining {dataset_name} data splits.")
        combine_dataset(__data_root, dataset_name)
    else:
        generate_hpe_feature_df(__data_root, dataset_name)