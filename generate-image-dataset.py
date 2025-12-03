from argparse import ArgumentParser


__data_root = "./data"

if __name__ == '__main__':
    parser = ArgumentParser(prog="generate-image-dataset",
        description="Generate image dataset for SOTA training, based on labelled segments.")
    parser.add_argument("--accepted", action="store_true", 
        help="Only use data from accepted segments.")
    args = parser.parse_args()

    from src.sampling.images import generate_image_dataset_from_samples

    generate_image_dataset_from_samples(__data_root, 
        data_split=(0.7, 0.15, 0.15),
        only_accepted=args.accepted)