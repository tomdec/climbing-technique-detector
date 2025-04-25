from src.sampling.images import generate_image_dataset_from_samples

__data_root = "./data"

if __name__ == '__main__':
    generate_image_dataset_from_samples(__data_root, (0.7, 0.15, 0.15))