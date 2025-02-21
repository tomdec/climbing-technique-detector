from src.video.sampling.images import generate_image_dataset_from_samples

__data_root = "./data"

if __name__ == '__main__':
    generate_image_dataset_from_samples(__data_root)