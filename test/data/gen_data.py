from cv2 import VideoWriter, imread, VideoWriter_fourcc
from os import makedirs
from os.path import join, exists


test_data = "test/data"

if __name__ == "__main__":
    from os.path import dirname, join
    from sys import path

    this_dir = dirname(__file__)
    mymodule_dir = this_dir.replace("/test/data", "")
    print(mymodule_dir)
    path.append(mymodule_dir)

    from src.labels import get_valid_label_count

    video_file_dir = join(test_data, "video")
    video_file_path = join(video_file_dir, "test_video.mp4")
    if not exists(video_file_path):
        makedirs(video_file_dir, exist_ok=True)

        image = imread(join(test_data, "img", "NONE", "test_image.jpg"))
        frame_width = image.shape[0]
        frame_height = image.shape[1]
        writer = VideoWriter(
            filename=video_file_path,
            fourcc=VideoWriter_fourcc("m", "p", "4", "v"),
            fps=30,
            frameSize=(frame_width, frame_height),
        )
        try:
            for x in range(100):
                writer.write(image)
        finally:
            writer.release()
