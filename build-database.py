from argparse import ArgumentParser


__list = []
__output_dir = "data/videos"


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="build-database",
        description=f"""
            Download publically available videos to use as training, validation or testing data.
            Videos are stored in the output folder '{__output_dir}'.

            Add your own videos in this folder if you want to expand the video used in the project.
        """,
    )
    args = parser.parse_args()

    from src.video.download import download_yt_video

    for url in __list:
        download_yt_video(url, __output_dir)
