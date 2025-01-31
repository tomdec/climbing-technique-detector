from src.video.download import download_yt_video

__list = [
    # Movement for climbers
    "https://www.youtube.com/watch?v=jLsii8s6acM&t=8s"
    
]
__output_dir = "./data/videos"


if __name__ == "__main__":
    for url in __list:
        download_yt_video(url, __output_dir)