from src.video.download import download_yt_video

__list = [
    # Movement for climbers
    #"https://www.youtube.com/watch?v=jLsii8s6acM&t=8s",
    #"https://www.youtube.com/watch?v=2LEnGELJxw0",
    #"https://www.youtube.com/watch?v=U0v9xxqEjfY",
    "https://www.youtube.com/watch?v=LOkhYCi-IrE",
    
    #Lattice Training
    "https://www.youtube.com/watch?v=s-ArwEzr-2M",
    "https://www.youtube.com/watch?v=EBYpXAii9WQ"
]
__output_dir = "./data/videos"


if __name__ == "__main__":
    for url in __list:
        download_yt_video(url, __output_dir)