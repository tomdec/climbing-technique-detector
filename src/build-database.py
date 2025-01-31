from yt_dlp import YoutubeDL

list = [
    # Movement for climbers
    "https://www.youtube.com/watch?v=jLsii8s6acM&t=8s"
    
]

output_dir = "../data/videos"

def download_yt_video(video_url):
    yt_opts = {
        'format': 'best',  # Specify the format you want
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s'
    }

    with YoutubeDL(yt_opts) as ytdl:
        ytdl.download(video_url)


if __name__ == "__main__":
    output_dir = "./data/videos"
    for url in list:
        download_yt_video(url)