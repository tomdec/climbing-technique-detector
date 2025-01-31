from yt_dlp import YoutubeDL

def download_yt_video(video_url, output_dir):
    yt_opts = {
        'format': 'best',  # Specify the format you want
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s'
    }

    with YoutubeDL(yt_opts) as ytdl:
        ytdl.download(video_url)