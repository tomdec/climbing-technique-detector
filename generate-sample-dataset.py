from argparse import ArgumentParser

__data_root = "data"

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="generate-sample-dataset",
        description="""
            This script will generate video fragments under 'data/samples' for videos in 
            'data/videos' and labelled by .csv files in 'data/labels'.
            These fragments are only generated for valid labels (all except INVALID) and their names
            will follow the pattern: '{video_file_name}__{start_frame}.mp4'
        """,
    )
    args = parser.parse_args()

    from src.sampling.segments import generate_all_segments

    generate_all_segments(__data_root)
