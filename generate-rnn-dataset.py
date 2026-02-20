from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="generate-rnn-dataset",
        description="Generate full dataset hpe features grouped by continuous valid segments.",
    )
    args = parser.parse_args()

    from pandas import concat
    from glob import glob

    from src.labels import (
        find_valid_segments,
        get_labels_from_video,
    )
    from src.sampling.landmarks import get_landmark_df_path
    from src.common.helpers import read_dataframe, save_dataframe

    group_id = 0
    all_features = None
    videos = glob("data/videos/*.*")
    for video in videos:
        label_path = get_labels_from_video(video)
        valids = find_valid_segments(label_path)

        hpe_path = get_landmark_df_path(video)
        hpe_features = read_dataframe(hpe_path)

        for segment in valids:
            seg_length = segment[1] - segment[0]
            seg_slice = slice(segment[0], segment[1])

            segment_features = hpe_features[seg_slice].copy()
            segment_features["video"] = video
            segment_features["group"] = group_id
            if all_features is None:
                all_features = segment_features
            else:
                all_features = concat(
                    [all_features, segment_features], axis=0, ignore_index=True
                )
            group_id += 1

    frame_num = all_features.pop("frame_num")
    video = all_features.pop("video")
    group = all_features.pop("group")

    all_features = concat([video, frame_num, group, all_features], axis=1)

    path = "data/df/rnn/cvs_features.pkl"
    save_dataframe(path, all_features)
