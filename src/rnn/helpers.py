from pandas import DataFrame
from pandas import concat
from typing import Tuple

from src.hpe.mp.landmarks import get_feature_labels
from src.labels import iterate_valid_labels


def get_features(df: DataFrame) -> DataFrame:
    return df.filter(items=get_feature_labels())


def get_binarized_labels(df: DataFrame) -> DataFrame:
    label_names = list(iterate_valid_labels())
    label_names.sort()
    return df.filter(items=label_names)


def get_admin_columns(df: DataFrame) -> DataFrame:
    return df.filter(items=["video", "frame_num", "group"])


def split_df(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    features = get_features(df)
    labels = get_binarized_labels(df)
    admin_cols = get_admin_columns(df)
    return features, labels, admin_cols


def combine_df(
    features: DataFrame, labels: DataFrame, admin_columns: DataFrame
) -> DataFrame:
    return concat([features, labels, admin_columns], axis=1)
