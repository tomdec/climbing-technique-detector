import pytest
from random import seed
from os.path import exists, join
from shutil import rmtree

from src.sampling.images import build_image_dirs, data_slice_factory, random_init_skip

@pytest.mark.parametrize("input,expected", [
        ((1, 0, 0), ["train"] * 10),
        ((0, 1, 0), ["val"] * 10),
        ((0, 0, 1), ["test"] * 10),
        ((1, 1, 1), ["train", "train", "val", "train", "test", "train", "val", "train",  "test", "train"]),
        ((0.33, 0.33, 0.33), ["train", "train", "val", "train", "test", "train", "val", "train",  "test", "train"]),
    ])
def test_data_slice_factory(input, expected):
    seed(123)

    factory = data_slice_factory(input)

    for elem in expected:
        actual = factory()
        assert actual == elem


def test_random_init_skip():
    seed(456)

    actual = random_init_skip(5)

    assert actual == 3


def test_random_init_skip__with_float():
    seed(456)

    actual = random_init_skip(5.5)

    assert actual == 3

def test_build_mage_dirs():
    root="./test/labelTestDir"

    try:
        build_image_dirs(root)

        assert not exists(join(root, "train", "INVALID"))
        assert exists(join(root, "train", "NONE"))
        assert exists(join(root, "test", "FOOT_SWAP"))
        assert exists(join(root, "val", "OUTSIDE_FLAG"))

    finally:
        rmtree(root)