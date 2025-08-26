import pytest
from os import makedirs
from os.path import exists, join
from shutil import rmtree

from src.labels import Technique, get_label, iterate_valid_labels, make_label_dirs, name_to_value, value_to_name

__location = "./data/labels/How to Flag - A Climbing Technique for Achieving Balance.csv"

@pytest.mark.parametrize("input,expected", [
        (100, Technique.INVALID),
        (500, Technique.NONE),
        (1500, Technique.NONE)
    ])
def test_get_labels(input, expected):
    actual = get_label(__location, input)

    assert actual == expected

@pytest.mark.parametrize("input,expected", [
        ("INVALID", 0),
        ("NONE", 1),
        ("FOOT_SWAP", 2),
        ("OUTSIDE_FLAG", 3),
        ("BACK_FLAG", 4),
        ("INSIDE_FLAG", 5),
        ("DROP_KNEE", 7),
        ("CROSS_MIDLINE", 8),
    ])
def test_name_to_value(input, expected):
    actual = name_to_value(input)

    assert actual == expected

@pytest.mark.parametrize("input,expected", [
        (0, "INVALID"),
        (1, "NONE"),
        (2, "FOOT_SWAP"),
        (3, "OUTSIDE_FLAG"),
        (4, "BACK_FLAG"),
        (5, "INSIDE_FLAG"),
        (7, "DROP_KNEE"),
        (8, "CROSS_MIDLINE")
    ])
def test_value_to_name(input, expected):
    actual = value_to_name(input)

    assert actual == expected

def test_iterate_valid_labels():
    expected_names = ["NONE",
        "FOOT_SWAP",
        "OUTSIDE_FLAG",
        "BACK_FLAG",
        "INSIDE_FLAG",
        "DROP_KNEE",
        "CROSS_MIDLINE"]

    iterator = iterate_valid_labels()

    for expected in expected_names:
        assert next(iterator) == expected

def test_make_label_dirs_when_they_do_not_exist():
    root="./test/labelTestDir"

    try:
        make_label_dirs(root)

        assert not exists(join(root, "INVALID"))
        assert exists(join(root, "NONE"))
        assert exists(join(root, "FOOT_SWAP"))
        assert exists(join(root, "OUTSIDE_FLAG"))

    finally:
        rmtree(root)

def test_make_label_dirs_when_they_do_exist():
    root="./test/labelTestDir"

    try:
        makedirs(join(root, "NONE"))

        make_label_dirs(root)

        assert not exists(join(root, "INVALID"))
        assert exists(join(root, "NONE"))
        assert exists(join(root, "FOOT_SWAP"))
        assert exists(join(root, "OUTSIDE_FLAG"))
        
    finally:
        rmtree(root)