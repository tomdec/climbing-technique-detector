from math import inf
import pytest
from os import makedirs
from os.path import exists, join
from shutil import rmtree

import src.labels as mut

"""
Mock the __labels variable, so that the test are independent from the actual /labels.yml file. 
"""
test_labels = {
        "name": "test-techniques",
        "values": [
            "INVALID", 
            "NONE", 
            "FOOT_SWAP",
            "OUTSIDE_FLAG",
            "BACK_FLAG",
            "INSIDE_FLAG",
            "DROP_KNEE",
            "CROSS_MIDLINE"
        ]
    }
mut.__labels = test_labels

def test_get_dataset_name():
    actual = mut.get_dataset_name()

    assert actual == "test-techniques"

@pytest.mark.parametrize("input,expected", [
    (100, "INVALID"),
    (500, "NONE"),
    (1500, "NONE")
])
def test_get_label_name(input, expected):
    valid_label_file = "test/data/labels/valid.csv"

    actual = mut.get_label_name(valid_label_file, input)

    assert actual == expected

@pytest.mark.parametrize("input,expected", [
    ("INVALID", 0),
    ("NONE", 1),
    ("FOOT_SWAP", 2),
    ("OUTSIDE_FLAG", 3),
    ("BACK_FLAG", 4),
    ("INSIDE_FLAG", 5),
    ("DROP_KNEE", 6),
    ("CROSS_MIDLINE", 7),
])
def test_name_to_value(input, expected):
    actual = mut.name_to_value(input)

    assert actual == expected

def test_name_to_value_with_invalid_name():
    with pytest.raises(ValueError):
        _ = mut.name_to_value("incorrect value")

@pytest.mark.parametrize("input,expected", [
    (0, "INVALID"),
    (1, "NONE"),
    (2, "FOOT_SWAP"),
    (3, "OUTSIDE_FLAG"),
    (4, "BACK_FLAG"),
    (5, "INSIDE_FLAG"),
    (6, "DROP_KNEE"),
    (7, "CROSS_MIDLINE")
])
def test_value_to_name(input, expected):
    actual = mut.value_to_name(input)

    assert actual == expected

@pytest.mark.parametrize("input,error", [
    (-1, IndexError), 
    (10, IndexError), 
    (inf, TypeError), 
    ("NONE", TypeError)
])
def test_value_to_name_with_invalid_input(input, error):
    with pytest.raises(error):
        _ = mut.value_to_name(input)

def test_iterate_valid_labels():
    expected_names = ["NONE",
        "FOOT_SWAP",
        "OUTSIDE_FLAG",
        "BACK_FLAG",
        "INSIDE_FLAG",
        "DROP_KNEE",
        "CROSS_MIDLINE"]

    iterator = mut.iterate_valid_labels()

    for expected in expected_names:
        assert next(iterator) == expected

def test_make_label_dirs_when_they_do_not_exist():
    root="./test/labelTestDir"

    try:
        mut.make_label_dirs(root)

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

        mut.make_label_dirs(root)

        assert not exists(join(root, "INVALID"))
        assert exists(join(root, "NONE"))
        assert exists(join(root, "FOOT_SWAP"))
        assert exists(join(root, "OUTSIDE_FLAG"))
        
    finally:
        rmtree(root)

def test_validate_all_labels():
    errors = mut.validate_all("data/labels")

    assert len(errors) == 0

def test_validate_invalid_csv():
    errors = mut.validate_label("test/data/labels/invalid.csv")

    assert len(errors) == 13