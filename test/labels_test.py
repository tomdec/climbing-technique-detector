from math import inf
import pytest
from os import makedirs
from os.path import exists, join
from shutil import rmtree

import src.labels as mut


def test_get_dataset_name():
    actual = mut.get_dataset_name()

    assert actual == "test-labels"


@pytest.mark.parametrize(
    "input,expected",
    [
        ("INVALID", 0),
        ("LABEL1", 1),
        ("LABEL2", 2),
        ("LABEL3", 3),
        ("LABEL4", 4),
        ("LABEL5", 5),
        ("LABEL6", 6),
        ("LABEL7", 7),
    ],
)
def test_name_to_value(input, expected):
    actual = mut.name_to_value(input)

    assert actual == expected


def test_name_to_value_with_invalid_name():
    with pytest.raises(ValueError):
        _ = mut.name_to_value("incorrect value")


@pytest.mark.parametrize(
    "input,expected",
    [
        (0, "INVALID"),
        (1, "LABEL1"),
        (2, "LABEL2"),
        (3, "LABEL3"),
        (4, "LABEL4"),
        (5, "LABEL5"),
        (6, "LABEL6"),
        (7, "LABEL7"),
    ],
)
def test_value_to_name(input, expected):
    actual = mut.value_to_name(input)

    assert actual == expected


@pytest.mark.parametrize(
    "input,error",
    [(-1, IndexError), (10, IndexError), (inf, TypeError), ("NONE", TypeError)],
)
def test_value_to_name_with_invalid_input(input, error):
    with pytest.raises(error):
        _ = mut.value_to_name(input)


def test_iterate_valid_labels():
    expected_names = [
        "LABEL1",
        "LABEL2",
        "LABEL3",
        "LABEL4",
        "LABEL5",
        "LABEL6",
        "LABEL7",
    ]

    iterator = mut.iterate_valid_labels()

    for expected in expected_names:
        assert next(iterator) == expected


def test_make_label_dirs_when_they_do_not_exist():
    root = "./test/labelTestDir"

    try:
        mut.make_label_dirs(root)

        assert not exists(join(root, "INVALID"))
        assert exists(join(root, "LABEL1"))
        assert exists(join(root, "LABEL2"))
        assert exists(join(root, "LABEL3"))

    finally:
        rmtree(root)


def test_make_label_dirs_when_they_do_exist():
    root = "./test/labelTestDir"

    try:
        makedirs(join(root, "LABEL1"))

        mut.make_label_dirs(root)

        assert not exists(join(root, "INVALID"))
        assert exists(join(root, "LABEL1"))
        assert exists(join(root, "LABEL2"))
        assert exists(join(root, "LABEL3"))

    finally:
        rmtree(root)


def test_validate_all_labels():
    errors = mut.validate_all("data/labels")

    assert len(errors) == 0


def test_validate_invalid_csv():
    errors = mut.validate_label("test/data/labels/invalid.csv")

    assert len(errors) == 13
