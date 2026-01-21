import pytest

from os.path import dirname, join
from sys import path

this_dir = dirname(__file__)
mymodule_dir = join(this_dir, "..", "src")
path.append(mymodule_dir)

from src.common.helpers import (
    get_filename,
    get_current_train_run,
    get_current_test_run,
    get_next_train_run,
    get_next_test_run,
)


@pytest.mark.parametrize(
    "input,expected",
    [
        ("file.ext", "file"),
        ("path/to/file.txt", "file"),
        ("path/to/file", "file"),
        ("", ""),
    ],
)
def test_get_filename(input, expected):
    actual = get_filename(input)

    assert actual == expected


@pytest.mark.parametrize(
    "input,expected",
    [
        ("test/data/runs/hpe_dnn/test_model", "train5"),
        ("test/data/runs/sota/test_model", "train5"),
    ],
)
def test_get_current_train_run(input, expected):
    actual = get_current_train_run(input)

    assert actual == expected


@pytest.mark.parametrize(
    "input,expected",
    [
        ("test/data/runs/hpe_dnn/test_model", "test1"),
        ("test/data/runs/sota/test_model", "test2"),
    ],
)
def test_get_current_test_run(input, expected):
    actual = get_current_test_run(input)

    assert actual == expected


@pytest.mark.parametrize(
    "input,expected",
    [
        ("test/data/runs/hpe_dnn/test_model", "train6"),
        ("test/data/runs/sota/test_model", "train6"),
    ],
)
def test_get_next_train_run(input, expected):
    actual = get_next_train_run(input)

    assert actual == expected


@pytest.mark.parametrize(
    "input,expected",
    [
        ("test/data/runs/hpe_dnn/test_model", "test2"),
        ("test/data/runs/sota/test_model", "test3"),
    ],
)
def test_get_next_test_run(input, expected):
    actual = get_next_test_run(input)

    assert actual == expected
