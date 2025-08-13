import pytest

from src.sampling.dataframe import __label_encoding

testdata = [
    ("NONE", 1),
    ("FOOT_SWAP", 2),
    ("OUTSIDE_FLAG", 3),
    ("BACK_FLAG", 4),
    ("INSIDE_FLAG", 5),
    ("DROP_KNEE", 7),
    ("CROSS_MIDLINE", 8),
]

@pytest.mark.parametrize("label,expected", testdata)
def test_label_encoding(label, expected):

    actual = __label_encoding(label)

    assert actual == expected