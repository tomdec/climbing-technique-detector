import pytest

from os.path import dirname, join
from sys import path

this_dir = dirname( __file__ )
mymodule_dir = join( this_dir, '..', 'src' )
path.append( mymodule_dir )

from labels import Technique, get_label

__location = "./data/labels/How to Flag - A Climbing Technique for Achieving Balance.csv"

@pytest.mark.parametrize("input,expected", [
        (100, Technique.INVALID),
        (500, Technique.NONE),
        (1500, Technique.NONE)
    ])
def test_get_labels(input, expected):
    actual = get_label(__location, input)

    assert actual == expected