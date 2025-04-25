import pytest
from random import seed

from os.path import dirname, join
from sys import path

this_dir = dirname( __file__ )
mymodule_dir = join( this_dir, '..', 'src' )
path.append( mymodule_dir )

from sampling.images import data_slice_factory

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