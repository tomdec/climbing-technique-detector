import pytest

from os.path import dirname, join
from sys import path

this_dir = dirname( __file__ )
mymodule_dir = join( this_dir, '..', 'src' )
path.append( mymodule_dir )

from common import get_filename

@pytest.mark.parametrize("input,expected", [
    ("file.ext", "file"),
    ("path/to/file.txt", "file"),
    ("path/to/file", "file"),
    ("", "")
])
def test_get_filename(input, expected):
    actual = get_filename(input)

    assert actual == expected