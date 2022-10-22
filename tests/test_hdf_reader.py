"""Tests the different readers"""
import pytest
from mapping.reader.reader import MapData
from mapping.reader.hdf import HdfReader


@pytest.fixture()
def map_data(request):
    """Returns the map data from the h5 testfile"""
    return HdfReader().read(request.fspath.join("../data/2022-10-11T09-04.h5"))


# pylint: disable=redefined-outer-name
def test_h5_file_read_wo_error(map_data):
    """Check if an h5 file is read without errors"""
    assert isinstance(map_data, MapData)


# pylint: disable=redefined-outer-name
def test_correct_set_of_defaults(map_data):
    """Check if the mapdata object has correct defaults"""
    assert not map_data.is_interpolated
