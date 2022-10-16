# -*- coding: utf-8 -*-
"""
    Setup file for SpectroscopicMapping.
"""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup

try:
    require("setuptools>=45")
except VersionConflict:
    print("Error: version of setuptools is too old (<45)!")
    sys.exit(1)


if __name__ == "__main__":
    setup()
