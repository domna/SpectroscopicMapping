# -*- coding: utf-8 -*-
"""
    Setup file for SpectroscopicMapping.
"""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup

try:
    require("setuptools>=58.1.0")
except VersionConflict:
    print("Error: version of setuptools is too old (<58.1.0)!")
    sys.exit(1)


if __name__ == "__main__":
    setup()
