"""A reader for data in the csv file format"""
from dataclasses import dataclass
import numpy.typing as npt
import pandas as pd

from mapping.reader.reader import Reader


@dataclass
class CsvReader(Reader):
    """This is a class to read csv data"""

    def read(
        self, fname: str, interpolate: npt.ArrayLike, custom_wavelength: pd.Index
    ) -> None:
        self.dataframe = pd.read_csv(
            fname,
            sep=r"\s+",
            names=["x", "y", "Wavelength", "Intensity"],
            index_col=[0, 1, 2],
        )

        if interpolate:
            self.index = custom_wavelength
            self.is_interp = True
        else:
            self.index = self.dataframe.index.droplevel(level=2).drop_duplicates()
