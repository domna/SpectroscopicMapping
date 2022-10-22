"""A reader for data in the csv file format"""
import numpy.typing as npt
import pandas as pd

from mapping.reader.reader import Reader, MapData


class CsvReader(Reader):
    """This is a class to read csv data"""

    def read(
        self, fname: str, interpolate_to: npt.ArrayLike, custom_wavelength: pd.Index
    ) -> MapData:
        dataframe = pd.read_csv(
            fname,
            sep=r"\s+",
            names=["x", "y", "Wavelength", "Intensity"],
            index_col=[0, 1, 2],
        )

        if interpolate_to:
            return MapData(
                data=dataframe, index=custom_wavelength, is_interpolated=True
            )

        return MapData(
            data=dataframe,
            index=dataframe.index.droplevel(level=2).drop_duplicates(),
            is_interpolated=False,
        )
