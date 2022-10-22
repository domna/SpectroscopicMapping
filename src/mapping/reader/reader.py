"""An abstract reader class to serve as a template for specific readers"""
from typing import Optional
from abc import abstractmethod
from dataclasses import dataclass
from numpy.typing import ArrayLike
import pandas as pd


@dataclass
class MapData:
    """Represents a mapping data object"""

    is_interpolated: bool
    index: pd.Index
    data: pd.DataFrame


class Reader:
    """This is a generic reader class"""

    @abstractmethod
    def read(
        self,
        fname: str,
        interpolate_to: Optional[ArrayLike],
        custom_wavelength: Optional[pd.Index],
    ) -> MapData:
        """Reads a datafile into a multiindex dataframe

        Args:
            fname (str): The filename of the datafile
            interpolate (Union[str, npt.ArrayLike]):
                This parameter is either a boolean flag indicating if the data
                shall be interpolated from the sensor positions in the datafile
                or an array to interpolate to.
            custom_wavelength (npt.ArrayLike):
                A custom wavelength to override the existing wavelength axis.
        """
