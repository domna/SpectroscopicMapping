"""An abstract reader class to serve as a template for specific readers"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Union
import numpy.typing as npt
import pandas as pd


@dataclass
class Reader:
    """This is a generic reader class"""

    index: pd.Index
    dataframe: pd.DataFrame
    is_interp: bool = False

    @abstractmethod
    def read(
        self,
        fname: str,
        interpolate: Union[str, npt.ArrayLike],
        custom_wavelength: pd.Index,
    ) -> None:
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
