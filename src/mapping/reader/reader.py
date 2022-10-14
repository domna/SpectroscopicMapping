"""An abstract reader class to serve as a template for specific readers"""
from abc import abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class Reader:
    """This is a generic reader class"""

    index: pd.Index
    is_interp: bool
    dataframe: pd.DataFrame

    @abstractmethod
    def read(self, fname: str, interpolate: str, custom_wavelength: pd.Index) -> None:
        """_summary_

        Args:
            fname (str): _description_
            interpolate (str): _description_
            custom_wavelength (npt.ArrayLike): _description_
        """
