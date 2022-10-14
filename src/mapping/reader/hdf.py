"""A reader for data in the hdf file format"""
from dataclasses import dataclass
import numpy as np
import h5py
import pandas as pd

from mapping.reader.reader import Reader


@dataclass
class HdfReader(Reader):
    """This is a class to read hdf data"""

    def read(self, fname: str, interpolate: str, custom_wavelength: pd.Index) -> None:
        f = h5py.File(fname, "r")

        measurement_index = list(f.keys())[0]
        x = list(filter(lambda x: x != "Wavelength", f[measurement_index].keys()))
        y = list(f[f"{measurement_index}/{x[0]}"].keys())
        xm, ym = np.meshgrid(x, y, indexing="ij")

        if custom_wavelength is None:
            data = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [xm.flatten(), ym.flatten()], names=["x", "y"]
                ),
                columns=f[f"{measurement_index}/Wavelength"],
            )
        else:
            data = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [xm.flatten(), ym.flatten()], names=["x", "y"]
                ),
                columns=custom_wavelength,
            )
        data.columns.name = "Wavelength"

        sensX = []
        sensY = []
        for x, y in data.index:
            dataset = f[f"{measurement_index}/{x}/{y}/Spectrum"]
            sensX.append(dataset.attrs["Sensor X"])
            sensY.append(dataset.attrs["Sensor Y"])

            data.loc[x, y] = np.array(dataset)

        # Convert x, y to float
        data.index = data.index.set_levels(
            [data.index.levels[0].astype(float), data.index.levels[1].astype(float)]
        )
        self.index = data.index.copy()

        if interpolate:
            data.index = pd.MultiIndex.from_arrays([sensX, sensY], names=["x", "y"])
            self.is_interp = True

        f.close()

        self.dataframe = data
