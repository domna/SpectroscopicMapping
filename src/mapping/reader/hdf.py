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
        with h5py.File(fname, "r") as h5_file:

            measurement_index = list(h5_file.keys())[0]
            x_axis = list(
                filter(lambda x: x != "Wavelength", h5_file[measurement_index].keys())
            )
            y_axis = list(h5_file[f"{measurement_index}/{x_axis[0]}"].keys())
            x_meshg, y_meshg = np.meshgrid(x_axis, y_axis, indexing="ij")

            if custom_wavelength is None:
                data = pd.DataFrame(
                    index=pd.MultiIndex.from_arrays(
                        [x_meshg.flatten(), y_meshg.flatten()], names=["x", "y"]
                    ),
                    columns=h5_file[f"{measurement_index}/Wavelength"],
                )
            else:
                data = pd.DataFrame(
                    index=pd.MultiIndex.from_arrays(
                        [x_meshg.flatten(), y_meshg.flatten()], names=["x", "y"]
                    ),
                    columns=custom_wavelength,
                )
            data.columns.name = "Wavelength"

            x_pos_sens = []
            y_pos_sens = []
            for x_axis, y_axis in data.index:
                dataset = h5_file[f"{measurement_index}/{x_axis}/{y_axis}/Spectrum"]
                x_pos_sens.append(dataset.attrs["Sensor X"])
                y_pos_sens.append(dataset.attrs["Sensor Y"])

                data.loc[x_axis, y_axis] = np.array(dataset)

            # Convert x, y to float
            data.index = data.index.set_levels(
                [data.index.levels[0].astype(float), data.index.levels[1].astype(float)]
            )
            self.index = data.index.copy()

            if interpolate:
                data.index = pd.MultiIndex.from_arrays(
                    [x_pos_sens, y_pos_sens], names=["x", "y"]
                )
                self.is_interp = True

        self.dataframe = data
