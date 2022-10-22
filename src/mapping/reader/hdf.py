"""A reader for data in the hdf file format"""
from typing import Optional
from numpy.typing import ArrayLike
import numpy as np
import h5py
import pandas as pd

from mapping.reader.reader import Reader, MapData


def read_axes(h5_file: h5py.File, measurement_index: list) -> pd.MultiIndex:
    """Reads the axes from an hdf5 file into a pandas multiindex.

    Args:
        h5_file (h5py.File): The h5py file object to read from.
        measurement_index (list): The index of the current measurement.

    Raises:
        ValueError:
            The entries should be position groups in the hdf5 file.
            Otherwise this error is thrown.

    Returns:
        pd.MultiIndex: The multiindex containing (x, y) data value pairs.
    """

    x_axis_grp = h5_file[measurement_index]
    if not isinstance(x_axis_grp, h5py.Group):
        raise ValueError(f"Entry {measurement_index} should be a group.")
    x_axis = list(filter(lambda x: x != "Wavelength", x_axis_grp.keys()))

    y_axis_grp = h5_file[f"{measurement_index}/{x_axis[0]}"]
    if not isinstance(y_axis_grp, h5py.Group):
        raise ValueError(f"Entry {measurement_index}/{x_axis[0]} should be a group.")
    y_axis = list(y_axis_grp.keys())

    x_meshg, y_meshg = np.meshgrid(x_axis, y_axis, indexing="ij")

    idx = pd.MultiIndex.from_arrays(
        [x_meshg.flatten(), y_meshg.flatten()], names=["x", "y"]
    )

    return idx


class HdfReader(Reader):
    """This is a class to read hdf data"""

    def read(
        self,
        fname: str,
        interpolate_to: Optional[ArrayLike] = None,
        custom_wavelength: Optional[pd.Index] = None,
    ) -> MapData:
        with h5py.File(fname, "r") as h5_file:

            measurement_index = list(h5_file.keys())[0]
            idx = read_axes(h5_file, measurement_index)

            if custom_wavelength is None:
                wavelength_grp = h5_file[f"{measurement_index}/Wavelength"]
                if not isinstance(wavelength_grp, h5py.Dataset):
                    raise ValueError(
                        f"Entry {measurement_index}/Wavelength should be a dataset."
                    )

                data = pd.DataFrame(
                    index=idx,
                    columns=np.array(wavelength_grp[:]),
                )
            else:
                data = pd.DataFrame(
                    index=idx,
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

            if interpolate_to is not None:
                raise NotImplementedError(
                    "Axes interpolation is not implemented for hdf files."
                )

            data.index = pd.MultiIndex.from_arrays(
                [x_pos_sens, y_pos_sens], names=["x", "y"]
            )
            idx = idx.set_levels(
                [
                    idx.levels[0].astype(float),
                    idx.levels[1].astype(float),
                ]
            )
            return MapData(data=data, index=idx, is_interpolated=False)
