# Mapping

A small tool to map three dimensional microscopy/spectroscopic images in jupyter

## Description

Installation::

    python setup.py install

Example usage in jupyter::

    from mapping.pl_mapping_hdf import Mapper
    m = Mapper(filename, central_wavelength, bandwidth_wavelength, interpolate=True/False)
    m.generate_interactive_plot()

You may use the file_helper function to read hdf5 files in the current directory::

    datafile = Mapper.file_helper(basepath='./')
    datafile

The selected is accesible by::

    datafile.result

You should see a 2D Image generated from your data now.
