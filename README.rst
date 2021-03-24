=======
Mapping
=======


A small tool to map three dimensional microscopy/spectroscopic images in jupyter

Description
===========
Installation::

    python setup.py install

Example usage in jupyter::

    from mapping.pl_mapping_hdf import Mapper
    m = Mapper(filename, central_wavelength, bandwidth_wavelength, interpolation=True/False)
    m.generate_interactive_plot()

You should see a 2D Image generated from your data now.
