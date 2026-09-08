# tmdc_optics_tools/__init__.py
"""
tmdc_optics_tools
==================
A personal research toolkit for TMD optoelectronics and photonics.

Submodules
----------
constants   Physical constants, material parameters.
loaders     Data loaders (DeviceGeometry, AttoCubeSpectralSweep).
plotting    Publication-ready figure style and common plot types.
fitting     Spectral fitting (Lorentzian, Gaussian, multi-peak) and sparse TRPL lifetime fitting.
processing  Smoothing, normalisation, spectral conversions.
hdf5        Self-describing HDF5 storage for spectral sweeps.
converters  AttoCube CSV exports to TIFF (images) and HDF5 (sweeps).

Quick start
-----------
>>> from tmdc_optics_tools.loaders import DeviceGeometry, AttoCubeSpectralSweep
>>> from tmdc_optics_tools import plotting
>>>
>>> plotting.set_style("paper")
>>> geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
>>> scan = AttoCubeSpectralSweep(
...     "myscan.csv", spectra_type="PL", sweep="electric_field", geometry=geom,
...     gates={"top": "V_A", "bottom": "V_B"},   # which channel reached which gate
... )
>>> print(scan)
>>> fig, ax, mesh = plotting.plot_spectral_map(scan, x_axis="energy")
>>> scan.to_hdf5("myscan.h5")          # metadata travels with the data
"""

__version__ = "0.1.0"
__author__  = "Brandon Loke"

from . import constants, fitting, plotting

from .loaders import (
    DeviceGeometry,
    AttoCubeSpectralSweep,
    AttoCubeTRPLSweep,
    AttoCubePLVabScan,
    BigTableSpectralSweep,
    AttoCubePLScanRealSpace,
    StackLayer,
    SingleSpectrum,
    SingleImage,
    RamanSpectrum,
    RamanMap,
)
from . import converters, hdf5, processing
from .converters import (
    convert_image_csv_to_tiff,
    convert_image_dir_to_tiff_stack,
    convert_spectral_csv_to_hdf5,
    convert_trpl_dir_to_hdf5,
)

__all__ = [
    "StackLayer",
    "DeviceGeometry",
    "AttoCubeSpectralSweep",
    "AttoCubeTRPLSweep",
    "AttoCubePLVabScan",
    "BigTableSpectralSweep",
    "AttoCubePLScanRealSpace",
    "SingleSpectrum",
    "SingleImage",
    "RamanSpectrum",
    "RamanMap",
    "convert_image_csv_to_tiff",
    "convert_image_dir_to_tiff_stack",
    "convert_spectral_csv_to_hdf5",
    "convert_trpl_dir_to_hdf5",
    "constants",
    "converters",
    "hdf5",
    "plotting",
    "fitting",
    "processing",
]