# leman/__init__.py
"""
Léman — Library for Exciton and Moiré Analysis in Nanostructures
================================================================

A research toolkit for TMDC optoelectronics and photonics, developed in the
LANES group at EPFL.

Submodules
----------
constants   Physical constants, material parameters.
loaders     Data loaders (DeviceGeometry, ACSpectralSweep, BTSpectralSweep, ...).
plotting    Publication-ready figure style and common plot types.
fitting     Spectral fitting (Lorentzian, Gaussian, multi-peak) and sparse TRPL lifetime fitting.
processing  Smoothing, normalisation, spectral conversions.
hdf5        Self-describing HDF5 storage for spectral sweeps.
converters  AttoCube CSV exports to TIFF (images) and HDF5 (sweeps).

Class-name prefixes
-------------------
AC  AttoCube cryogenic confocal (e.g. ACSpectralSweep, ACTRPLSweep, ACImg).
BT  BigTable LabRAM export format (e.g. BTSpectralSweep).

Quick start
-----------
>>> from leman import plotting, ACSpectralSweep, DeviceGeometry
>>>
>>> plotting.set_style("paper")
>>> geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)
>>> scan = ACSpectralSweep(
...     "myscan.csv", spectra_type="PL", sweep="electric_field", geometry=geom,
...     gates={"top": "V_A", "bottom": "V_B"},
... )
>>> print(scan)
>>> fig, ax, mesh = plotting.plot_spectral_map(scan, x_axis="energy")
>>> scan.to_hdf5("myscan.h5")
"""

__version__ = "0.1.0"
__author__  = "Brandon Loke"

from . import constants, fitting, plotting

from .loaders import (
    DeviceGeometry,
    ACSpectralSweep,
    ACTRPLSweep,
    ACPLVabScan,
    BTSpectralSweep,
    ACImgSweep,
    ACImg,
    ACSampleImg,
    ACLaserRefImg,
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
    "ACSpectralSweep",
    "ACTRPLSweep",
    "ACPLVabScan",
    "BTSpectralSweep",
    "ACImgSweep",
    "ACImg",
    "ACSampleImg",
    "ACLaserRefImg",
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