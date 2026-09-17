# src/leman/reference/processors/tagarelli2023.py
#
# Tagarelli et al., Nature Photonics 2023
# "Electrical control of hybrid exciton transport in a van der Waals heterostructure"
# DOI: 10.1038/s41566-023-01198-w
# Zenodo: 10.5281/zenodo.7660668
#
# Figure 1d: PL spectra of homobilayer WSe2
# Nested sweep: electric field (mV/nm) × excitation power (µW)
#
# .mat structure:
#   ef300 and efm300: spectra(1340, 82), power(1, 82), power_rounded(1, 82)
#   ef0:              spectra(1340, 70),  power(1, 70)  [no power_rounded]
#
# Power axes are irregularly spaced and inconsistent across files.
# Default power is chosen as the index nearest to TARGET_POWER_UW in each file.
# Default electric field is E=0 (no applied field).
#
# Energy slicing: ef0 uses [290:-420], ef300/efm300 use [10:-10]
# per the original plotting code (authors' choice, removes noisy edges).

import io
import requests
import zipfile
import numpy as np
import scipy.io
import h5py
from pathlib import Path
from .processor import Processor

# Target power for default spectrum — nearest available index used per file
TARGET_POWER_UW = 40.0


class Tagarelli2023Processor(Processor):
    """
    Processor for Tagarelli et al., Nature Photonics 2023
    
    - Title: Electrical control of hybrid exciton transport in a van der Waals heterostructure
    - DOI: 10.1038/s41566-023-01198-w
    - Dataset: 10.5281/zenodo.7660668
    - Figure 1d: PL spectra of homobilayer WSe2 at increasing excitation power.
    - Data contains: Nested sweep: electric field (mV/nm) × excitation power (µW)
    """
    
    def _nearest_power_index(self, power_array: np.ndarray) -> int:
        return int(np.argmin(np.abs(power_array - TARGET_POWER_UW)))

    def run(self):
        print(f"  Fetching Figure 1.zip from Zenodo ({self.meta['dataset_doi']})...")
        z = self._fetch_zip(
            "https://zenodo.org/records/7660668/files/Figure%201.zip?download=1"
        )

        # Electric field value (mV/nm) → zip path
        # The authors' plotting code trims noisy edges:
        #   ef0: [290:-420], ef300/efm300: [10:-10]
        # Those slices are not applied here — full spectra are kept.
        ef_files = {
             0:   "Figure 1d/hb_0field.mat",
            +300: "Figure 1d/homobilayer_ef300_power_spectral.mat",
            -300: "Figure 1d/homobilayer_efm300_spectral.mat",
        }

        DEFAULT_FIELD = 0

        conditions = []
        for field_val, zip_path in ef_files.items():
            print(f"  Processing E={field_val:+d} mV/nm ...")

            with z.open(zip_path) as f:
                mat = scipy.io.loadmat(io.BytesIO(f.read()))

            energy  = mat["energy"].squeeze().astype(np.float64)
            spectra = mat["spectra"].astype(np.float64)          # (n_pixels, n_powers)
            power   = mat["power"].squeeze().astype(np.float64)  # (n_powers,)

            default_power_idx = self._nearest_power_index(power)

            conditions.append({
                "outer_value":    float(field_val),
                "is_default":     (field_val == DEFAULT_FIELD),
                "axis_values":    energy,
                "axis_name":      "energy",
                "axis_units":     "eV",
                "axis_label":     "Energy",
                "inner_values":   power,
                "inner_name":     "power",
                "inner_units":    "uW",
                "inner_label":    "Excitation power",
                "default_index":  default_power_idx,
                "intensity_2d":   spectra,
                "intensity_units": "counts",
            })

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            self._write_nested_sweep(hf, conditions, sweep_meta={
                "outer_name":    "electric_field",
                "outer_units":   "mV/nm",
                "outer_label":   "Electric field",
                "default_value": float(DEFAULT_FIELD),
            })

        print(f"  -> Saved to {self.out_path}")