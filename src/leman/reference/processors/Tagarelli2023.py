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

HEADERS = {"User-Agent": "LANES-Tools/1.0"}

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

        with h5py.File(self.out_path, "w") as hf:

            self._write_metadata(hf)
            
            # --- Outer sweep: electric field ---
            ef_sweep = hf.create_group("electric_field")
            ef_sweep.attrs["parameter_name"] = "electric_field"
            ef_sweep.attrs["parameter_unit"] = "mV/nm"
            ef_sweep.attrs["default_value"]  = float(DEFAULT_FIELD)

            for field_val, zip_path in ef_files.items():
                print(f"  Processing E={field_val:+d} mV/nm ...")

                with z.open(zip_path) as f:
                    mat = scipy.io.loadmat(io.BytesIO(f.read()))

                energy        = mat["energy"].squeeze().astype(np.float64)
                spectra       = mat["spectra"].astype(np.float64)        # (1340, N)
                power         = mat["power"].squeeze().astype(np.float64)  # (N,)
                power_rounded = mat.get("power_rounded", None)
                if power_rounded is not None:
                    power_rounded = power_rounded.squeeze().astype(np.float64)

                default_power_idx = self._nearest_power_index(power)
                default_power_val = float(power[default_power_idx])

                # --- Electric field group ---
                label    = f"{field_val:+d}" if field_val != 0 else "0"
                ef_group = ef_sweep.create_group(label)
                ef_group.attrs["parameter_value"]  = float(field_val)
                ef_group.attrs["is_default"]        = (field_val == DEFAULT_FIELD)

                ef_group.create_dataset("energy", data=energy)
                ef_group.attrs["energy_unit"] = "eV"

                # --- Inner sweep: excitation power ---
                pw_sweep = ef_group.create_group("power")
                pw_sweep.attrs["parameter_name"]  = "excitation_power"
                pw_sweep.attrs["parameter_unit"]  = "uW"
                pw_sweep.attrs["default_value"]   = default_power_val
                pw_sweep.attrs["target_power_uW"] = TARGET_POWER_UW
                pw_sweep.attrs["spectroscopy"] = "PL"

                for i, pwr in enumerate(power):
                    spectrum = spectra[:, i]

                    # Use rounded power as label if available, else raw
                    # Include index as tiebreaker to prevent duplicate group names
                    if power_rounded is not None:
                        pwr_label = f"{power_rounded[i]:.1f}"
                        if pwr_label in pw_sweep:          # collision — append index to disambiguate
                            pwr_label = f"{power_rounded[i]:.1f}_{i}"
                    else:
                        pwr_label = f"{pwr:.4f}"
                        if pwr_label in pw_sweep:
                            pwr_label = f"{pwr:.6f}"

                    pw_group = pw_sweep.create_group(pwr_label)
                    pw_group.create_dataset("spectrum", data=spectrum)
                    pw_group.attrs["parameter_value"] = pwr
                    pw_group.attrs["power_index"]     = i
                    pw_group.attrs["spectrum_unit"]   = "counts"
                    pw_group.attrs["is_default"]      = (i == default_power_idx)

        print(f"  -> Saved to {self.out_path}")