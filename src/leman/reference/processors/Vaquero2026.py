import io
import requests
import zipfile
import numpy as np
import scipy.io
import h5py
from pathlib import Path
import pandas as pd
from .processor import Processor

HEADERS = {"User-Agent": "LANES-Tools/1.0"}

class Vaquero2026Processor(Processor):
    """
    Processor for Vaquero et al., Nano Letters 2026.
    
    - Title: Valley-Controlled Many-Body Exciton Interactions in Monolayer WSe2 Phototransistors
    - DOI: 10.1021/acs.nanolett.6c01091
    - Dataset: 10.48550/arXiv.2604.08382
    - Figure 2a: PL spectra of monolayer WSe2 with linearly polarized light sweept by exciton densities.
    """

    def parse_density(self,s):
        base, power = s.split("x10")
        return float(base) * 10**int(power)
    
    def run(self):
        print(f"  Fetching Figure 2.zip from Zenodo ({self.meta['dataset_doi']})...")
        z = self._fetch_zip(
            "https://zenodo.org/records/19887546/files/Data.zip?download=1"
        )

        filepath = "Zenodo_repository/figure_2/panel_a/fig_2a.csv"
        DEFAULT_EXCITON_DENSITY_CM2 = 1e12  # cm^-2, nearest available index used per file

        with h5py.File(self.out_path, "w") as hf:
            # --- File-level metadata ---
            self._write_metadata(hf)


            nI_sweep = hf.create_group("exciton_density")
            nI_sweep.attrs["parameter_name"] = "exciton_density"
            nI_sweep.attrs["parameter_unit"] = "cm^-2" 
            nI_sweep.attrs["default_value"] = float(DEFAULT_EXCITON_DENSITY_CM2)

            print(f"Processing file...")
            
            with z.open(filepath) as f:
                df = pd.read_csv(io.BytesIO(f.read()), header=[0, 1])
                new_columns = []
                current_density = None

                for density, label in df.columns:
                    if not density.startswith("Unnamed"):
                        current_density = density
                    new_columns.append((current_density, label))

                df.columns = pd.MultiIndex.from_tuples(new_columns)

                density_strings = []   # ordered unique strings, as they appear left-to-right
                for density, _ in df.columns:
                    if density not in density_strings:
                        density_strings.append(density)

                densities = [self.parse_density(d) for d in density_strings]
                labels    = [f"{d/1e12:g}e12" for d in densities]

                default_density_idx = np.argmin(
                    np.abs(np.array(densities) - DEFAULT_EXCITON_DENSITY_CM2)
                )

                for i, density_str in enumerate(density_strings):   # ← iterate the ordered list
                    energy = df[density_str].iloc[:, 0].to_numpy()
                    counts = df[density_str].iloc[:, 1].to_numpy()

                    grp = nI_sweep.create_group(labels[i])

                    grp.attrs["parameter_value"] = densities[i]
                    grp.attrs["density_index"] = i
                    grp.attrs["spectrum_unit"] = "counts"
                    grp.attrs["is_default"] = (i == default_density_idx)

                    if i == 0:
                        nI_sweep.create_dataset("energy", data=energy)

                    grp.create_dataset("counts", data=counts)
        
        print(f"  -> Saved to {self.out_path}")