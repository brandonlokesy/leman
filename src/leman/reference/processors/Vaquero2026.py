import io
import numpy as np
import h5py
import pandas as pd
from .processor import Processor

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

        print(f"  Processing file...")

        with z.open(filepath) as f:
            df = pd.read_csv(io.BytesIO(f.read()), header=[0, 1])
            new_columns = []
            current_density = None

            for density, label in df.columns:
                if not density.startswith("Unnamed"):
                    current_density = density
                new_columns.append((current_density, label))

            df.columns = pd.MultiIndex.from_tuples(new_columns)

            density_strings = []
            for density, _ in df.columns:
                if density not in density_strings:
                    density_strings.append(density)

            densities = [self.parse_density(d) for d in density_strings]

            default_density_idx = int(np.argmin(
                np.abs(np.array(densities) - DEFAULT_EXCITON_DENSITY_CM2)
            ))

            all_counts = []
            energy = None
            for density_str in density_strings:
                if energy is None:
                    energy = df[density_str].iloc[:, 0].to_numpy()
                all_counts.append(df[density_str].iloc[:, 1].to_numpy())

            # (n_pixels, n_sweeps)
            intensity_2d = np.column_stack(all_counts)

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            self._write_1d_sweep(
                hf,
                axis_values=energy,
                axis_name="energy",
                axis_units="eV",
                axis_label="Energy",
                sweep_values=np.array(densities),
                sweep_name="exciton_density",
                sweep_units="cm^-2",
                sweep_label="Exciton density",
                default_index=default_density_idx,
                intensity_2d=intensity_2d,
                intensity_units="counts",
            )

        print(f"  -> Saved to {self.out_path}")