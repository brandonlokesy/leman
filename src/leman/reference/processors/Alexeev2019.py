import io
import pickle
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from ...constants import HC_EV_NM

from .processor import Processor

class Alexeev2019ProcessorMoSe2(Processor):
    """ 
    Processor for Alexeev et al., Nature 2019.

    - Title: Resonantly hybridized excitons in moiré superlattices in van der Waals heterostructures
    - DOI: 10.1038/s41586-019-0986-9
    - Dataset: 10.1038/s41586-019-0986-9 (files on Nature)
    - Figure 2a: PL spectra of monolayer MoSe2.
    """
    def run(self):

        url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-0986-9/MediaObjects/41586_2019_986_MOESM4_ESM.xlsx"
        print(f"Fetching data from {url} from Nature.")
        
        df = pd.read_excel(url, sheet_name= "(a) Normalised PL", header = [0,1])

        energy = df[("Energy, eV" ,'Unnamed: 0_level_1')]
        spectra_MoSe2 = df['Normalised PL intensity', 'MoSe2']

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            self._write_single_spectrum(
                hf,
                energy.to_numpy(), "energy", "eV", "Energy",
                spectra_MoSe2.to_numpy(), "dimensionless",
            )

        print(f"  -> Saved to {self.out_path}")

class Alexeev2019ProcessorWS2(Processor):
    """
    Processor for Alexeev et al., Nature 2019.
    
    - Title: Resonantly hybridized excitons in moiré superlattices in van der Waals heterostructures
    - DOI: 10.1038/s41586-019-0986-9
    - Dataset: 10.1038/s41586-019-0986-9 (files on Nature)
    - Figure 2a: PL spectra of monolayer WS2.
    """
    def run(self):

        url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-0986-9/MediaObjects/41586_2019_986_MOESM2_ESM.xlsx"
        print(f"Fetching data from {url} from Nature. "  )
        
        df = pd.read_excel(url, sheet_name="(a) Full Spectra", header = [0,1])

        energy = df[("Energy,eV" ,'Unnamed: 0_level_1')]
        spectra_WS2 = df[('Intensity, cts/s', 'WS2')]

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            self._write_single_spectrum(
                hf,
                energy.to_numpy(), "energy", "eV", "Energy",
                spectra_WS2.to_numpy(), "counts/s",
            )

        print(f"  -> Saved to {self.out_path}")
