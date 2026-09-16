import pandas as pd
import h5py
from .processor import Processor

class Louca2023Processor(Processor):
    """
    Processor for Louca et al., Nature Communications 2023.
    
    - Title: Interspecies exciton interactions lead to enhanced nonlinearity of dipolar excitons and polaritons in MoS2 homobilayers
    - DOI: 10.1038/s41467-023-39358-9
    - Dataset: 10.24435/materialscloud:d2-ta
    - Figure 1: T=4K. Reflectance spectra for homobilayer MoS2
    """
    def run(self):
        url = "https://archive.materialscloud.org/records/q0deg-ag137/files/Fig1c.xlsx?download=1"

        df = pd.read_csv(url)

        energy = df['Energy (eV)']
        spectra = df['RC']

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            hf.create_dataset('energy', data = energy)
            hf.create_dataset('spectra', data = spectra)