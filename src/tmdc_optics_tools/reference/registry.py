from .processors import *

        # "material" : "",
        # "source"  : "",
        # "dataset_doi": "",
        # "title" : "",
        # "doi" : "",
        # "about" : "",
        # "spectroscopy": "",
        # "processor" : "",

REGISTRY = [
    {
        "material":  "2L_WSe2",
        "source":    "Tagarelli2023",
        "dataset_doi": "10.5281/zenodo.7660668",
        "title":     "Electrical control of hybrid exciton transport in a van der Waals heterostructure",
        "doi" : "10.1038/s41566-023-01198-w",
        "about" : "Figure 1d: PL spectra of homobilayer WSe2. Nested sweep: electric field (mV/nm) × excitation power (µW).",
        "spectroscopy": "PL",
        "processor": Tagarelli2023Processor,
    },

    {
        "material" : "1L_WSe2",
        "source"  : "Dijkstra2025",
        "dataset_doi": "10.14459/2025mp1793118",
        "title" : "Ten-valley excitonic complexes in charge-tunable monolayer WSe2",
        "doi" : "10.1038/s41467-025-65731-x",
        "about": "Figure 1a: gate-dependent reflectance contrast (ΔR/R₀) of monolayer WSe2. Sweep: gate voltage (V).",        "spectroscopy": "R",
        "processor" : Dijkstra2025Processor,
    },

    {
        "material" : "1L_WSe2",
        "source"  : "Vaquero2026",
        "dataset_doi": "10.48550/arXiv.2604.08382",
        "title" : "Valley-Controlled Many-Body Exciton Interactions in Monolayer WSe2 Phototransistors",
        "doi" : "10.1021/acs.nanolett.6c01091",
        "about" : "Figure 2a: PL spectra of monolayer WSe2 with linearly polarized light sweept by exciton densities.",
        "spectroscopy": "PL",
        "processor" : Vaquero2026Processor,
    },

    {
        "material" : "1L_WS2",
        "source"  : "Alexeev2019",
        "dataset_doi": "10.1038/s41586-019-0986-9",
        "title" : "Resonantly hybridized excitons in moiré superlattices in van der Waals heterostructures",
        "doi" : "10.1038/s41586-019-0986-9",
        "about" : "",
        "spectroscopy": "PL",
        "processor" : Alexeev2019ProcessorWS2,
    },

    {
        "material" : "1L_MoSe2",
        "source"  : "Alexeev2019",
        "dataset_doi": "10.1038/s41586-019-0986-9",
        "title" : "Resonantly hybridized excitons in moiré superlattices in van der Waals heterostructures",
        "doi" : "10.1038/s41586-019-0986-9",
        "about" : "Fig. 4. T=10K",
        "spectroscopy": "PL",
        "processor" : Alexeev2019ProcessorMoSe2,
    },

    {
        "material" : "1L_WSe2",
        "source"  : "Lin2024",
        "dataset_doi": "10.5281/zenodo.13629283",
        "title" : "Moiré-engineered light-matter interactions in MoS2/WSe2 heterobilayers at room temperature",
        "doi" : "10.1038/s41467-024-53083-x",
        "about" : "Fig. 2. T=300K",
        "spectroscopy": "R",
        "processor" : Lin2024ProcessorWSe2,
    },

    {
        "material" : "1L_MoS2",
        "source"  : "Lin2024",
        "dataset_doi": "10.5281/zenodo.13629283",
        "title" : "Moiré-engineered light-matter interactions in MoS2/WSe2 heterobilayers at room temperature",
        "doi" : "10.1038/s41467-024-53083-x",
        "about" : "Fig. 2. T=300K",
        "spectroscopy": "R",
        "processor" : Lin2024ProcessorMoS2,
    },

    {
        "material" : "1L_MoS2",
        "source"  : "Louca2023",
        "dataset_doi": "10.24435/materialscloud:d2-ta",
        "title" : "Interspecies exciton interactions lead to enhanced nonlinearity of dipolar excitons and polaritons in MoS2 bilayers",
        "doi" : "10.1038/s41467-023-39358-9",
        "about" : "Fig. 1. T=4K",
        "spectroscopy": "R",
        "processor" : Louca2023Processor,
    },

    # {
    #     "material" : "MoSe2_monolayer",
    #     "source"  : "Shimazaki2020",
    #     "dataset_doi": "10.3929/ethz-b-000399579",
    #     "title" : "Strongly correlated electrons and hybrid excitons in a moiré heterostructure",
    #     "doi" : "10.1038/s41586-020-2191-2",

    # }
]

if __name__ == "__main__":
    from pathlib import Path
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    for entry in REGISTRY:
        print(f"Processing {entry['source']} ({entry['material']})...")
        proc = entry["processor"](entry, out_dir)
        proc.run()
        print(f"  -> Done\n")