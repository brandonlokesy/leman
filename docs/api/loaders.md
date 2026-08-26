# Loaders

Data loaders for device geometry and confocal scans — spectral sweeps over any
scanned parameter, time-resolved PL, real-space image sequences, and single spectra
or images — plus separate loaders for single Raman spectra and 2-D Raman maps from
a LabRAM-style ``.txt`` export, unrelated to the other instruments.

The sweep classes share private bases and differ only in what is genuinely
different about them, which is the whole reason they are separate:

| Class | Axis | Export layout | Input |
|---|---|---|---|
| `AttoCubeSpectralSweep` | wavelength / energy | `[Par, Wavelength, ExpROI1, ExpROI2]`, named by a header | one `.csv` or `.h5` |
| `BigTableSpectralSweep` | wavelength / energy | the same four fields with **no header**, and the signal written twice | one `.csv` |
| `AttoCubeTRPLSweep` | time (ns) | `[Par, Wavelength, Exp]` — the "Wavelength" column holds **time** | one `.csv`, a **directory**, or `.h5` |

The two spectral classes are siblings under a shared `_SpectralSweep`, so every
correction, both axes and every accessor behave identically; what differs is the
decoder and which parameter rows the instrument writes. A BigTable export carries
no labels at all, so its parameter row names are this package's rather than the
file's — see ``dev/instruments/big-table.md``.

A single decay is simply `n_sweeps == 1`, so that is not what divides them.
`energy = hc/t` is meaningless and divides by zero at `t = 0`, so the energy
machinery — `energy`, `energy_spectra`, `apply_jacobian` — does not exist on a
TRPL sweep, and neither does `spectra`: handing one to a spectral plot raises
rather than drawing time as if it were wavelength. Each class rejects the other's
files by name.

::: tmdc_optics_tools.loaders
