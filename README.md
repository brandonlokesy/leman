# LÉMAN — Library for Exciton and Moiré Analysis in Nanostructures

A Python toolkit for TMDC optoelectronics and photonics measurements, developed
in the LANES group at EPFL. Covers data loading from AttoCube cryogenic confocal
and LabRAM setups, device geometry modelling, spectral processing, peak fitting,
DC Stark shift and dipole length extraction, time-resolved PL, real-space
imaging, Raman spectroscopy, and publication-ready plotting and animation.

The Python package name is `leman` (no accent).

📖 **Documentation:** https://brandonlokesy.github.io/leman/

### Class-name prefixes

Loader classes are prefixed by the instrument or export format they read:

| Prefix | Stands for | Instrument / format |
|--------|------------|---------------------|
| **AC** | AttoCube   | AttoCube cryogenic confocal |
| **BT** | BigTable   | LabRAM BigTable CSV export |

---

## Contents

- [Installation](#installation)
- [Package structure](#package-structure)
- [Key workflows](#key-workflows)
  - [1. Define device geometry](#1-define-device-geometry)
  - [2. Load a gate-dependent PL scan](#2-load-a-gate-dependent-pl-scan)
  - [3. Plot a spectral map](#3-plot-a-spectral-map)
  - [4. Inspect a single spectrum](#4-inspect-a-single-spectrum)
  - [5. Fit a peak across a sweep](#5-fit-a-peak-across-a-sweep)
  - [6. Extract the excitonic dipole length](#6-extract-the-excitonic-dipole-length)
  - [7. Real-space PL imaging](#7-real-space-pl-imaging)
  - [8. Check for dielectric breakdown](#8-check-for-dielectric-breakdown)
- [Module reference](#module-reference)

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/brandonlokesy/leman.git
cd leman
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-image`, `ffmpeg` (recommended for saving .mp4 files)

Optional (diverging colormaps used in plots):
```bash
pip install cmcrameri
```

---

## Package structure

```
leman/
├── constants.py   # Physical constants, material parameters (ε, thickness, exciton energies)
├── loaders.py     # DeviceGeometry, ACPLVabScan, ACImgSweep, image classes
├── processing.py  # Normalisation, smoothing, background subtraction, spectral conversions
├── fitting.py     # Lorentzian/Gaussian fitting, multi-peak fitting, dipole length extraction
└── plotting.py    # PL maps, spectrum plots, real-space animations, Stark shift plots
```

---

## Key workflows

### 1. Define device geometry

The `DeviceGeometry` class models the dielectric stack and computes the displacement field from gate voltages. It is required for converting raw gate voltages into a physical field axis.

**Simple monolayer:**
```python
from leman.loaders import DeviceGeometry

geom = DeviceGeometry.from_single(
    tmdc         = "WS2",
    d_hbn_top    = 30,   # nm
    d_hbn_bottom = 50,   # nm
)
```

**Heterobilayer (e.g. MoSe2/WSe2):**
```python
from leman.loaders import DeviceGeometry, StackLayer

geom = DeviceGeometry(
    tmdc_stack   = [StackLayer("MoSe2"), StackLayer("WSe2")],
    d_hbn_top    = 30,
    d_hbn_bottom = 50,
    label        = "hBN/MoSe2/WSe2/hBN",
)
```

`StackLayer` looks up monolayer thickness and dielectric constant from `constants.py` automatically; override with `d_monolayer` and `eps` if needed. Supported materials out of the box: `WS2`, `WSe2`, `MoSe2`, `MoS2`.

The dielectric constant of the TMDC layers alone (excluding hBN) can be called with `DeviceGeometry.eps_2d`, the same quantity for the whole gate stack (hBN included) with `DeviceGeometry.eps_stack`, the corresponding thicknesses in nm with `DeviceGeometry.d_2d` and `DeviceGeometry.d_stack`, and the stack description with `DeviceGeometry.stack_label`. The naming is consistent throughout: the `eps_`/`d_` prefix is the quantity, the `_2d`/`_stack` suffix is the scope — TMDC layers alone, or the whole gate stack including hBN.

---

### 2. Load a gate-dependent PL scan

These scans are a map of the PL spectra with respect to the voltages A and B applied to the sample. Typically, we apply these voltages to the top and bottom gate to tune the electric field applied to the heterostructure. When the sample geometry is given, the electric field applied to the heterostructure is calculated.

Which channel reached which gate depends on how the sample was connected, and no export file records it — so `ACSpectralSweep` requires it as `gates={"top": "V_A", "bottom": "V_B"}` and refuses to produce `v_top`, `v_bot` or `ef` without it. Transposing the two mirrors the field axis and flips the sign of any dipole extracted from it. The deprecated `ACPLVabScan` below assumes `V_A`→top / `V_B`→bottom so that older scripts keep running; check it against your wiring before trusting a sign.

```python
from leman.loaders import ACPLVabScan

scan = ACPLVabScan(
    path     = "PL_dual_gate_sweep_26_05_15_11_42_07_iter_0.csv",
    geometry = geom,   # optional — enables displacement field axis
)
print(scan)
# ACSpectralSweep — 101 sweeps × 1340 pixels
#   λ range : 850.0 – 1000.0 nm  (1.240 – 1.459 eV)
#   V_top   : -5.0 → 5.0 V
#   E_F     : -12.3 → 12.3 mV/nm
```

Key attributes after loading:

| Attribute | Description |
|---|---|
| `scan.wavelength` | Spectrometer wavelength axis (nm) |
| `scan.energy` | Photon energy axis (eV) |
| `scan.spectra` | The file's own counts, shape `(n_pixels, n_sweeps)` |
| `scan.energy_spectra` | The same counts on the energy axis |
| `scan.best_spectra`, `scan.best_energy_spectra` | Most-corrected array available on each axis — cosmic-ray repaired and background-subtracted where those were declared at load time |
| `scan.v_top`, `scan.v_bot` | Gate voltages (V) |
| `scan.ef` | Displacement field (mV/nm), `None` if no geometry |
| `scan.power` | Excitation power (µW) |
| `scan.i_top`, `scan.i_bot` | Gate leakage currents (nA) |
| `scan.i_channel` | Current into the TMDC contact (nA) — transport, not leakage |

---

### 3. Plot a spectral map

```python
from leman import plotting

plotting.set_style("paper")   # or "talk", "poster". Optional

fig, ax, mesh = plotting.plot_spectral_map(
    scan,
    x_axis        = "energy",      # or "wavelength"
    cmap          = "magma",
    median_kernel = 1,             # 1 = no filter. Above 1 medians a square
                                   # footprint, across sweeps as well as pixels
)
```

The y-axis is whatever was declared as `sweep=` at load time — displacement field, gate voltage, excitation power, piezo position, or the sweep index if nothing was declared.


---

### 4. Inspect a single spectrum

Name the point the way you took the measurement — a coordinate on the sweep axis, in its own units:

```python
fig, ax, line, ax_twin = plotting.plot_spectrum(
    scan,
    value     = 2.5,               # sweep axis units: V, mV/nm, µW, …
    x_axis    = "energy",
    normalize = True,
)
ax.set_xlim(1.30, 1.45)
```

The nearest sweep point is used, and a request that lands far from any real point warns rather than failing silently. A coordinate matching several points is refused — declare the nest and address it below, or pick the point by position.

Integer positions work the same way, and a 2-D sweep is addressed on both of its axes:

```python
plotting.plot_spectrum(scan, index = 50)                      # by position
plotting.plot_spectrum(scan, index = -1)                      # last point

plotting.plot_spectrum(scan, value = 15.0, axis = "top_voltage")  # by another quantity

# a nest declared with fast_sweep= / slow_sweep= at load time
plotting.plot_spectrum(scan, fast = 2.5, slow = 100.0)        # by coordinate
plotting.plot_spectrum(scan, index_fast = 3, index_slow = 1)  # by position
```

**Every selector is keyword-only** — `plot_spectrum(scan, 50)` is a `TypeError`, not a guess. A bare number could be a coordinate or a position, and on a sweep whose coordinates span the same range as its positions (a power sweep in µW, say) neither the result nor a warning would tell you which was taken.

---

### 5. Fit a peak across a sweep

Fit a Lorentzian (or Gaussian) to a chosen spectral window at every sweep point:

Background subtraction is **not** an argument here. It is declared once when the scan
is loaded, with `bg_region_nm=` or `bg_region_eV=` on the loader, and `fit_scan_peak`
then fits the most-corrected array the scan holds — so a declared pedestal is already
gone, on either axis. What the fit adds is `baseline`, a flat or sloping offset fitted
*alongside* the peak so an un-subtracted pedestal cannot inflate the amplitude and width:

```python
from leman import fitting

results = fitting.fit_scan_peak(
    scan,
    x_axis   = "energy",
    x_range  = (1.30, 1.42),   # eV — zoom into your exciton
    model    = "lorentzian",
    baseline = "constant",     # the default: a flat offset fitted with the peak
)

# results is a list of FitResult, one per sweep
print(results[50])
# FitResult [lorentzian]  R²=0.9971
#   amplitude    = 4312.1 ± 38
#   center       = 1.3847 ± 0.00021
#   fwhm         = 0.00831 ± 0.00019
```

For a single spectrum:
```python
x = scan.energy
y = scan.best_energy_spectra[:, 50]   # corrected where corrections were declared

result = fitting.fit_lorentzian(x, y, p0=(y.max(), 1.385, 0.01))
```

Multi-peak fitting:
```python
result = fitting.fit_multi_lorentzian(
    x, y,
    n_peaks = 2,
    p0      = [(4000, 1.385, 0.01), (1500, 1.40, 0.01)],
)
```

---

### 6. Extract the excitonic dipole length

The DC Stark shift gives the out-of-plane dipole length of an interlayer exciton. `extract_dipole_length` fits a Lorentzian at every sweep point and performs a weighted linear fit E(F) = slope · F + intercept to extract d = |slope| × 1000 nm.

```python
result = fitting.extract_dipole_length(
    scan,                          # background declared at load, as in §5
    x_range      = (1.30, 1.42),   # eV — spectral window for peak fitting
    model        = "lorentzian",
    ef_range     = (-8, 8),        # mV/nm — restrict to linear Stark regime
    baseline     = "constant",     # the default, passed to every peak fit
)
print(result)
# DipoleResult
#   Dipole length : 0.5821 ± 0.0034 nm  (5.82 Å)
#   Slope dE/dF   : -5.821e-04 ± 3.4e-06 eV/(mV/nm)
#   Intercept E₀  : 1.3849 ± 0.0002 eV
#   R²            : 0.9963
#   Sweep points  : 87 / 101 converged

# Plot it
fig, ax = plotting.plot_stark_shift(result, show_fit=True)
```

---

### 7. Real-space PL imaging

Load a folder of real-space PL image CSVs (one per gate voltage step) and animate them:

```python
from leman.loaders import ACImgSweep

rs_scan = ACImgSweep(
    path   = "./images/",
    prefix = "PLdualgatesweep_iter_",
)

fig, anim = plotting.animate_real_space_PL_map(
    rs_scan,
    var_array = scan.ef,
    var_label = "E-field",
    units     = r"mV nm$^{-1}$",
    title     = "Device A — gate sweep",
)

# Save as gif or mp4
anim.save("gate_sweep.gif", fps=5)
```

Optionally annotate the laser spot position using a reference image:

```python
from leman.loaders import ACLaserRefImg

laser_ref = ACLaserRefImg("laser_ref.csv")
print(laser_ref)
# Center: (63.4, 71.2) px  |  1/e² Radius: 8.3 px

rs_scan = ACImgSweep(..., laser_ref=laser_ref)
```

---

### 8. Check for dielectric breakdown

Plot gate leakage currents and excitation power together to verify the device was not in breakdown during a sweep:

```python
plot = plotting.plot_current(scan)          # members: fig, ax_left, ax_right, lines
plot.ax_left.set_ylim(-5, 5)                # current axis, in nA
```

---

## Module reference

### `constants`
Literature values for hBN, WS2, WSe2, MoSe2, MoS2: out-of-plane dielectric constants, monolayer thicknesses, approximate exciton energies and binding energies, and bulk/bilayer bandgaps.

### `loaders`
| Class | Purpose |
|---|---|
| `StackLayer` | One material slab in a vdW stack |
| `DeviceGeometry` | Dielectric stack model; computes ε_eff, displacement field |
| `ACSpectralSweep` | Spectral sweep from AttoCube confocal CSV or HDF5 |
| `BTSpectralSweep` | Spectral sweep from LabRAM BigTable CSV |
| `ACTRPLSweep` | Time-resolved PL sweep from AttoCube confocal |
| `ACImgSweep` | Sequence of real-space CCD image CSVs |
| `ACImg` | Single CCD frame (e.g. PL image) |
| `ACSampleImg` | White-light sample reference image |
| `ACLaserRefImg` | Laser spot image with fitted 1/e² radius |
| `SingleSpectrum` | A single two-row CSV spectrum |
| `SingleImage` | A single numeric-grid CSV image |
| `RamanSpectrum` | Single Raman spectrum from LabRAM |
| `RamanMap` | 2-D Raman spatial map from LabRAM |
| `ACPLVabScan` | Deprecated; use `ACSpectralSweep` with explicit `sweep=` |

### `processing`
| Function | Purpose |
|---|---|
| `normalise_peak` | Normalise each spectrum to its maximum |
| `normalise_area` | Normalise each spectrum to its integrated area |
| `subtract_background` | Subtract a constant background from a spectral region |
| `smooth_median` | Median filter (1D or 2D) |
| `smooth_savgol` | Savitzky-Golay smoothing |
| `crop` | Crop spectra and x-axis to a range |
| `wavelength_to_energy` / `energy_to_wavelength` | Unit conversion |
| `jacobian_correction_wvl2E` | Apply dλ/dE Jacobian when converting to energy axis |

### `fitting`
| Function / Class | Purpose |
|---|---|
| `fit_lorentzian` | Single Lorentzian peak fit |
| `fit_gaussian` | Single Gaussian peak fit |
| `fit_multi_lorentzian` | Sum of N Lorentzians, with automatic peak detection |
| `fit_scan_peak` | Fit a single peak at every sweep point in a scan |
| `extract_dipole_length` | DC Stark shift → weighted linear fit → dipole length |
| `FitResult` | Dataclass: params, errors, y_fit, residuals, R² |
| `DipoleResult` | Dataclass: slope, intercept, dipole length ± error, R² |

### `plotting`
| Function | Purpose |
|---|---|
| `set_style` | Apply publication-ready Matplotlib rcParams |
| `plot_spectral_map` | 2D map of a sweep's spectra (energy/wavelength vs sweep axis) |
| `plot_spectrum` | Single spectrum from a scan |
| `plot_current` | Leakage current and power monitor |
| `plot_real_space_PL_map` | Single real-space PL image |
| `animate_real_space_PL_map` | Animated gate-dependent real-space PL |
| `plot_stark_shift` | Peak energy vs field with linear fit overlay |
| `save_figure` | Save figure to disk (png, pdf, or both) |
