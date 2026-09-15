# src/tmdc_optics_tools/reference/processors/Dijkstra2025.py
#
# Dijkstra et al., Nature Communications 2025
# "Ten-valley excitonic complexes in charge-tunable monolayer WSe2"
# DOI: 10.1038/s41467-025-65731-x
# Dataset: https://mediatum.ub.tum.de/1793118
#
# Figure 1a: Gate-dependent reflectance contrast (ΔR/R₀) of monolayer WSe2
# Sweep: gate voltage (V) × energy (eV)
#
# Raw data structure (pickle dicts):
#   Reference file : WSe2_reference_1p74eV_8s.pkl
#       wavelength          (n_wl,)
#       counts              (n_pol, n_wl)  or  (n_pol, n_repeat, n_wl)
#       background_counts_det (n_wl,) or (n_pol, n_wl)
#       int_time            float
#
#   Main file      : WSe2_RC_DGS_1p74eV_8s.pkl
#       wavelength          (n_wl,)
#       counts              (n_vg, n_pol, n_repeat, n_wl)
#       background_counts_det (n_wl,) or (n_pol, n_wl)
#       int_time            float
#       v1                  (n_vg, 1)   gate voltages (V)
#       v1_planned          (n_vg,)     nominal gate voltages
#
# Processing pipeline (mirrors authors' notebook):
#   1. Average polarisation axis of counts                      (axis=1, always)
#   2. [Optional] cosmic ray removal across repeat axis         (axis=1 post-pol-avg → axis=2 pre)
#   3. Subtract detector background
#   4. Divide by integration time → counts/s
#   5. Optionally smooth spectra
#   6. Compute RC = (sample - reference) / reference per gate voltage
#   7. Average RC over polarisation axis → RC_2D (n_vg, n_wl)
#   8. Fringe background correction via polygon masks in (energy, voltage) space
#
# The cosmic ray removal and fringe correction are both togglable.

import io
import pickle
import numpy as np
import h5py
from pathlib import Path
from ...constants import HC_EV_NM

from .processor import Processor

# ---------------------------------------------------------------------------
# Constants matching the authors' notebook
# ---------------------------------------------------------------------------

# Fringe-background polygon masks in (energy_eV, voltage_V) space.
# Copied verbatim from the authors' notebook (Cell 8).
# Each entry is an Nx2 array of [energy, voltage] vertices.
_FRINGE_POLYGONS = [
    np.array([[1.74, -5], [1.82, -7], [1.0, -7], [1.0, -5]]),   # block 1
    np.array([[1.74,  7], [1.82,  5], [2.2,  5], [2.2,  7]]),   # block 2
]


# ---------------------------------------------------------------------------
# Processing helpers  (stateless functions — easy to test independently)
# ---------------------------------------------------------------------------

def _wavelength_to_energy(wavelength_nm: np.ndarray) -> np.ndarray:
    return HC_EV_NM / wavelength_nm


def _find_nearest(array: np.ndarray, value: float) -> int:
    """Return the index of the element in *array* closest to *value*."""
    return int(np.argmin(np.abs(array - value)))


def _cosmic_ray_removal(
    counts_array: np.ndarray,
    axis: int,
    threshold: float = 150,
    average_excluded: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove cosmic rays from a stack of repeated spectra.

    Mirrors the authors' ``cosmic_ray_removal`` function exactly.
    Sorts spectra along *axis*, separates the highest values (potential
    cosmics), and selectively re-includes them if they fall within
    *threshold* counts of the running average.

    Parameters
    ----------
    counts_array     : array with a repeat dimension along *axis*
    axis             : axis over which multiple repeats are stored
    threshold        : counts above running average to flag as cosmic
    average_excluded : how many top-sorted spectra to inspect for cosmics;
                       defaults to n_spectra - 2

    Returns
    -------
    cleaned_average : np.ndarray  (repeat axis removed, averaged)
    cosmic_flag     : np.ndarray  bool mask, True where a cosmic was found
    """
    n_spectra = counts_array.shape[axis]
    if average_excluded is None:
        average_excluded = n_spectra - 2

    counts_sorted = np.sort(counts_array, axis=axis)
    counts_sorted = np.swapaxes(counts_sorted, 0, axis)

    counts_no_cos = np.swapaxes(counts_sorted[: n_spectra - average_excluded], 0, axis)
    counts_cos    = np.swapaxes(counts_sorted[n_spectra - average_excluded :], 0, axis)

    avg = np.average(counts_no_cos, axis=axis)
    cosmic_flag = np.zeros_like(avg, dtype=bool)

    for i in range(average_excluded):
        candidate = np.swapaxes(counts_cos, 0, axis)
        candidate = np.average(np.swapaxes(candidate[i : i + 1], 0, axis), axis=axis)
        is_cosmic = (candidate - avg) > threshold

        n_so_far = n_spectra - average_excluded + i
        avg = np.where(is_cosmic,
                       avg,
                       avg * n_so_far / (n_so_far + 1) + candidate / (n_so_far + 1))
        cosmic_flag = np.logical_or(is_cosmic, cosmic_flag)

    return avg, cosmic_flag


def _smooth(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """Box-car moving average along the last axis of *data*."""
    kernel = np.ones(kernel_size) / kernel_size
    if data.ndim == 1:
        return np.convolve(data, kernel, mode="same")
    # Apply along last axis for 2D arrays
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis=-1, arr=data
    )


def _process_reference(raw: dict) -> np.ndarray:
    """
    Process the reference (white-light) pickle into a counts/s spectrum.

    Parameters
    ----------
    raw : dict loaded from the reference .pkl file

    Returns
    -------
    reference_counts_ps : np.ndarray, shape (n_wl,)
    """
    counts = raw["counts"]

    # Average polarisation axis if present (may be 1D or 2D)
    if counts.ndim == 2:
        counts = np.average(counts, axis=0)

    # Subtract detector background
    bg = raw["background_counts_det"]
    if bg.ndim == 2:
        bg = np.average(bg, axis=0)
    counts_bg = counts - bg

    return counts_bg / raw["int_time"]


def _process_main(
    raw: dict,
    reference_ps: np.ndarray,
    remove_cosmics: bool = False,
    smooth_kernel:  int  = None,
    fringe_correction: bool = True,
) -> dict:
    """
    Process the main gate-sweep pickle into a background-corrected RC map.

    Parameters
    ----------
    raw              : dict loaded from the main .pkl file
    reference_ps     : reference spectrum in counts/s, shape (n_wl,)
    remove_cosmics   : whether to run cosmic ray removal on the repeat axis
    smooth_kernel    : box-car kernel size for smoothing before RC; None = no smoothing
    fringe_correction: whether to subtract the polygon-masked fringe background

    Returns
    -------
    dict with keys:
        energy      (n_wl,)      energy axis in eV
        voltage     (n_vg,)      gate voltages in V
        RC          (n_vg, n_wl) reflectance contrast ΔR/R₀
        cosmic_flag (n_vg, n_wl) or None
    """
    wavelength = raw["wavelength"]           # (n_wl,)
    energy     = _wavelength_to_energy(wavelength)
    voltage    = raw["v1"][:, 0]            # (n_vg,) — drop the trailing length-1 dim
    counts     = raw["counts"]              # (n_vg, n_pol, n_repeat, n_wl)

    cosmic_flag = None

    # Step 1 — cosmic ray removal across the repeat axis (axis=2) before averaging
    if remove_cosmics:
        counts, cosmic_flag = _cosmic_ray_removal(counts, axis=2)
        # counts is now (n_vg, n_pol, n_wl) after repeat axis is averaged away
    else:
        # Average the repeat axis directly
        counts = np.average(counts, axis=2)   # (n_vg, n_pol, n_wl)

    # Step 2 — average the polarisation axis
    counts = np.average(counts, axis=1)       # (n_vg, n_wl)

    # Step 3 — subtract detector background
    bg = raw["background_counts_det"]
    if bg.ndim == 2:
        bg = np.average(bg, axis=0)
    counts = counts - bg                      # (n_vg, n_wl)

    # Step 4 — convert to counts/s
    counts_ps = counts / raw["int_time"]      # (n_vg, n_wl)

    # Step 5 — optional smoothing
    ref = np.copy(reference_ps)
    counts_for_rc = np.copy(counts_ps)
    if smooth_kernel is not None:
        counts_for_rc = _smooth(counts_for_rc, smooth_kernel)
        ref           = _smooth(ref,           smooth_kernel)

    # Step 6 — reflection contrast RC = (sample - reference) / reference
    RC = (counts_for_rc - ref[np.newaxis, :]) / ref[np.newaxis, :]   # (n_vg, n_wl)

    # Step 7 — fringe background correction via polygon masks
    if fringe_correction:
        RC = _subtract_fringe_background(RC, energy, raw["v1_planned"])

    return dict(energy=energy, voltage=voltage, RC=RC, cosmic_flag=cosmic_flag)


def _subtract_fringe_background(
    RC:       np.ndarray,
    energy:   np.ndarray,
    voltages: np.ndarray,
) -> np.ndarray:
    """
    Subtract a slowly varying fringe background from the RC map.

    Builds a boolean mask from the polygon regions defined in
    ``_FRINGE_POLYGONS`` (feature-free regions of the map), averages
    the RC values inside the mask along the voltage axis to get a
    background spectrum, and subtracts it.

    Mirrors the authors' Cell 8 exactly.

    Parameters
    ----------
    RC       : (n_vg, n_wl)  reflectance contrast map
    energy   : (n_wl,)       energy axis in eV
    voltages : (n_vg,)       gate voltage axis in V

    Returns
    -------
    RC_corrected : (n_vg, n_wl)
    """
    from skimage.draw import polygon as skimage_polygon

    mask = np.zeros_like(RC)

    for poly_ev in _FRINGE_POLYGONS:
        if np.any(np.isnan(poly_ev)):
            continue
        # Convert (energy, voltage) vertices to (row, col) pixel indices
        col_idx = np.array([_find_nearest(energy,   e) for e in poly_ev[:, 0]])
        row_idx = np.array([_find_nearest(voltages, v) for v in poly_ev[:, 1]])
        rr, cc  = skimage_polygon(row_idx, col_idx, RC.shape)
        mask[rr, cc] += 1

    mask_nan        = np.where(mask == 0, np.nan, 1.0)
    background      = np.nanmean(RC * mask_nan, axis=0)   # (n_wl,) average over voltage
    return RC - background[np.newaxis, :]


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

FTP_HOST = "dataserv.ub.tum.de"
FTP_ID   = "m1793118"   # mediaTUM dataset ID — used as both user and password
_BASE_URL = "https://dataserv.ub.tum.de/public.php/dav/files/m1793118/dataset/Figure%201"

class Dijkstra2025Processor(Processor):
    """
    Processor for Dijkstra et al., Nature Communications 2025.
    
    - Title: Ten-valley excitonic complexes in charge-tunable monolayer WSe2
    - DOI: 10.1038/s41467-025-65731-x
    - Dataset: https://mediatum.ub.tum.de/1793118
    - Figure 1a: gate-dependent reflectance contrast of monolayer WSe2.

    Parameters
    ----------
    meta    : registry entry dict
    out_dir : output directory for the .h5 file
    remove_cosmics   : toggle cosmic ray removal (default False)
    smooth_kernel    : box-car smoothing kernel size; None = no smoothing (default 3,
                       matching the authors' notebook)
    fringe_correction: toggle fringe background subtraction (default True)
    """

    def __init__(
        self,
        meta:              dict,
        out_dir:           Path,
        remove_cosmics:    bool = False,
        smooth_kernel:     int  = 3,
        fringe_correction: bool = True,
    ):
        super().__init__(meta, out_dir)
        self.remove_cosmics    = remove_cosmics
        self.smooth_kernel     = smooth_kernel
        self.fringe_correction = fringe_correction

    def _load_pickle(self, z, path: str) -> dict:
        """Read a pickle file from inside a ZipFile object."""
        with z.open(path) as f:
            return pickle.load(io.BytesIO(f.read()))

    def run(self):
        print(f"  Fetching data ({self.meta['dataset_doi']})...")

        # base = "dataset/Figure 1/"
        # print("  Downloading reference pickle via FTP...")
        # raw_ref  = pickle.loads(
        #     self._fetch_ftp_file("Figure 1/WSe2_reference_1p74eV_8s.pkl", FTP_HOST, FTP_ID)
        # )
        # print("  Downloading main data pickle via FTP...")
        # raw_main = pickle.loads(
        #     self._fetch_ftp_file("Figure 1/WSe2_RC_DGS_1p74eV_8s.pkl", FTP_HOST, FTP_ID)
        # )

        print("  Downloading reference pickle...")
        raw_ref = pickle.loads(self._fetch_file(
            f"{_BASE_URL}/WSe2_reference_1p74eV_8s.pkl",
            verify_ssl=False,
        ))

        print("  Downloading main data pickle...")
        raw_main = pickle.loads(self._fetch_file(
            f"{_BASE_URL}/WSe2_RC_DGS_1p74eV_8s.pkl",
            verify_ssl=False,
        ))
        print(f"  Processing (cosmics={'on' if self.remove_cosmics else 'off'}, "
              f"smooth={self.smooth_kernel}, "
              f"fringe_correction={'on' if self.fringe_correction else 'off'})...")

        reference_ps = _process_reference(raw_ref)
        result       = _process_main(
            raw_main,
            reference_ps,
            remove_cosmics    = self.remove_cosmics,
            smooth_kernel     = self.smooth_kernel,
            fringe_correction = self.fringe_correction,
        )

        with h5py.File(self.out_path, "w") as hf:
            self._write_metadata(hf)

            # Processing parameters — stored so the file is self-documenting
            hf.attrs["remove_cosmics"]    = self.remove_cosmics
            hf.attrs["smooth_kernel"]     = self.smooth_kernel if self.smooth_kernel else 0
            hf.attrs["fringe_correction"] = self.fringe_correction

            # Outer sweep: gate voltage
            vg_sweep = hf.create_group("gate_voltage")
            vg_sweep.attrs["parameter_name"] = "gate_voltage"
            vg_sweep.attrs["parameter_unit"] = "V"
            vg_sweep.attrs["default_value"]  = 0.0   # closest to charge neutrality

            # Shared energy axis
            vg_sweep.create_dataset("energy", data=result["energy"])
            vg_sweep.attrs["energy_unit"] = "eV"

            voltages      = result["voltage"]
            RC            = result["RC"]           # (n_vg, n_wl)
            default_idx   = _find_nearest(voltages, 0.0)

            for i, vg in enumerate(voltages):
                label   = f"{vg:.4f}"
                grp     = vg_sweep.create_group(label)
                grp.create_dataset("spectrum",  data=RC[i])
                grp.create_dataset("energy",    data=result["energy"])
                grp.attrs["parameter_value"]  = float(vg)
                grp.attrs["voltage_index"]    = i
                grp.attrs["spectrum_unit"]    = "dimensionless"   # ΔR/R₀
                grp.attrs["is_default"]       = (i == default_idx)

            # Store cosmic flags if they were computed
            if result["cosmic_flag"] is not None:
                hf.create_dataset("cosmic_flag", data=result["cosmic_flag"].astype(np.uint8))

        print(f"  -> Saved to {self.out_path}")