# tmdc_optics_tools/fitting.py
"""
Common fitting routines for TMD spectroscopy.

All public fitting functions follow the convention:
    fit_*(x, y, ...) -> FitResult

where FitResult is a lightweight dataclass holding parameters, errors,
the best-fit curve, and goodness-of-fit metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from . import constants, processing
from .constants import _x_axis_name_unit


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Container for a single spectral fit.

    Attributes
    ----------
    params : dict[str, float]
        Best-fit parameter values keyed by name.
    errors : dict[str, float]
        1-sigma uncertainties (sqrt of diagonal of covariance matrix).
    x_fit : np.ndarray
        x values used for the fit.
    y_fit : np.ndarray
        Best-fit curve evaluated on *x_fit*.
    residuals : np.ndarray
        y - y_fit on *x_fit*.
    r_squared : float
        Coefficient of determination R².
    model : str
        Name of the model function used.
    converged : bool
        Whether ``curve_fit`` converged.
    """
    params    : dict
    errors    : dict
    x_fit     : np.ndarray
    y_fit     : np.ndarray
    residuals : np.ndarray
    r_squared : float
    model     : str
    converged : bool = True

    def __repr__(self) -> str:
        lines = [f"FitResult [{self.model}]  R²={self.r_squared:.4f}"]
        for k, v in self.params.items():
            lines.append(f"  {k:12s} = {v:.5g} ± {self.errors[k]:.2g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Line-shape functions
# ---------------------------------------------------------------------------

def lorentzian(x, amplitude, center, fwhm):
    """Single Lorentzian peak."""
    gamma = fwhm / 2.0
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)


def gaussian(x, amplitude, center, fwhm):
    """Single Gaussian peak."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)


def voigt_approx(x, amplitude, center, fwhm_g, fwhm_l):
    """
    Pseudo-Voigt approximation (Thompson et al. 1987).
    A weighted sum of Gaussian and Lorentzian with the same FWHM.
    """
    fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
    eta  = (1.36603 * (fwhm_l / fwhm)
            - 0.47719 * (fwhm_l / fwhm)**2
            + 0.11116 * (fwhm_l / fwhm)**3)
    return amplitude * (
        eta * lorentzian(x, 1.0, center, fwhm) +
        (1 - eta) * gaussian(x, 1.0, center, fwhm)
    )


def multi_lorentzian(x, *params):
    """
    Sum of N Lorentzians. ``params`` must have length 3N:
    ``[amp1, cen1, fwhm1, amp2, cen2, fwhm2, ...]``
    """
    if len(params) % 3 != 0:
        raise ValueError("multi_lorentzian requires 3 parameters per peak.")
    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        result += lorentzian(x, params[i], params[i + 1], params[i + 2])
    return result


def multi_voigt(x, *params):
    """
    Sum of N pseudo-Voigt peaks. ``params`` must have length 4N:
    ``[amp1, cen1, fwhm_g1, fwhm_l1, amp2, cen2, fwhm_g2, fwhm_l2, ...]``
    """
    if len(params) % 4 != 0:
        raise ValueError("multi_voigt requires 4 parameters per peak.")
    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 4):
        result += voigt_approx(x, params[i], params[i + 1],
                                params[i + 2], params[i + 3])
    return result


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
#
# The peak models above all decay to zero in their wings, so a dark-count
# pedestal has nowhere to go in the fit: it gets absorbed by inflating the
# amplitude and width instead, and biases the centre whenever the pedestal is
# not symmetric across the fit window.  Adding a baseline term to the model
# removes that bias without needing a separate background region.
#
# Note that a Lorentzian's 1/x^2 wings are partly degenerate with a flat
# offset over a finite window, so the fitted FWHM becomes noisier (and
# somewhat window-dependent) once a baseline is included.  The centre is set
# by symmetry rather than by the wings and stays robust.  This is why the
# default is "constant" rather than "linear": on a narrow window the wings
# already mimic a slope.
#
# Baseline values are reported in FitResult.params — they are a useful
# diagnostic (an offset that drifts across a sweep usually means changing
# dark counts, laser leakage, or a badly chosen window) but are not physics.

_PEAK_PARAM_NAMES = ["amplitude", "center", "fwhm"]

_BASELINES = {
    "none":     ([],                  lambda x: 0.0),
    "constant": (["offset"],          lambda x, c0: c0),
    "linear":   (["offset", "slope"], lambda x, c0, c1: c0 + c1 * x),
}


def _resolve_baseline(baseline) -> tuple:
    """
    Return ``(key, param_names, baseline_fn)`` for a baseline selector.

    Accepts any key of :data:`_BASELINES`, or ``None`` as an alias for
    ``"none"``.
    """
    key = "none" if baseline is None else str(baseline).lower()
    if key not in _BASELINES:
        raise ValueError(
            f"baseline={baseline!r} is not recognised. "
            f"Choose from {tuple(_BASELINES)} (or None for 'none')."
        )
    names, fn = _BASELINES[key]
    return key, list(names), fn


def _model_label(model_name: str, baseline_key: str) -> str:
    """``"lorentzian"`` + ``"linear"`` -> ``"lorentzian+linear"``."""
    return model_name if baseline_key == "none" else f"{model_name}+{baseline_key}"


def _with_baseline(peak_fn, n_peak_params: int, baseline_fn):
    """
    Compose a peak model with a baseline term into a single ``curve_fit`` model.

    The first *n_peak_params* values are forwarded to *peak_fn* and the
    remainder to *baseline_fn*.  With ``baseline="none"`` the remainder is
    empty and the baseline contributes a constant zero.
    """
    def model(x, *params):
        return (peak_fn(x, *params[:n_peak_params])
                + baseline_fn(x, *params[n_peak_params:]))
    return model


def _baseline_p0(key: str, x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Seed the baseline parameters from the edges of the fit window.

    Returns ``(baseline_p0, level)``, where *level* is a representative
    baseline height.  Callers subtract it from the amplitude guess so the
    peak seed measures height *above the pedestal* rather than above zero —
    getting this wrong is the main way a baseline degrades a fit instead of
    improving it.
    """
    if key == "none":
        return (), 0.0

    n      = max(1, len(y) // 10)          # outer decile at each end
    lo, hi = float(np.median(y[:n])), float(np.median(y[-n:]))
    level  = 0.5 * (lo + hi)

    if key == "constant":
        return (level,), level

    x_lo, x_hi = float(np.median(x[:n])), float(np.median(x[-n:]))
    slope = (hi - lo) / (x_hi - x_lo) if x_hi != x_lo else 0.0
    return (lo - slope * x_lo, slope), level


def _peak_amplitude_p0(y: np.ndarray, level: float) -> float:
    """Amplitude seed measured above *level*, guarded against a <= 0 result."""
    amp = float(y.max()) - level
    if not np.isfinite(amp) or amp <= 0:
        amp = float(np.ptp(y)) or 1.0
    return amp


def _complete_p0(p0, n_peak: int, base_p0: tuple, model_name: str) -> np.ndarray:
    """
    Accept a peak-only *p0* and append the auto-seeded baseline values.

    A full-length *p0* (peak + baseline) is passed through unchanged, so a
    caller can override the baseline seed when they need to.  Any other
    length is an error — silently mis-sized guesses would otherwise surface
    as an opaque SciPy traceback.
    """
    p0     = np.asarray(p0, dtype=float).ravel()
    n_base = len(base_p0)

    if p0.size == n_peak:
        return (np.concatenate([p0, np.asarray(base_p0, dtype=float)])
                if n_base else p0)
    if p0.size == n_peak + n_base:
        return p0

    if n_base:
        raise ValueError(
            f"{model_name}: p0 has {p0.size} value(s); expected {n_peak} "
            f"(peak parameters only — the baseline is seeded automatically) "
            f"or {n_peak + n_base} (peak + baseline)."
        )
    raise ValueError(
        f"{model_name}: p0 has {p0.size} value(s); expected {n_peak}."
    )


def _complete_bounds(bounds, n_peak: int, n_base: int, model_name: str) -> tuple:
    """
    Extend sequence-form *bounds* to cover the baseline parameters.

    Scalar bounds — including the default ``(-inf, inf)`` — broadcast to any
    parameter count and are returned unchanged.  Baseline parameters are left
    unbounded.
    """
    if n_base == 0:
        return bounds

    out = []
    for side, value in zip(("lower", "upper"), bounds):
        if np.ndim(value) == 0:
            out.append(value)
            continue
        seq = list(value)
        if len(seq) == n_peak:
            seq = seq + [-np.inf if side == "lower" else np.inf] * n_base
        elif len(seq) != n_peak + n_base:
            raise ValueError(
                f"{model_name}: {side} bounds have {len(seq)} entries; "
                f"expected {n_peak} or {n_peak + n_base}."
            )
        out.append(seq)
    return tuple(out)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear(x, slope, intercept):
    """Simple linear model for use with curve_fit."""
    return slope * x + intercept


def _r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _make_result(
    model_name  : str,
    model_fn,
    param_names : list,
    popt        : np.ndarray,
    pcov,
    x           : np.ndarray,
    y           : np.ndarray,
    converged   : bool,
) -> FitResult:
    """Build a FitResult from curve_fit outputs."""
    perr  = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
    y_fit = model_fn(x, *popt)
    return FitResult(
        params    = dict(zip(param_names, popt)),
        errors    = dict(zip(param_names, perr)),
        x_fit     = x,
        y_fit     = y_fit,
        residuals = y - y_fit,
        r_squared = _r_squared(y, y_fit),
        model     = model_name,
        converged = converged,
    )


def _fit_single_peak(
    peak_fn,
    model_name : str,
    x          : np.ndarray,
    y          : np.ndarray,
    p0         : tuple,
    bounds     : tuple,
    baseline   : str  = "constant",
    peak_names : list = None,
) -> FitResult:
    """
    Shared implementation for single-peak fits (Lorentzian / Gaussian / Voigt).

    Composes *peak_fn* with the selected baseline, completes the initial
    guess and bounds, and runs the fit.  Avoids duplicating the try/except
    and _make_result call in every fitter.

    *peak_names* defaults to the 3-parameter ``(amplitude, center, fwhm)``
    convention; pass an explicit list for a peak shape with a different
    parameter count, e.g. pseudo-Voigt's ``(amplitude, center, fwhm_g,
    fwhm_l)`` — the rest of this function only cares how many there are.
    """
    peak_names  = list(peak_names) if peak_names is not None else _PEAK_PARAM_NAMES
    key, base_names, base_fn = _resolve_baseline(baseline)
    label       = _model_label(model_name, key)
    model_fn    = _with_baseline(peak_fn, len(peak_names), base_fn)
    param_names = peak_names + base_names

    base_p0, _ = _baseline_p0(key, x, y)
    p0         = _complete_p0(p0, len(peak_names), base_p0, label)
    bounds     = _complete_bounds(bounds, len(peak_names),
                                  len(base_names), label)

    try:
        popt, pcov = curve_fit(model_fn, x, y, p0=p0, bounds=bounds, maxfev=5000)
        converged  = True
    except RuntimeError:
        popt, pcov = np.array(p0, dtype=float), None
        converged  = False
        warnings.warn(f"{label} fit did not converge.")

    return _make_result(label, model_fn, param_names,
                        popt, pcov, x, y, converged)


# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def _auto_peak_p0(x: np.ndarray, y: np.ndarray, baseline_key: str) -> tuple:
    """Auto initial guess for a single peak, measured above the baseline."""
    _, level = _baseline_p0(baseline_key, x, y)
    return (_peak_amplitude_p0(y, level),
            x[np.argmax(y)],
            (x[-1] - x[0]) / 10)


def fit_lorentzian(
    x        : np.ndarray,
    y        : np.ndarray,
    p0       : tuple = None,
    bounds   : tuple = (-np.inf, np.inf),
    baseline : str   = "constant",
) -> FitResult:
    """
    Fit a single Lorentzian peak on top of a baseline.

    Parameters
    ----------
    x, y : array-like
        Spectral data.
    p0 : tuple of (amplitude, center, fwhm), optional
        Initial guess for the peak. Auto-estimated if ``None``. The baseline
        is seeded automatically; pass a full-length tuple (peak + baseline)
        to set it explicitly.
    bounds : tuple
        Passed to ``scipy.optimize.curve_fit``. Sequence-form bounds sized
        for the peak parameters alone are extended with unbounded entries
        for the baseline.
    baseline : {"constant", "linear", "none"}
        Baseline added to the peak model. ``"constant"`` (default) fits a
        flat offset, so a dark-count pedestal no longer inflates the
        amplitude and FWHM. ``"linear"`` additionally fits a slope — use it
        when the background visibly tilts across the window. ``"none"``
        reproduces the offset-free model used before this option existed.

    Returns
    -------
    FitResult
        ``params`` gains an ``offset`` key (and ``slope`` for
        ``baseline="linear"``); ``model`` records the choice, e.g.
        ``"lorentzian+constant"``.

    Notes
    -----
    A Lorentzian's ``1/x**2`` wings are partly degenerate with a flat offset
    over a finite window, so the fitted ``fwhm`` (and its uncertainty) is
    more window-sensitive with a baseline than without. The fitted ``center``
    is set by symmetry and stays robust.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    key, _, _ = _resolve_baseline(baseline)
    if p0 is None:
        p0 = _auto_peak_p0(x, y, key)
    return _fit_single_peak(lorentzian, "lorentzian", x, y, p0, bounds, baseline)


def fit_gaussian(
    x        : np.ndarray,
    y        : np.ndarray,
    p0       : tuple = None,
    bounds   : tuple = (-np.inf, np.inf),
    baseline : str   = "constant",
) -> FitResult:
    """
    Fit a single Gaussian peak. Same signature as :func:`fit_lorentzian`.

    A Gaussian decays fast enough that the baseline/wing degeneracy noted in
    :func:`fit_lorentzian` barely applies here.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    key, _, _ = _resolve_baseline(baseline)
    if p0 is None:
        p0 = _auto_peak_p0(x, y, key)
    return _fit_single_peak(gaussian, "gaussian", x, y, p0, bounds, baseline)


_VOIGT_PARAM_NAMES = ["amplitude", "center", "fwhm_g", "fwhm_l"]


def _auto_voigt_p0(x: np.ndarray, y: np.ndarray, baseline_key: str) -> tuple:
    """Auto initial guess for a single Voigt peak: even Gaussian/Lorentzian split."""
    _, level = _baseline_p0(baseline_key, x, y)
    fwhm0 = (x[-1] - x[0]) / 10
    return (_peak_amplitude_p0(y, level), x[np.argmax(y)], fwhm0, fwhm0)


def fit_voigt(
    x        : np.ndarray,
    y        : np.ndarray,
    p0       : tuple = None,
    bounds   : tuple = (-np.inf, np.inf),
    baseline : str   = "constant",
) -> FitResult:
    """
    Fit a single pseudo-Voigt peak on top of a baseline.

    The pseudo-Voigt (:func:`voigt_approx`) is a weighted sum of a Gaussian
    and a Lorentzian sharing one effective FWHM, with the mix set by
    *fwhm_g* and *fwhm_l* — Thompson et al. 1987's standard approximation to
    a true Voigt (Gaussian ⊛ Lorentzian convolution), adequate whenever the
    line shape itself is not what is being used to extract a physical
    broadening mechanism.

    Parameters
    ----------
    x, y : array-like
        Spectral data.
    p0 : tuple of (amplitude, center, fwhm_g, fwhm_l), optional
        Initial guess. Auto-estimated if ``None``, splitting the
        auto-estimated width evenly between the Gaussian and Lorentzian
        components. The baseline is seeded automatically; pass a full-length
        tuple (peak + baseline) to set it explicitly.
    bounds : tuple
        As :func:`fit_lorentzian`.
    baseline : {"constant", "linear", "none"}
        As :func:`fit_lorentzian`.

    Returns
    -------
    FitResult
        ``params`` has ``amplitude``, ``center``, ``fwhm_g``, ``fwhm_l``,
        plus the baseline terms.

    See Also
    --------
    fit_multi_voigt : the same line shape, for several overlapping peaks.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    key, _, _ = _resolve_baseline(baseline)
    if p0 is None:
        p0 = _auto_voigt_p0(x, y, key)
    return _fit_single_peak(voigt_approx, "voigt", x, y, p0, bounds, baseline,
                             peak_names=_VOIGT_PARAM_NAMES)


def fit_multi_lorentzian(
    x           : np.ndarray,
    y           : np.ndarray,
    n_peaks     : int  = None,
    p0          : list = None,
    bounds      : tuple = None,
    peak_kwargs : dict = None,
    baseline    : str  = "constant",
) -> FitResult:
    """
    Fit a sum of N Lorentzian peaks on top of a baseline.

    Parameters
    ----------
    x, y : array-like
    n_peaks : int, optional
        Number of peaks. Inferred from ``p0`` length if not given.
    p0 : list of (amp, center, fwhm) per peak, optional
        If ``None``, peaks are found automatically via
        ``scipy.signal.find_peaks``. The baseline is seeded automatically;
        append baseline values to set them explicitly.
    bounds : tuple, optional
        ``([lower, ...], [upper, ...])`` passed to ``curve_fit``.
        Auto-constructed if ``None``. Sequence-form bounds sized for the
        peak parameters alone are extended with unbounded baseline entries.
    peak_kwargs : dict, optional
        Extra kwargs forwarded to ``scipy.signal.find_peaks`` during
        automatic peak detection.
    baseline : {"constant", "linear", "none"}
        Baseline added to the summed peaks — see :func:`fit_lorentzian`.
        Overlapping wings make the baseline harder to separate here than in
        the single-peak case, so prefer ``"constant"`` unless the background
        visibly tilts.

    Returns
    -------
    FitResult
        ``params`` keys are ``amp_0``, ``center_0``, ``fwhm_0``, ``amp_1``, …
        followed by ``offset`` (and ``slope`` for ``baseline="linear"``).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    span = x[-1] - x[0]

    key, base_names, base_fn = _resolve_baseline(baseline)
    label      = _model_label("multi_lorentzian", key)
    base_p0, level = _baseline_p0(key, x, y)

    if p0 is None:
        pk_kw = peak_kwargs or {}
        peaks, _ = find_peaks(y, height=y.max() * 0.1, **pk_kw)
        if n_peaks is not None:
            order = np.argsort(y[peaks])[::-1]
            peaks = peaks[order[:n_peaks]]
        p0 = []
        for pk in peaks:
            # Height above the pedestal, not above zero.
            amp = float(y[pk]) - level
            if not np.isfinite(amp) or amp <= 0:
                amp = float(y[pk])
            p0.extend([amp, x[pk], span / 20])

    # Peak count comes from the peak parameters only — trailing baseline
    # values (if the caller supplied them) must not be counted as a peak.
    # A peak-only p0 is an exact multiple of 3; anything else carries the
    # baseline as well.  _complete_p0 rejects genuinely mis-sized input.
    n_supplied = np.asarray(p0, dtype=float).ravel().size
    n = (n_supplied // 3 if n_supplied % 3 == 0
         else (n_supplied - len(base_names)) // 3)
    n_peak_params = 3 * n

    p0_flat = _complete_p0(p0, n_peak_params, base_p0, label)

    if bounds is None:
        lo = [0,      x.min(), 0   ] * n
        hi = [np.inf, x.max(), span] * n
        bounds = (lo, hi)
    bounds = _complete_bounds(bounds, n_peak_params, len(base_names), label)

    param_names = [name for i in range(n)
                   for name in (f"amp_{i}", f"center_{i}", f"fwhm_{i}")]
    param_names += base_names

    model_fn = _with_baseline(multi_lorentzian, n_peak_params, base_fn)

    try:
        popt, pcov = curve_fit(
            model_fn, x, y, p0=p0_flat, bounds=bounds, maxfev=10000
        )
        converged = True
    except RuntimeError:
        popt, pcov = p0_flat, None
        converged  = False
        warnings.warn(f"{label} fit did not converge.")

    return _make_result(label, model_fn, param_names,
                        popt, pcov, x, y, converged)


def fit_multi_voigt(
    x           : np.ndarray,
    y           : np.ndarray,
    n_peaks     : int  = None,
    p0          : list = None,
    bounds      : tuple = None,
    peak_kwargs : dict = None,
    baseline    : str  = "constant",
) -> FitResult:
    """
    Fit a sum of N pseudo-Voigt peaks on top of a baseline.

    Same idea as :func:`fit_multi_lorentzian`, with each component a
    pseudo-Voigt (:func:`voigt_approx`, 4 parameters) rather than a pure
    Lorentzian (3): overlapping peaks are not forced into a single line
    shape when neither pure Lorentzian nor pure Gaussian broadening is known
    to hold for all of them — e.g. Raman modes broadened by both a finite
    phonon lifetime (Lorentzian) and the instrument resolution (Gaussian).

    Parameters
    ----------
    x, y : array-like
    n_peaks : int, optional
        Number of peaks. Inferred from ``p0`` length if not given.
    p0 : list of (amp, center, fwhm_g, fwhm_l) per peak, optional
        If ``None``, peaks are found automatically via
        ``scipy.signal.find_peaks``, with the Gaussian and Lorentzian widths
        seeded equal (an even split of the auto-estimated peak width). Pass
        this explicitly to seed known peak positions — e.g. from literature
        — rather than relying on automatic detection, which can merge or
        miss components that overlap heavily. The baseline is seeded
        automatically; append baseline values to set it explicitly.
    bounds : tuple, optional
        ``([lower, ...], [upper, ...])`` passed to ``curve_fit``.
        Auto-constructed if ``None``. Sequence-form bounds sized for the
        peak parameters alone are extended with unbounded baseline entries.
    peak_kwargs : dict, optional
        Extra kwargs forwarded to ``scipy.signal.find_peaks`` during
        automatic peak detection.
    baseline : {"constant", "linear", "none"}
        As :func:`fit_multi_lorentzian`.

    Returns
    -------
    FitResult
        ``params`` keys are ``amp_0``, ``center_0``, ``fwhm_g_0``,
        ``fwhm_l_0``, ``amp_1``, … followed by ``offset`` (and ``slope`` for
        ``baseline="linear"``).

    Examples
    --------
    Two overlapping modes near 250 cm⁻¹ plus one near 310 cm⁻¹, seeded from
    known peak positions rather than automatic detection (which would likely
    merge the first two into one broad peak):

    >>> result = fit_multi_voigt(
    ...     spectrum.shift, spectrum.counts,
    ...     p0=[(2000, 245, 4, 4), (2000, 253, 4, 4), (500, 310, 4, 4)],
    ... )
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    span = x[-1] - x[0]

    key, base_names, base_fn = _resolve_baseline(baseline)
    label      = _model_label("multi_voigt", key)
    base_p0, level = _baseline_p0(key, x, y)

    if p0 is None:
        pk_kw = peak_kwargs or {}
        peaks, _ = find_peaks(y, height=y.max() * 0.1, **pk_kw)
        if n_peaks is not None:
            order = np.argsort(y[peaks])[::-1]
            peaks = peaks[order[:n_peaks]]
        p0 = []
        for pk in peaks:
            # Height above the pedestal, not above zero.
            amp = float(y[pk]) - level
            if not np.isfinite(amp) or amp <= 0:
                amp = float(y[pk])
            fwhm0 = span / 20
            p0.extend([amp, x[pk], fwhm0, fwhm0])

    # Peak count comes from the peak parameters only, same logic as
    # fit_multi_lorentzian but 4 parameters per peak instead of 3.
    n_supplied = np.asarray(p0, dtype=float).ravel().size
    n = (n_supplied // 4 if n_supplied % 4 == 0
         else (n_supplied - len(base_names)) // 4)
    n_peak_params = 4 * n

    p0_flat = _complete_p0(p0, n_peak_params, base_p0, label)

    if bounds is None:
        lo = [0,      x.min(), 0,    0   ] * n
        hi = [np.inf, x.max(), span, span] * n
        bounds = (lo, hi)
    bounds = _complete_bounds(bounds, n_peak_params, len(base_names), label)

    param_names = [name for i in range(n)
                   for name in (f"amp_{i}", f"center_{i}",
                                f"fwhm_g_{i}", f"fwhm_l_{i}")]
    param_names += base_names

    model_fn = _with_baseline(multi_voigt, n_peak_params, base_fn)

    try:
        popt, pcov = curve_fit(
            model_fn, x, y, p0=p0_flat, bounds=bounds, maxfev=10000
        )
        converged = True
    except RuntimeError:
        popt, pcov = p0_flat, None
        converged  = False
        warnings.warn(f"{label} fit did not converge.")

    return _make_result(label, model_fn, param_names,
                        popt, pcov, x, y, converged)


def locate_residual_peak(result: FitResult, search_range: tuple) -> tuple:
    """
    Find where a fit's residual is largest within a given x range.

    For a component you know is missing from the model but do not know the
    position of — e.g. a shoulder a single peak's fit leaves behind, whose
    literature position (if any) does not actually match this measurement —
    seed it here instead of guessing. Fit the peaks you *are* confident
    about first, call this on the result to see where the leftover signal
    actually peaks, and use that as the next fit's seed.

    Only looks for a positive residual (real signal the model
    under-predicts), not the most negative one (over-prediction).

    Parameters
    ----------
    result : FitResult
        A previous fit, e.g. from :func:`fit_voigt` or :func:`fit_multi_voigt`.
    search_range : tuple of (x_min, x_max)
        Restrict the search to this range — e.g. away from an already-fitted
        peak's own core, where residual structure more often reflects the
        fit's imperfect peak shape than a genuine distinct feature.

    Returns
    -------
    position : float
        ``result.x_fit`` value where ``result.residuals`` is largest within
        *search_range*.
    height : float
        The residual's value there.

    Examples
    --------
    >>> main_fit = fit_multi_voigt(x, y, p0=[(60000, 250.5, 2, 2), (2000, 310, 4, 4)])
    >>> shoulder_x, shoulder_height = locate_residual_peak(main_fit, (253, 285))
    >>> refit = fit_multi_voigt(
    ...     x, y,
    ...     p0=[(60000, 250.5, 2, 2), (shoulder_height, shoulder_x, 4, 4), (2000, 310, 4, 4)],
    ... )
    """
    mask = (result.x_fit >= search_range[0]) & (result.x_fit <= search_range[1])
    if not np.any(mask):
        raise ValueError(
            f"No fitted points fall within search_range={search_range}; "
            f"result.x_fit spans {result.x_fit.min():.4g} to {result.x_fit.max():.4g}."
        )
    x_in_range      = result.x_fit[mask]
    residual_in_range = result.residuals[mask]
    idx = np.argmax(residual_in_range)
    return float(x_in_range[idx]), float(residual_in_range[idx])


def _n_peaks(result: FitResult) -> int:
    return sum(1 for key in result.params if key.startswith("center_"))


def find_extra_peaks(
    x,
    y,
    base_result : FitResult,
    max_new     : int   = 6,
    snr         : float = 5.0,
    min_gap     : float = None,
    edge_margin : float = None,
) -> FitResult:
    """
    Iteratively seed and fit additional Voigt peaks from a fit's own residual.

    Generalises the single, once-only lookup :func:`locate_residual_peak`
    already does for :func:`fit_raman_modes`'s unseeded shoulder mode: the
    same idea (find the residual's biggest remaining peak, seed a new Voigt
    component there, refit) repeats here rather than stopping after one, so
    it covers a fit missing one more feature or several without knowing the
    count ahead of time.

    Parameters
    ----------
    x, y : array-like
        The same arrays *base_result* was fit to.
    base_result : FitResult
        A previous :func:`fit_multi_voigt` result, or a wrapper's (e.g.
        :func:`fit_raman_modes`, :func:`fit_pl_peaks`), to extend.
    max_new : int
        Upper bound on how many peaks may be added.
    snr : float
        A candidate peak must exceed ``snr`` times the residual's MAD (a
        robust noise estimate) to be accepted.
    min_gap, edge_margin : float, optional
        Minimum distance, in *x*'s own units, a candidate must keep from an
        existing peak's center, and from the data's own edges, respectively.
        ``None`` (default) derives both from the fit's x-range — ``min_gap``
        2% of it, ``edge_margin`` 5% — scale-relative so the same defaults
        suit a Raman shift (cm⁻¹) window or a PL energy (eV) one without a
        domain-specific number. A small ``min_gap`` matters: two genuinely
        distinct features can sit only a few percent of the window apart,
        and excluding that whole neighbourhood as "too close to an existing
        peak" silently discards a real, resolvable feature. ``edge_margin``
        matters separately: a multi-Voigt fit is least constrained at the
        boundary of what it was fit to, so a residual "peak" sitting exactly
        at the data's edge is far more often that boundary effect than a
        real feature.

    Returns
    -------
    FitResult
        *base_result* unchanged if nothing was added, otherwise a new
        :func:`fit_multi_voigt` result with *base_result*'s peaks first, in
        their original order, followed by however many were added.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    span = x.max() - x.min()
    if min_gap is None:
        min_gap = 0.02 * span
    if edge_margin is None:
        edge_margin = 0.05 * span

    result = base_result
    n_start = _n_peaks(result)
    lo_edge, hi_edge = x.min() + edge_margin, x.max() - edge_margin

    while _n_peaks(result) - n_start < max_new:
        mask = (result.x_fit >= lo_edge) & (result.x_fit <= hi_edge)
        x_fit, residual = result.x_fit[mask], result.residuals[mask]
        noise = 1.4826 * np.median(np.abs(residual - np.median(residual)))

        n = _n_peaks(result)
        centers = [result.params[f"center_{i}"] for i in range(n)]
        far_enough = np.all([np.abs(x_fit - c) > min_gap for c in centers], axis=0)
        if not np.any(far_enough):
            break

        idx = np.argmax(np.where(far_enough, residual, -np.inf))
        peak_x, peak_h = x_fit[idx], residual[idx]
        if peak_h < snr * noise:
            break

        p0 = [(result.params[f"amp_{i}"], result.params[f"center_{i}"],
               result.params[f"fwhm_g_{i}"], result.params[f"fwhm_l_{i}"])
              for i in range(n)]
        p0.append((peak_h, peak_x, span / 20, span / 20))
        result = fit_multi_voigt(x, y, p0=p0)

    return result


def extract_fit_param_map(
    results_grid : np.ndarray,
    param_key     : str,
    mode_index    : int,
    label_grid    : np.ndarray = None,
    only_label    = None,
) -> np.ndarray:
    """
    Pull one fitted parameter into a 2-D array from a grid of fit results.

    For a spatial map fit pixel-by-pixel — e.g. :func:`fit_raman_modes` run
    over every position of a
    :class:`~tmdc_optics_tools.loaders.RamanMap` — not every pixel's fit
    necessarily has the requested mode at all (a monolayer pixel has no
    B₂g). A missing key is left ``NaN`` rather than raising, so
    :func:`plotting.plot_image` renders it as "not computed here", not
    "computed and found to be zero".

    Parameters
    ----------
    results_grid : np.ndarray of FitResult, shape (n_y, n_x)
        One fit per grid position.
    param_key : str
        The parameter type, e.g. ``"center"`` or ``"amp"``.
    mode_index : int
        Which peak, matching a :func:`fit_multi_voigt`/
        :func:`fit_multi_lorentzian` result's own ``f"{param_key}_{i}"``
        naming.
    label_grid : np.ndarray, shape (n_y, n_x), optional
        Per-position labels (e.g. a classified layer count from
        :func:`classify_raman_layer`) to restrict which positions are read.
    only_label : optional
        Requires *label_grid*. Positions where ``label_grid != only_label``
        are left ``NaN`` regardless of whether their fit has the key — use
        this to keep a mode's map from covering positions it was never fit
        for (e.g. B₂g over monolayer pixels), rather than plotting whatever
        that position's *other* mode happened to leave in the same slot.

    Returns
    -------
    np.ndarray, shape (n_y, n_x)
    """
    key = f"{param_key}_{mode_index}"
    arr = np.full(results_grid.shape, np.nan)
    for iy in range(results_grid.shape[0]):
        for ix in range(results_grid.shape[1]):
            if only_label is not None and label_grid[iy, ix] != only_label:
                continue
            result = results_grid[iy, ix]
            if key in result.params:
                arr[iy, ix] = result.params[key]
    return arr


def classify_raman_layer(shift: np.ndarray, counts: np.ndarray, material: str) -> int:
    """
    Identify a material's layer count from a Raman spectrum.

    Looks up *material* in :data:`constants.RAMAN_LAYER_DISCRIMINATOR` for
    a mode present at one layer count and absent at another, and checks its
    height above the *local* baseline (not just above zero, since the
    baseline level itself varies across a map) at that entry's own search
    range and threshold. WSe₂'s entry uses B₂g, which requires interlayer
    coupling a single layer does not have; a different material could use
    a different discriminating mode without changing this function.

    Parameters
    ----------
    shift, counts : array-like
        Raman shift (cm⁻¹) and counts.
    material : str
        Key into :data:`constants.RAMAN_LAYER_DISCRIMINATOR`, e.g.
        ``"WSe2"``.

    Returns
    -------
    int
        The layer count (``constants.RAMAN_LAYER_DISCRIMINATOR[material]``'s
        ``"present_in"`` or ``"absent_in"``).

    Raises
    ------
    KeyError
        If *material* has no entry in
        :data:`constants.RAMAN_LAYER_DISCRIMINATOR`.

    See Also
    --------
    fit_raman_modes : fit the identified layer count's modes.
    """
    config = constants.RAMAN_LAYER_DISCRIMINATOR[material]
    shift, counts = np.asarray(shift, float), np.asarray(counts, float)
    search_mask   = (shift >= config["search_range"][0]) & (shift <= config["search_range"][1])
    baseline_mask = (shift >= config["baseline_range"][0]) & (shift <= config["baseline_range"][1])
    height = counts[search_mask].max() - np.median(counts[baseline_mask])
    return config["present_in"] if height > config["threshold"] else config["absent_in"]


def fit_raman_modes(
    shift          : np.ndarray,
    counts         : np.ndarray,
    material       : str,
    n_layers       : int,
    fit_window     : tuple = None,
    shoulder_range : tuple = None,
    seeds          : dict  = None,
) -> FitResult:
    """
    Fit a material's known Raman modes for a given layer count.

    Mode identities, seed positions, and fit tolerances come from
    :data:`constants.RAMAN_MODES` rather than being encoded here, so a new
    material or layer count is a data addition, not a new function. One
    mode per (*material*, *n_layers*) entry — its ``"shoulder_mode"`` — is
    not seeded directly: the seeded ("known") modes are fit first (a
    discovery fit), the shoulder mode's true position is then found from
    that fit's own residual via :func:`locate_residual_peak`, and all modes
    are refit together. A shoulder mode is located from the residual rather
    than seeded because it is weak and its position is not fixed by its
    name: WSe₂'s 2LA(M) sits near 258.7 cm⁻¹ in the bilayer and 260.4 cm⁻¹
    in the monolayer. Mode assignments follow Pan et al., "Signature of
    lattice dynamics in twisted 2D homo/hetero-bilayers", 2D Materials 9,
    045018 (2022), doi:10.1088/2053-1583/ac83d4.

    Parameters
    ----------
    shift, counts : array-like
        Raman shift (cm⁻¹) and counts, e.g. from a
        :class:`~tmdc_optics_tools.loaders.RamanSpectrum`.
    material : str
        Key into :data:`constants.RAMAN_MODES`, e.g. ``"WSe2"``.
    n_layers : int
        Key into ``constants.RAMAN_MODES[material]``, e.g. ``1`` or ``2``.
    fit_window, shoulder_range : tuple of (shift_min, shift_max), optional
        Override the config's defaults.
    seeds : dict, optional
        ``{mode_name: seed_position}`` overriding the config's seed for one
        or more of the *known* (non-shoulder) modes — e.g. a strained
        sample whose modes sit measurably off the config's defaults. Only
        the seed position is overridden; that mode's ``fwhm_seed`` and
        ``center_tol`` still come from the config.

    Returns
    -------
    FitResult
        An N-peak :func:`fit_multi_voigt` result, N = the number of modes
        for this (*material*, *n_layers*). ``center_0``/``amp_0``/… is
        ``constants.RAMAN_MODES[material][n_layers]["modes"][0]``, and so
        on in that same order.

    Raises
    ------
    KeyError
        If (*material*, *n_layers*) has no entry in
        :data:`constants.RAMAN_MODES`.

    See Also
    --------
    classify_raman_layer : identify *n_layers* from the spectrum itself,
        for a map where it is not known ahead of time.
    fit_multi_voigt : the general-purpose fitter this wraps.
    locate_residual_peak : how the shoulder mode's seed is found.
    """
    config        = constants.RAMAN_MODES[material][n_layers]
    modes         = config["modes"]
    shoulder_mode = config["shoulder_mode"]
    peak_config   = config["peaks"]
    fit_window     = fit_window if fit_window is not None else config["fit_window"]
    shoulder_range = shoulder_range if shoulder_range is not None else config["shoulder_range"]
    seed_overrides = seeds or {}

    shift, counts = np.asarray(shift, float), np.asarray(counts, float)
    mask = (shift >= fit_window[0]) & (shift <= fit_window[1])
    x, y = shift[mask], counts[mask]

    known_modes = [m for m in modes if m != shoulder_mode]

    known_p0 = []
    for i, mode in enumerate(known_modes):
        seed = seed_overrides.get(mode, peak_config[mode]["seed"])
        fwhm_seed = peak_config[mode]["fwhm_seed"]
        if i == 0:
            amp = y.max() - y.min()  # the dominant mode, taken globally
        else:
            # A window max, not the single nearest-grid-point value: on a
            # weak signal (e.g. a map pixel near a layer-count boundary,
            # rather than the strong reference spectra these seeds were
            # tuned against), missing the true peak by even one bin
            # underestimates the seed enough that the final fit below
            # fails to converge.
            window = (x >= seed - 4.0) & (x <= seed + 4.0)
            amp = y[window].max() - y.min() if np.any(window) else y.max() - y.min()
        known_p0.append((amp, seed, fwhm_seed, fwhm_seed))

    def _bounds(mode_names, p0):
        lo, hi = [], []
        for mode, (amp, cen, fg, fl) in zip(mode_names, p0):
            tol = peak_config[mode]["center_tol"]
            lo += [0.0, cen - tol, 0.5, 0.5]
            hi += [np.inf, cen + tol, 30.0, 30.0]
        return lo, hi

    discovery_fit = fit_multi_voigt(x, y, p0=known_p0, bounds=_bounds(known_modes, known_p0))
    shoulder_x, shoulder_amp = locate_residual_peak(discovery_fit, shoulder_range)

    shoulder_fwhm = peak_config[shoulder_mode]["fwhm_seed"]
    shoulder_p0   = (shoulder_amp, shoulder_x, shoulder_fwhm, shoulder_fwhm)
    shoulder_index = modes.index(shoulder_mode)
    full_p0 = known_p0[:shoulder_index] + [shoulder_p0] + known_p0[shoulder_index:]

    return fit_multi_voigt(x, y, p0=full_p0, bounds=_bounds(modes, full_p0))


def fit_pl_peaks(
    x,
    y,
    material   : str,
    fit_window : tuple = None,
    seeds      : dict  = None,
) -> FitResult:
    """
    Fit a material's known PL peaks.

    Same shape as :func:`fit_raman_modes`: peak identities, seed positions,
    and fit tolerances come from :data:`constants.PL_PEAKS` rather than being
    encoded here, so a new material is a data addition, not a new function.
    Unlike :func:`fit_raman_modes`, every peak is seeded directly — there is
    no residual-located "shoulder" step, since :data:`constants.PL_PEAKS`
    carries no ``"shoulder_mode"`` the way :data:`constants.RAMAN_MODES`
    does: none of a material's listed peaks is assumed unresolved ahead of
    time. Smaller features the seeded peaks miss are a separate, composable
    step — see :func:`find_extra_peaks`.

    Parameters
    ----------
    x, y : array-like
        Spectral axis (energy or wavelength) and intensity.
    material : str
        Key into :data:`constants.PL_PEAKS`, e.g. ``"WSe2"``.
    fit_window : tuple of (x_min, x_max), optional
        Overrides the config's default.
    seeds : dict, optional
        ``{peak_name: seed_position}`` overriding the config's seed for one
        or more peaks — e.g. a strained or gated sample whose peaks sit
        measurably off the config's defaults. Only the seed position is
        overridden; that peak's ``fwhm_seed`` and ``center_tol`` still come
        from the config.

    Returns
    -------
    FitResult
        An N-peak :func:`fit_multi_voigt` result, N =
        ``len(constants.PL_PEAKS[material]["peaks"])``. ``center_0``/
        ``amp_0``/… is ``constants.PL_PEAKS[material]["peaks"][0]``, and so
        on in that same order.

    Raises
    ------
    KeyError
        If *material* has no entry in :data:`constants.PL_PEAKS`.

    See Also
    --------
    find_extra_peaks : locate and fit smaller features the seeded peaks miss,
        composable on this function's result.
    fit_raman_modes : same shape, for Raman modes instead of PL peaks.
    fit_multi_voigt : the general-purpose fitter this wraps.
    """
    config = constants.PL_PEAKS[material]
    peaks, peak_config = config["peaks"], config["peak_config"]
    fit_window = fit_window if fit_window is not None else config["fit_window"]
    seed_overrides = seeds or {}

    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_win, y_win = x[mask], y[mask]

    p0, lo, hi = [], [], []
    for peak in peaks:
        seed = seed_overrides.get(peak, peak_config[peak]["seed"])
        fwhm_seed = peak_config[peak]["fwhm_seed"]
        tol = peak_config[peak]["center_tol"]
        # Max within +/- tol of the seed, not the single nearest-grid-point
        # value -- missing the true peak by one bin would underestimate the
        # amplitude seed enough that the fit below fails to converge (same
        # reasoning as fit_raman_modes's amplitude seeding).
        window = (x_win >= seed - tol) & (x_win <= seed + tol)
        amp = y_win[window].max() - y_win.min() if np.any(window) else y_win.max() - y_win.min()
        p0.append((amp, seed, fwhm_seed, fwhm_seed))
        lo += [0.0, seed - tol, 0.002, 0.002]
        hi += [np.inf, seed + tol, 0.15, 0.15]

    return fit_multi_voigt(x_win, y_win, p0=p0, bounds=(lo, hi))


def fit_scan_peak(
    scan,
    x_axis     : str        = "energy",
    x_range    : tuple      = None,
    model      : str        = "lorentzian",
    sweep_mask : np.ndarray = None,
    baseline   : str        = "constant",
) -> list[FitResult]:
    """
    Fit a single peak in every sweep of an
    :class:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep`.

    Corrections are configured at load time on the scan object (via
    ``bg_region_nm`` / ``bg_region_eV``, ``apply_jacobian`` and ``cosmic_rays``).
    This function fits the most-corrected array the scan has for *x_axis* —
    :attr:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep.best_energy_spectra`
    or :attr:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep.best_spectra` — so
    a declared background or cosmic-ray repair reaches the fit on either axis.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
    x_axis : {"energy", "wavelength"}
    x_range : tuple of (x_min, x_max), optional
        Restrict the fit to this spectral window. Units match *x_axis*. Bounds
        are inclusive, and their order carries no information: ``(1.42, 1.30)``
        is the same window as ``(1.30, 1.42)``. Fits the full range if ``None``.
    model : {"lorentzian", "gaussian"}
        Peak shape to fit.
    sweep_mask : np.ndarray of bool, optional
        Boolean mask of length ``scan.n_sweeps``. Only sweeps where the
        mask is ``True`` are fitted; the rest receive a non-converged
        placeholder so that result indices stay aligned with ``scan.ef``.
        Fits all sweeps when ``None``.
    baseline : {"constant", "linear", "none"}
        Baseline fitted alongside the peak in every sweep — see
        :func:`fit_lorentzian`. ``"constant"`` (default) means an
        un-subtracted dark-count pedestal no longer inflates the fitted
        amplitude and width.

    Returns
    -------
    list of FitResult, length = scan.n_sweeps
        Each result's ``x_fit`` is a **view** into the scan's own axis, so it
        must not be modified in place.

    Raises
    ------
    ValueError
        If *x_axis* is not ``"energy"`` or ``"wavelength"``; if *x_range* selects
        no point of that axis, the message giving the axis's span; or if it
        selects fewer points than the model has free parameters.

    Warns
    -----
    UserWarning
        When a bound of *x_range* lies beyond the end of the axis by more than
        half a pixel, so the window fitted is narrower than the one asked for.
    """
    return _fit_scan_peak(scan, x_axis, x_range, model, sweep_mask, baseline,
                          stacklevel=4)


def _fit_scan_peak(
    scan,
    x_axis     : str,
    x_range    : tuple,
    model      : str,
    sweep_mask : np.ndarray,
    baseline   : str,
    *,
    stacklevel : int,
) -> list[FitResult]:
    """
    Implementation of :func:`fit_scan_peak`.

    *stacklevel* is a required argument because the public entry points sit at
    different depths — :func:`fit_scan_peak` one frame up,
    :func:`extract_dipole_length` two — and a warning about a window is only
    useful pointing at the line that chose the window.
    """
    # Ahead of the two ternaries, which would otherwise read an unrecognised axis
    # as the wavelength one and fit it without complaint.
    _, unit = _x_axis_name_unit(x_axis, what="fit_scan_peak()")

    values  = scan.energy     if x_axis == "energy" else scan.wavelength
    # The scan's own choice on each axis, so a declared cosmic-ray repair is
    # fitted rather than the file's spikes.  loaders._resolve_spectra answers the
    # same question, but importing it here would put fitting on top of the I/O
    # layer; the two accessors are reachable without that.
    spectra = (scan.best_energy_spectra if x_axis == "energy"
               else scan.best_spectra)
    fit_fn  = fit_lorentzian if model == "lorentzian" else fit_gaussian

    key, base_names, _ = _resolve_baseline(baseline)
    label      = _model_label(model, key)
    nan_names  = _PEAK_PARAM_NAMES + base_names

    # A slice, so one window cuts the axis and every sweep's counts to the same
    # length, and each cut is a view.  slice(None) is the whole axis, which keeps
    # the windowed and un-windowed paths on one code path.
    px = (processing._window_slice(values, x_range, axis=x_axis, unit=unit,
                                  what="fit_scan_peak()", stacklevel=stacklevel)
          if x_range is not None else slice(None))
    x  = values[px]

    # Fewer points than free parameters is not a fit, and curve_fit reports it as a
    # TypeError, which _fit_single_peak's RuntimeError handler does not catch — so
    # it escapes as neither a fit nor a non-converged result.  The minimum belongs
    # to the model rather than to the axis, which is why it is checked here and not
    # inside the window helper.
    n_params = len(_PEAK_PARAM_NAMES) + len(base_names)
    if x.size < n_params:
        raise ValueError(
            f"fit_scan_peak(): the {x_axis} window {x[0]:.6g}–{x[-1]:.6g} {unit} "
            f"covers {x.size} point{'' if x.size == 1 else 's'}, fewer than the "
            f"{n_params} free parameters of a {label} model. Widen x_range."
        )

    if sweep_mask is None:
        sweep_mask = np.ones(scan.n_sweeps, dtype=bool)

    # Fit only selected sweeps; insert a non-converged placeholder for the rest
    results = []
    for i in range(scan.n_sweeps):
        if sweep_mask[i]:
            results.append(fit_fn(x, spectra[px, i].astype(float),
                                  baseline=baseline))
        else:
            # Placeholder so indices stay aligned with scan.ef.  Fresh dicts
            # per result — a shared one would alias across every placeholder.
            results.append(FitResult(
                params    = {k: np.nan for k in nan_names},
                errors    = {k: np.nan for k in nan_names},
                x_fit     = x,
                y_fit     = np.full_like(x, np.nan),
                residuals = np.full_like(x, np.nan),
                r_squared = np.nan,
                model     = label,
                converged = False,
            ))
    return results


# ---------------------------------------------------------------------------
# TRPL sparse lifetime fitting
# ---------------------------------------------------------------------------
#
# A TRPL decay's rise and fall are both convolutions of the true carrier
# dynamics with the instrument response (IRF) — the detector and electronics
# don't respond instantaneously, so the measured rise time is a mixture of
# real physics and instrument bandwidth unless the IRF is deconvolved out.
# A second complication is that the number of physically-distinct decay
# channels (fast non-radiative capture, slower radiative recombination,
# diffusion-limited rise, ...) is generally not known ahead of time, which
# rules out fitting a fixed-size sum of exponentials the way
# :func:`fit_multi_lorentzian` fits a fixed number of peaks.
#
# The approach below (adapted from Tagarelli et al., SI Section 5, "Time
# resolved photoluminescence fitting") sidesteps that by fitting against a
# large, fixed dictionary of candidate lifetimes instead of a small, flexible
# one: every trace is expressed as a linear combination of IRF-convolved
# exponentials drawn from a broad, log-spaced grid (`tau_grid`), and an L1
# (Lasso) penalty is used to keep only as many of them active as the data can
# support. This turns "how many exponentials, with which lifetimes" — a hard
# nonlinear search — into "which coefficients on a fixed dictionary are
# nonzero" — a convex sparse regression. "Sparse" refers to this
# regularization (few nonzero coefficients), not to the data itself.
#
# Allowing each coefficient to be positive *or* negative is what lets one
# fit capture both rise and decay: a positive amplitude adds an ordinary
# decaying exponential, while a negative one subtracts one, which delays
# where the summed curve turns over and so reproduces a rise. Most sparse
# solvers (e.g. plain non-negative least squares) cannot do this — `Lasso`
# is used here specifically because its coefficients are unconstrained in
# sign.

@dataclass
class SparseLifetimeResult:
    """
    Result of a sparse, IRF-deconvolved lifetime-distribution fit to a decay.

    Attributes
    ----------
    t : np.ndarray
        Time axis used for the fit, in the same units as *tau_grid*
        (nanoseconds for :class:`~tmdc_optics_tools.loaders.AttoCubeTRPLSweep`
        data), relative to whatever the caller chose as t=0.
    y : np.ndarray
        Data fitted, in whatever normalization the caller passed in.
    y_fit : np.ndarray
        Best-fit curve on *t*.
    tau_grid : np.ndarray
        Candidate lifetimes the fit was allowed to draw on.
    amplitudes : np.ndarray
        Fitted coefficient for every entry of *tau_grid*, in the same order.
        Positive values are decay components, negative values are rise
        components (see module notes above); most entries are exactly zero.
    tau_rise, tau_rise_err : float
        Amplitude-weighted mean and spread (see :func:`fit_sparse_lifetime`)
        of the active rise-component lifetimes. ``NaN`` if none are active.
    tau_decay, tau_decay_err : float
        Same, for the active decay components.
    n_rise, n_decay : int
        Number of active (nonzero-amplitude) rise / decay components.
    alpha : float
        L1 regularization strength used for this fit.
    residual_std : float
        Standard deviation of ``y - y_fit``.
    """
    t             : np.ndarray
    y             : np.ndarray
    y_fit         : np.ndarray
    tau_grid      : np.ndarray
    amplitudes    : np.ndarray
    tau_rise      : float
    tau_rise_err  : float
    tau_decay     : float
    tau_decay_err : float
    n_rise        : int
    n_decay       : int
    alpha         : float
    residual_std  : float

    def __repr__(self) -> str:
        return (
            f"SparseLifetimeResult\n"
            f"  Rise         : {self.tau_rise:.4g} ± {self.tau_rise_err:.2g} "
            f"({self.n_rise} component(s))\n"
            f"  Decay        : {self.tau_decay:.4g} ± {self.tau_decay_err:.2g} "
            f"({self.n_decay} component(s))\n"
            f"  alpha        : {self.alpha:.3g}\n"
            f"  Residual std : {self.residual_std:.4g}"
        )


def build_irf_kernel(
    time          : np.ndarray,
    counts        : np.ndarray,
    dt            : float,
    window_before : float = 0.3,
    window_after  : float = 2.0,
) -> np.ndarray:
    """
    Build a discrete, unit-sum convolution kernel from a measured IRF trace.

    The trace's own background level is subtracted, it is re-centered so its
    peak sits at zero delay, and it is resampled onto a grid with spacing
    *dt* — which must match the sample spacing of whatever data the kernel
    will be convolved with in :func:`fit_sparse_lifetime`, or the convolution
    mixes time bins that do not correspond to the same delay.

    Parameters
    ----------
    time, counts : np.ndarray
        Measured instrument-response decay, e.g. an
        :class:`~tmdc_optics_tools.loaders.AttoCubeTRPLSweep`'s ``time`` and
        ``best_decays[:, 0]`` loaded from a dedicated IRF file.
    dt : float
        Target sample spacing, in the same units as *time*.
    window_before, window_after : float
        How much of the measured trace to keep, in *time* units before/after
        the IRF's own peak. The 2.0 default for *window_after* is not
        universal — an IRF with a fast rise but a slow tail (a real effect
        seen on the AttoCube setup: ~44 ps FWHM rise, tail extending roughly
        1.5 ns past the peak) needs the tail captured explicitly, since a
        symmetric placeholder shape would miss it entirely.

    Returns
    -------
    np.ndarray
        Kernel of length ``ceil(window_before/dt) + ceil(window_after/dt) + 1``,
        centered on the IRF peak, normalized to unit sum.
    """
    time, counts = np.asarray(time, float), np.asarray(counts, float)
    baseline = np.median(counts)
    counts   = np.clip(counts - baseline, 0, None)
    t_rel    = time - time[np.argmax(counts)]

    half_before = int(np.ceil(window_before / dt))
    half_after  = int(np.ceil(window_after / dt))
    offsets     = np.arange(-half_before, half_after + 1)
    kernel      = np.interp(offsets * dt, t_rel, counts, left=0.0, right=0.0)
    return kernel / kernel.sum()


def _build_lifetime_dictionary(
    t        : np.ndarray,
    tau_grid : np.ndarray,
    kernel   : np.ndarray,
) -> np.ndarray:
    """
    Design matrix for :func:`fit_sparse_lifetime`.

    Column i is ``exp(-t/tau_i)`` for ``t >= 0`` (zero before that — the
    exponential switches on at the pulse), convolved with *kernel* and
    rescaled to unit peak height so that every column contributes on the
    same amplitude scale regardless of *tau_i*.

    ``scipy.ndimage.convolve1d`` centers the kernel on each output sample;
    unlike ``np.convolve(..., mode="full")[:n]``, this introduces no spurious
    time shift, which matters because that shift would otherwise be
    degenerate with (and bias) the fitted lifetimes themselves.
    """
    # (n_t, n_tau): one IRF-convolved exponential column per candidate lifetime.
    n = len(t)
    X = np.zeros((n, len(tau_grid)))
    for i, tau in enumerate(tau_grid):
        decay = np.where(t >= 0, np.exp(-t / tau), 0.0)
        X[:, i] = convolve1d(decay, kernel, mode="constant", cval=0.0)
    peak = X.max(axis=0)
    peak[peak == 0] = 1.0
    return X / peak


def _weighted_lifetime(
    tau_grid   : np.ndarray,
    amplitudes : np.ndarray,
    sign       : str,
) -> tuple:
    """
    Amplitude-weighted mean and spread of the active lifetimes of one sign.

    *sign* selects ``amplitudes < 0`` ("rise") or ``amplitudes > 0``
    ("decay"). Weighting by ``|amplitude|`` means a component the fit barely
    used barely moves the average, rather than every active lifetime
    counting equally regardless of how much of the trace it explains.

    Returns
    -------
    mean, std, n_active : float, float, int
        ``(nan, nan, 0)`` if no component of the requested sign is active.
    """
    mask = amplitudes < 0 if sign == "rise" else amplitudes > 0
    if not mask.any():
        return np.nan, np.nan, 0
    w    = np.abs(amplitudes[mask])
    taus = tau_grid[mask]
    mean = np.sum(w * taus) / np.sum(w)
    std  = np.sqrt(np.sum(w * (taus - mean) ** 2) / np.sum(w))
    return mean, std, int(mask.sum())


def fit_sparse_lifetime(
    t          : np.ndarray,
    y          : np.ndarray,
    irf_kernel : np.ndarray,
    tau_grid   : np.ndarray = None,
    tau_range  : tuple      = (1e-3, 5.0),
    n_tau      : int        = 60,
    alpha      : float      = 1e-2,
) -> SparseLifetimeResult:
    """
    Fit a decay as a sparse, signed mixture of IRF-convolved exponentials.

    See the module notes above :class:`SparseLifetimeResult` for the idea in
    full; in short, *y* is expressed as a linear combination of exponentials
    drawn from *tau_grid*, each convolved with *irf_kernel*, with an L1
    penalty keeping most coefficients at exactly zero. Coefficients are
    unconstrained in sign: positive ones behave as decay, negative ones as
    rise, summarized in the returned result's ``tau_rise`` / ``tau_decay``.

    This is deliberately not a conventional fixed-count multi-exponential
    fit (contrast :func:`fit_multi_lorentzian`'s fixed *n_peaks*): choosing
    how many exponentials to fit ahead of time requires already knowing how
    many decay channels are present, which is exactly what is often unknown
    for, e.g., a spatially- or strain-varying sample. Letting a broad grid of
    candidates compete under sparsity turns that unknown count into a convex
    problem instead of a nonlinear search over model order.

    Parameters
    ----------
    t : np.ndarray
        Time axis, relative to the pulse arrival. Must already be
        pulse-relative (i.e. subtract the IRF peak time before calling) —
        the dictionary's exponentials are zero for ``t < 0``, so this is
        what makes "before" vs. "after" the excitation meaningful.
    y : np.ndarray
        Counts to fit. Not normalized internally — normalize beforehand
        (e.g. with :func:`~tmdc_optics_tools.processing.normalise_minmax`,
        as :func:`fit_scan_lifetime` does) if *alpha* and the resulting
        amplitudes are meant to be compared across traces with different
        absolute intensities.
    irf_kernel : np.ndarray
        Convolution kernel from :func:`build_irf_kernel`, built with the same
        sample spacing as *t*.
    tau_grid : np.ndarray, optional
        Candidate lifetimes. Built as *n_tau* log-spaced points across
        *tau_range* when not given explicitly; pass this directly to reuse
        an identical grid across multiple fits (e.g. for the rate-
        distribution comparison in :class:`SparseLifetimeResult.amplitudes`
        to line up trace-to-trace).
    tau_range : tuple of (tau_min, tau_max)
        Only used to build the default *tau_grid*. Keep *tau_max* within a
        few times the width of the *t* window: candidate lifetimes much
        longer than that are nearly indistinguishable from a flat offset
        over the window, and because the reported rise/decay lifetime is
        amplitude-weighted, even a tiny spurious coefficient at a very long
        tau can dominate that average. A lifetime pegged at *tau_max* in the
        result is a sign the window is too short to constrain a genuine
        lifetime there, not a physical value.
    n_tau : int
        Number of candidate lifetimes. Only used to build the default
        *tau_grid*.
    alpha : float
        L1 regularization strength passed to ``sklearn.linear_model.Lasso``.
        Too small lets the fit chase noise (many active lifetimes with tiny,
        physically meaningless weights); too large collapses everything to a
        single lifetime that can no longer distinguish rise from decay.
        There is no default that suits every dataset — sweep a few values
        and compare ``residual_std`` and ``n_rise``/``n_decay`` across them.

    Returns
    -------
    SparseLifetimeResult
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    tau_grid = (
        np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_tau)
        if tau_grid is None else np.asarray(tau_grid, float)
    )

    # Imported here, not at module level: scikit-learn is a heavy import and
    # this is the only function in the package that needs it, so `import
    # tmdc_optics_tools` should not pay for it.
    from sklearn.linear_model import Lasso

    X = _build_lifetime_dictionary(t, tau_grid, irf_kernel)
    lasso = Lasso(alpha=alpha, max_iter=200_000, fit_intercept=False)
    lasso.fit(X, y)
    amplitudes = lasso.coef_
    y_fit = X @ amplitudes

    tau_rise,  tau_rise_err,  n_rise  = _weighted_lifetime(tau_grid, amplitudes, "rise")
    tau_decay, tau_decay_err, n_decay = _weighted_lifetime(tau_grid, amplitudes, "decay")

    return SparseLifetimeResult(
        t=t, y=y, y_fit=y_fit, tau_grid=tau_grid, amplitudes=amplitudes,
        tau_rise=tau_rise, tau_rise_err=tau_rise_err,
        tau_decay=tau_decay, tau_decay_err=tau_decay_err,
        n_rise=n_rise, n_decay=n_decay,
        alpha=alpha, residual_std=float(np.std(y - y_fit)),
    )


def fit_scan_lifetime(
    scan,
    t0         : float,
    irf_kernel : np.ndarray,
    t_range    : tuple      = (-0.2, 1.3),
    tau_grid   : np.ndarray = None,
    tau_range  : tuple      = (1e-3, 5.0),
    n_tau      : int        = 60,
    alpha      : float      = 1e-2,
) -> list[SparseLifetimeResult]:
    """
    Fit a sparse lifetime distribution to every sweep of an
    :class:`~tmdc_optics_tools.loaders.AttoCubeTRPLSweep`.

    Each sweep's decay is windowed to *t_range* relative to *t0* and
    normalized to its own ``[0, 1]`` range
    (:func:`~tmdc_optics_tools.processing.normalise_minmax`) before being
    passed to :func:`fit_sparse_lifetime`, so *alpha* means the same thing
    across sweeps regardless of absolute count rate or a residual baseline
    within the window. This is the same function whether *scan* came from a
    directory sweep or a single file — a one-point sweep (e.g. a single spot
    loaded on its own) just returns a length-1 list.

    Parameters
    ----------
    scan : AttoCubeTRPLSweep
    t0 : float
        Pulse arrival time, in the same units as ``scan.time`` (ns) — e.g.
        the IRF's own peak time.
    irf_kernel : np.ndarray
        Convolution kernel from :func:`build_irf_kernel`, built with the
        same sample spacing as ``scan.time``.
    t_range : tuple of (t_min, t_max)
        Fit window relative to *t0*. See *tau_range* below for why this
        should stay short relative to its upper edge.
    tau_grid, tau_range, n_tau, alpha
        Forwarded to :func:`fit_sparse_lifetime` for every sweep, so every
        sweep is fit against an identical dictionary.

    Returns
    -------
    list of SparseLifetimeResult, length = scan.n_sweeps
    """
    window = (scan.time - t0 >= t_range[0]) & (scan.time - t0 <= t_range[1])
    t_rel  = scan.time[window] - t0

    results = []
    for i in range(scan.n_sweeps):
        y = processing.normalise_minmax(scan.best_decays[window, i])
        results.append(fit_sparse_lifetime(
            t_rel, y, irf_kernel,
            tau_grid=tau_grid, tau_range=tau_range, n_tau=n_tau, alpha=alpha,
        ))
    return results


# ---------------------------------------------------------------------------
# Dipole length extraction
# ---------------------------------------------------------------------------

@dataclass
class DipoleResult:
    """
    Result of a dipole length extraction from a gate-dependent PL scan.

    The dipole length is extracted from the linear Stark shift:

        ΔE = −d · F   →   d = −dE/dF · (1/e)

    Because energies are already in eV (divided by e), the (1/e) factor
    drops out and the dipole length in nm is:

        d [nm] = |dE [eV] / dF [V/nm]|
               = |slope [eV/(mV/nm)]| × 1000

    Attributes
    ----------
    ef : np.ndarray
        Electric field values for all sweep points (mV/nm).
    peak_energies : np.ndarray
        Fitted peak center energies at each field point (eV).
    peak_errors : np.ndarray
        1-sigma uncertainty on each peak center (eV).
        ``NaN`` where the lineshape fit did not converge.
    slope : float
        Linear slope dE/dF in eV/(mV/nm).
    slope_err : float
        1-sigma uncertainty on the slope (eV/(mV/nm)).
    intercept : float
        Linear intercept E₀ in eV (energy at zero field).
    intercept_err : float
        1-sigma uncertainty on the intercept (eV).
    dipole_length : float
        |slope| × 1000, in nm.
    dipole_length_err : float
        Propagated 1-sigma uncertainty on the dipole length (nm).
    dipole_length_angstrom : float
        Dipole length in Ångström (dipole_length × 10).
    r_squared : float
        R² of the linear fit.
    peak_model : str
        Lineshape model used to extract peak centers (e.g. ``"lorentzian"``).
    converged_mask : np.ndarray of bool
        True for sweep points where the lineshape fit converged.
    method : str
        Linear-fit method used: ``"wls"``, ``"minmax"``, or ``"bootstrap"``.
    n_bootstrap : int or None
        Number of bootstrap iterations. ``None`` unless ``method="bootstrap"``.
    """
    ef                    : np.ndarray
    peak_energies         : np.ndarray
    peak_errors           : np.ndarray
    slope                 : float
    slope_err             : float
    intercept             : float
    intercept_err         : float
    dipole_length         : float
    dipole_length_err     : float
    dipole_length_angstrom: float
    r_squared             : float
    peak_model            : str
    converged_mask        : np.ndarray
    method                : str = "wls"
    n_bootstrap           : int = None   # only meaningful for method="bootstrap"

    def __repr__(self) -> str:
        method_str = self.method
        if self.method == "bootstrap" and self.n_bootstrap is not None:
            method_str += f" (n={self.n_bootstrap})"
        return (
            f"DipoleResult\n"
            f"  Dipole length : {self.dipole_length:.4f} ± {self.dipole_length_err:.4f} nm"
            f"  ({self.dipole_length_angstrom:.2f} Å)\n"
            f"  Slope dE/dF   : {self.slope:.4e} ± {self.slope_err:.2e} eV/(mV/nm)\n"
            f"  Intercept E₀  : {self.intercept:.4f} ± {self.intercept_err:.4f} eV\n"
            f"  R²            : {self.r_squared:.4f}\n"
            f"  Peak model    : {self.peak_model}\n"
            f"  Method        : {method_str}\n"
            f"  Sweep points  : {self.converged_mask.sum()} / {len(self.converged_mask)} converged"
        )


def _prepare_dipole_data(
    scan,
    x_range      : tuple,
    model        : str,
    active_range : tuple,
    baseline     : str = "constant",
) -> tuple:
    """
    Shared setup for all dipole extraction methods.

    Runs the per-sweep lineshape fits and constructs the masked arrays
    (ef_fit, E_fit, sig_fit) ready for a linear fit.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
    x_range : tuple or None
    model : str
    active_range : tuple or None
        Combined ef_range / Efield_range already resolved by the caller.
    baseline : str
        Baseline model forwarded to :func:`fit_scan_peak`.

    Returns
    -------
    ef : np.ndarray
        Full electric field array (all sweeps).
    peak_energies : np.ndarray
        Fitted peak centers (all sweeps, NaN where not converged).
    peak_errors : np.ndarray
        1-sigma uncertainties on peak centers (NaN where not converged or
        where the covariance was unusable).
    converged : np.ndarray of bool
    ef_fit, E_fit, sig_fit : np.ndarray
        Masked arrays for the linear fit (converged + within active_range).
        sig_fit contains NaN where the covariance was unusable; each
        linear fitter handles these internally.
    """
    sweep_mask = None
    if active_range is not None:
        sweep_mask = (scan.ef >= active_range[0]) & (scan.ef <= active_range[1])

    # stacklevel=5: _window_slice, _fit_scan_peak, here, extract_dipole_length,
    # the researcher's line.  Measured, not counted off the def lines.
    fit_results   = _fit_scan_peak(
        scan, x_axis="energy", x_range=x_range, model=model, stacklevel=5,
        sweep_mask=sweep_mask, baseline=baseline,
    )
    peak_energies = np.array([r.params["center"] for r in fit_results])
    peak_errors   = np.array([r.errors["center"]  for r in fit_results])
    converged     = np.array([r.converged          for r in fit_results])

    # Mark bad/zero errors as inf so they act as zero-weight points
    peak_errors = np.where(
        np.isfinite(peak_errors) & (peak_errors > 0), peak_errors, np.inf
    )

    ef   = scan.ef.copy()
    mask = converged.copy()
    if active_range is not None:
        mask &= (ef >= active_range[0]) & (ef <= active_range[1])

    if mask.sum() < 2:
        raise ValueError(
            f"Only {mask.sum()} usable sweep point(s) after applying field range "
            f"and removing non-converged fits. Need at least 2."
        )

    # Restore inf → NaN for the returned full arrays (clean display)
    peak_errors_out = np.where(np.isinf(peak_errors), np.nan, peak_errors)
    # sig_fit passed to fitters: NaN where inf (each fitter handles it)
    sig_fit = np.where(np.isinf(peak_errors[mask]), np.nan, peak_errors[mask])

    return ef, peak_energies, peak_errors_out, converged, ef[mask], peak_energies[mask], sig_fit


def _dipole_wls(
    ef_fit  : np.ndarray,
    E_fit   : np.ndarray,
    sig_fit : np.ndarray,
) -> tuple:
    """
    Weighted least squares (WLS / χ² minimisation) linear fit.

    Each point is weighted by 1/σ², and ``absolute_sigma=True`` ensures
    the covariance matrix has correct physical units so slope_err is a
    genuine 1-sigma uncertainty in eV/(mV/nm).

    Points with NaN sigma are given a very large sigma (effectively zero
    weight) so they don't influence the fit but don't cause it to fail.
    Falls back to unweighted polyfit if curve_fit fails.

    Returns
    -------
    slope, slope_err, intercept, intercept_err
    """
    # When all sigmas are NaN (e.g. argmax-tracked peaks with no per-point
    # errors), omit sigma so curve_fit estimates covariance from residuals.
    all_nan = not np.any(np.isfinite(sig_fit))
    if all_nan:
        try:
            popt, pcov = curve_fit(_linear, ef_fit, E_fit)
            slope, intercept         = popt
            slope_err, intercept_err = np.sqrt(np.diag(pcov))
        except (RuntimeError, ValueError):
            warnings.warn("WLS fit failed; falling back to unweighted polyfit.")
            slope, intercept         = np.polyfit(ef_fit, E_fit, 1)
            slope_err = intercept_err = np.nan
        return slope, slope_err, intercept, intercept_err

    sig_safe = np.where(np.isfinite(sig_fit), sig_fit, 1e10)
    try:
        popt, pcov = curve_fit(
            _linear, ef_fit, E_fit,
            sigma=sig_safe, absolute_sigma=True,
        )
        slope, intercept         = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError):
        warnings.warn("WLS fit failed; falling back to unweighted polyfit.")
        slope, intercept         = np.polyfit(ef_fit, E_fit, 1)
        slope_err = intercept_err = np.nan
    return slope, slope_err, intercept, intercept_err


def _dipole_minmax(
    ef_fit  : np.ndarray,
    E_fit   : np.ndarray,
    sig_fit : np.ndarray,
) -> tuple:
    """
    Min/max slope method (extremal fit).

    Finds the steepest and shallowest lines still consistent with the
    data by solving a linear program with per-point slack variables s_i ≥ 0:

        minimise / maximise   slope
        subject to            slope·F_i + intercept ≤ E_i + σ_i + s_i
                              slope·F_i + intercept ≥ E_i - σ_i - s_i
                              s_i ≥ 0  for all i

    The slack terms penalise constraint violations so the LP is always
    feasible. A large penalty (1e6) on Σ s_i discourages slack from being
    used except where unavoidable (e.g. a point whose noise exceeds σ_i).

    Points with NaN σ are skipped — they impose no constraint.

    The best-fit slope and intercept come from an unweighted polyfit of
    the valid points.

    Returns
    -------
    slope, slope_err, intercept, intercept_err

    Notes
    -----
    slope_err     = (slope_max - slope_min) / 2
    intercept_err = (intercept_max - intercept_min) / 2
    """
    from scipy.optimize import linprog

    finite = np.isfinite(sig_fit)
    ef_v, E_v, sig_v = ef_fit[finite], E_fit[finite], sig_fit[finite]
    n = len(ef_v)

    if n < 2:
        warnings.warn("minmax: fewer than 2 finite-error points; returning NaN errors.")
        slope, intercept = np.polyfit(ef_fit, E_fit, 1)
        return slope, np.nan, intercept, np.nan

    # Variables: x = [slope, intercept, s_0, ..., s_{n-1}]
    # Constraints (2n inequalities):
    #   upper:  slope·F_i + intercept - s_i ≤  E_i + σ_i
    #   lower: -slope·F_i - intercept - s_i ≤ -(E_i - σ_i)
    A_ub = np.vstack([
        np.hstack([np.column_stack([ ef_v,  np.ones(n)]), -np.eye(n)]),
        np.hstack([np.column_stack([-ef_v, -np.ones(n)]), -np.eye(n)]),
    ])
    b_ub    = np.concatenate([E_v + sig_v, -(E_v - sig_v)])
    bounds  = [(None, None), (None, None)] + [(0, None)] * n

    slack_penalty = 1e6
    slopes, intercepts = [], []
    for sign in (+1, -1):   # maximise then minimise slope
        c     = np.zeros(2 + n)
        c[0]  = sign
        c[2:] = slack_penalty
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            slopes.append(res.x[0])
            intercepts.append(res.x[1])
        else:
            warnings.warn(f"minmax linprog did not converge (sign={sign:+d}).")
            slopes.append(np.nan)
            intercepts.append(np.nan)

    slope_max, slope_min         = max(slopes), min(slopes)
    intercept_max, intercept_min = max(intercepts), min(intercepts)

    slope, intercept = np.polyfit(ef_v, E_v, 1)
    slope_err        = (slope_max - slope_min) / 2.0
    intercept_err    = (intercept_max - intercept_min) / 2.0

    return slope, slope_err, intercept, intercept_err


def _dipole_bootstrap(
    ef_fit      : np.ndarray,
    E_fit       : np.ndarray,
    sig_fit     : np.ndarray,
    n_bootstrap : int = 2000,
    rng         : np.random.Generator = None,
) -> tuple:
    """
    Bootstrap resampling of the linear slope.

    For each iteration, each energy point is perturbed by a Gaussian
    draw scaled by its 1-sigma uncertainty:

        E'_i = E_i + ε_i,   ε_i ~ N(0, σ_i)

    A weighted least squares line is then fitted to the perturbed dataset.
    The slope uncertainty is the standard deviation of the resulting slope
    distribution. The best-fit slope and intercept come from a single WLS
    fit to the unperturbed data.

    Points with NaN σ receive zero perturbation (their uncertainty is
    unknown, so they are kept fixed) and very large sigma in the WLS
    weight (effectively zero weight).

    Parameters
    ----------
    ef_fit, E_fit, sig_fit : np.ndarray
    n_bootstrap : int
        Number of resampling iterations. Default 2000.
    rng : np.random.Generator, optional
        For reproducibility: pass ``np.random.default_rng(seed)``.

    Returns
    -------
    slope, slope_err, intercept, intercept_err
    """
    if rng is None:
        rng = np.random.default_rng()

    slope, _, intercept, _ = _dipole_wls(ef_fit, E_fit, sig_fit)

    all_nan = not np.any(np.isfinite(sig_fit))

    boot_slopes     = np.empty(n_bootstrap)
    boot_intercepts = np.empty(n_bootstrap)

    if all_nan:
        # No per-point errors — perturb by the residual scatter and fit
        # unweighted, so the bootstrap distribution reflects the data's own
        # noise level.
        residuals = E_fit - (slope * ef_fit + intercept)
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 2 else 0.0
        for i in range(n_bootstrap):
            E_perturbed = E_fit + rng.normal(0.0, residual_std, size=len(E_fit))
            try:
                popt, _ = curve_fit(_linear, ef_fit, E_perturbed)
                boot_slopes[i], boot_intercepts[i] = popt
            except (RuntimeError, ValueError):
                boot_slopes[i]     = np.nan
                boot_intercepts[i] = np.nan
    else:
        sig_perturb = np.where(np.isfinite(sig_fit), sig_fit, 0.0)
        sig_wls     = np.where(np.isfinite(sig_fit), sig_fit, 1e10)
        for i in range(n_bootstrap):
            E_perturbed = E_fit + rng.normal(0.0, sig_perturb)
            try:
                popt, _ = curve_fit(
                    _linear, ef_fit, E_perturbed,
                    sigma=sig_wls, absolute_sigma=True,
                )
                boot_slopes[i], boot_intercepts[i] = popt
            except (RuntimeError, ValueError):
                boot_slopes[i]     = np.nan
                boot_intercepts[i] = np.nan

    slope_err     = np.nanstd(boot_slopes)
    intercept_err = np.nanstd(boot_intercepts)

    return slope, slope_err, intercept, intercept_err


def extract_dipole_length(
    scan,
    x_range      : tuple = None,
    model        : str   = "lorentzian",
    ef_range     : tuple = None,
    Efield_range : tuple = None,
    method       : str   = "wls",
    n_bootstrap  : int   = 2000,
    rng          : np.random.Generator = None,
    baseline     : str   = "constant",
) -> DipoleResult:
    """
    Extract the excitonic dipole length from the DC Stark shift in a
    gate-dependent PL scan.

    The procedure is:

    1. Fit a lineshape to the PL spectrum at every sweep point on the
       **energy** axis (restricted to *x_range* if supplied). This gives
       a fitted peak center energy E(F) at each field F, together with a
       per-point uncertainty σ from the covariance matrix.
    2. Optionally restrict the field range used for the linear fit to
       *ef_range* (e.g. to exclude the non-linear high-field regime).
    3. Fit a line E(F) = slope · F + intercept using the chosen *method*.
    4. Derive the dipole length and propagate uncertainties.

    .. note::
        Background subtraction and cosmic-ray repair are configured at load time
        on the scan object, via ``bg_region_nm`` / ``bg_region_eV`` and
        ``cosmic_rays``.  The per-point fits run on
        :attr:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep.best_energy_spectra`,
        which is the most-corrected energy-axis array available.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep
        Must have ``ef`` set (requires a
        :class:`~tmdc_optics_tools.loaders.DeviceGeometry`).
    x_range : tuple of (E_min, E_max) in eV, optional
        Spectral window for the lineshape fit. Strongly recommended to
        zoom in on the exciton of interest.
    model : {"lorentzian", "gaussian"}
        Lineshape model. Lorentzian is the physical choice for
        homogeneously broadened excitons.
    ef_range : tuple of (F_min, F_max) in mV/nm, optional
        Restrict the linear fit to this field range.
    Efield_range : tuple of (F_min, F_max) in mV/nm, optional
        Alias for *ef_range*. Takes precedence if both are supplied.
    method : {"wls", "minmax", "bootstrap"}
        Linear-fit method used to extract the slope and its uncertainty:

        ``"wls"``
            Weighted least squares (χ² minimisation). Each point is
            weighted by 1/σ². The slope uncertainty comes from the
            covariance matrix with ``absolute_sigma=True``. Statistically
            optimal when the σ_i are accurate and errors are Gaussian.

        ``"minmax"``
            Extremal fit (min/max slope). Finds the steepest and
            shallowest lines still consistent with all error bars via a
            linear program. The uncertainty is half the range between
            these extremes. Conservative and visually intuitive — the two
            extreme lines can be overlaid directly on the Stark-shift plot.

        ``"bootstrap"``
            Bootstrap resampling. Perturbs each E_i by a Gaussian draw
            scaled by σ_i, refits the slope n_bootstrap times, and
            reports the standard deviation of the slope distribution.
            Makes no assumptions beyond Gaussian per-point errors.

    n_bootstrap : int
        Number of bootstrap iterations. Only used when
        ``method="bootstrap"``. Default 2000.
    rng : np.random.Generator, optional
        Random number generator for reproducibility when using bootstrap,
        e.g. ``np.random.default_rng(42)``.
    baseline : {"constant", "linear", "none"}
        Baseline fitted alongside the peak at every field point — see
        :func:`fit_lorentzian`. With ``"constant"`` (default) a dark-count
        pedestal no longer biases the fitted centres, so the extracted
        slope is cleaner even without a load-time background region.
        Pass ``"none"`` to reproduce results from before this option
        existed.

        Note that the per-point ``σ`` on the centre generally grows slightly
        with a baseline (the offset is partly degenerate with a Lorentzian's
        wings), and those σ are the WLS weights — so slopes can shift a
        little even where the centres barely move.

    Returns
    -------
    DipoleResult

    Raises
    ------
    ValueError
        If ``scan.ef`` is ``None``, fewer than 2 usable sweep points remain,
        *method* is not recognised, or *x_range* selects no point of the energy
        axis.

    Warns
    -----
    UserWarning
        When a bound of *x_range* lies beyond the end of the energy axis, so the
        window fitted is narrower than the one asked for.

    Examples
    --------
    >>> # Default: weighted least squares
    >>> result = extract_dipole_length(scan, x_range=(1.30, 1.42))

    >>> # Min/max slope (conservative, visually intuitive)
    >>> result = extract_dipole_length(scan, x_range=(1.30, 1.42), method="minmax")

    >>> # Bootstrap with fixed seed for reproducibility
    >>> result = extract_dipole_length(
    ...     scan, x_range=(1.30, 1.42),
    ...     method="bootstrap", n_bootstrap=5000,
    ...     rng=np.random.default_rng(42),
    ... )
    >>> print(result)
    """
    _METHODS = ("wls", "minmax", "bootstrap")
    if method not in _METHODS:
        raise ValueError(
            f"method='{method}' is not recognised. Choose from {_METHODS}."
        )

    if scan.ef is None:
        raise ValueError(
            "scan.ef is None — supply a DeviceGeometry when loading the scan."
        )

    active_range = Efield_range if Efield_range is not None else ef_range

    # --- Shared setup: lineshape fits + masking ---
    ef, peak_energies, peak_errors, converged, ef_fit, E_fit, sig_fit = (
        _prepare_dipole_data(scan, x_range, model, active_range, baseline)
    )

    # --- Linear fit: dispatch to chosen method ---
    if method == "wls":
        slope, slope_err, intercept, intercept_err = _dipole_wls(
            ef_fit, E_fit, sig_fit
        )
    elif method == "minmax":
        slope, slope_err, intercept, intercept_err = _dipole_minmax(
            ef_fit, E_fit, sig_fit
        )
    else:  # bootstrap
        slope, slope_err, intercept, intercept_err = _dipole_bootstrap(
            ef_fit, E_fit, sig_fit, n_bootstrap=n_bootstrap, rng=rng,
        )

    # --- Derived quantities ---
    r_squared         = _r_squared(E_fit, slope * ef_fit + intercept)
    dipole_length     = abs(slope) * 1000.0
    dipole_length_err = abs(slope_err) * 1000.0 if np.isfinite(slope_err) else np.nan

    return DipoleResult(
        ef                     = ef,
        peak_energies          = peak_energies,
        peak_errors            = peak_errors,
        slope                  = slope,
        slope_err              = slope_err,
        intercept              = intercept,
        intercept_err          = intercept_err,
        dipole_length          = dipole_length,
        dipole_length_err      = dipole_length_err,
        dipole_length_angstrom = dipole_length * 10.0,
        r_squared              = r_squared,
        peak_model             = _model_label(model, _resolve_baseline(baseline)[0]),
        converged_mask         = converged,
        method                 = method,
        n_bootstrap            = n_bootstrap if method == "bootstrap" else None,
    )


# ---------------------------------------------------------------------------
# Multi-dipole peak tracking and extraction
# ---------------------------------------------------------------------------

@dataclass
class PeakTrack:
    """
    Peak energy and amplitude tracked across every sweep point.

    Produced by :func:`track_peak_energies`.  The tracked peaks can be
    inspected and plotted on their own (peak energy vs. any sweep
    coordinate), or passed to :func:`extract_dipole_lengths` to fit dipole
    moments over user-defined field ranges.

    Attributes
    ----------
    sweep_values    : np.ndarray
        Sweep coordinate at each sweep point, shape ``(n_sweeps,)``.
        Contains whatever the scan's declared sweep axis is (electric
        field, power, voltage, or ``arange`` for an index sweep).
    sweep_label     : str
        Display label for the sweep axis (e.g. ``r"$E_F$"``,
        ``"Power"``, ``"Sweep index"``).
    sweep_unit      : str
        Unit string for the sweep axis (e.g. ``"mV/nm"``, ``"µW"``,
        ``""``).
    peak_energies   : np.ndarray
        Peak energy at each sweep point (eV).
    peak_amplitudes : np.ndarray
        Peak intensity at each sweep point (detector counts).
        For ``method="argmax"`` this is the spectral maximum within the
        detection window.
    peak_errors     : np.ndarray
        1-sigma uncertainty on each peak energy (eV).
        All ``NaN`` when ``method="argmax"`` (no formal error bars).
    converged       : np.ndarray of bool
        ``True`` for every sweep point where the peak was found.
        All ``True`` when ``method="argmax"``.
    method          : str
        Tracking method used: ``"argmax"``.
    """
    sweep_values    : np.ndarray
    sweep_label     : str
    sweep_unit      : str
    peak_energies   : np.ndarray
    peak_amplitudes : np.ndarray
    peak_errors     : np.ndarray
    converged       : np.ndarray
    method          : str

    def __repr__(self) -> str:
        n = len(self.peak_energies)
        lo = float(np.nanmin(self.peak_energies))
        hi = float(np.nanmax(self.peak_energies))
        if self.sweep_unit:
            sweep_str = f"{self.sweep_label} ({self.sweep_unit}), {n} points"
        else:
            sweep_str = f"{self.sweep_label}, {n} points"
        return (
            f"PeakTrack\n"
            f"  Method        : {self.method}\n"
            f"  Sweep axis    : {sweep_str}\n"
            f"  Peak range    : {lo:.4f} – {hi:.4f} eV\n"
            f"  Converged     : {self.converged.sum()} / {n}"
        )


def track_peak_energies(
    scan,
    method  : str   = "argmax",
    x_range : tuple = None,
) -> PeakTrack:
    """
    Track the peak emission energy at every sweep point.

    Returns a :class:`PeakTrack` that can be plotted on its own
    (to visualise peak shifts against the scan's sweep coordinate) or
    passed to :func:`extract_dipole_lengths` for multi-range dipole
    extraction.

    Parameters
    ----------
    scan : AttoCubeSpectralSweep or BigTableSpectralSweep
        Any sweep object with ``best_energy_spectra`` (shape
        ``(n_pixels, n_sweeps)``), ``energy`` (shape ``(n_pixels,)``),
        and the sweep-axis properties ``sweep_axis``, ``sweep_label``,
        ``sweep_unit``.
    method : {"argmax"}
        ``"argmax"``
            Energy of maximum intensity at each sweep point.
            Simple and robust when the peak shifts over a wide range
            (e.g. hybridised bilayer systems).  No per-point error bars.
    x_range : tuple of (E_min, E_max) in eV, optional
        Restrict the energy window used for peak detection.
        Useful to avoid locking onto a laser line or substrate feature.

    Returns
    -------
    PeakTrack

    Raises
    ------
    ValueError
        If *method* is not recognised, the scan lacks the required
        attributes, or *x_range* selects no pixel.
    NotImplementedError
        If ``method="fit"`` (reserved for a future sliding-window
        lineshape tracker).
    """
    _METHODS = ("argmax",)
    if method == "fit":
        raise NotImplementedError(
            "method='fit' (sliding-window lineshape tracking) is not yet "
            "implemented. Use method='argmax'."
        )
    if method not in _METHODS:
        raise ValueError(
            f"method={method!r} is not recognised. Choose from {_METHODS}."
        )

    for attr in ("best_energy_spectra", "energy"):
        if not hasattr(scan, attr):
            raise ValueError(
                f"scan has no {attr!r} attribute. Expected an "
                f"AttoCubeSpectralSweep, BigTableSpectralSweep, or similar."
            )

    # (n_pixels, n_sweeps) — each column is one spectrum.
    spectra = scan.best_energy_spectra
    energy  = scan.energy

    if x_range is not None:
        lo, hi = sorted(x_range)
        pixel_mask = (energy >= lo) & (energy <= hi)
        if pixel_mask.sum() == 0:
            raise ValueError(
                f"x_range=({lo}, {hi}) selects no pixel on the energy axis "
                f"(spans {float(energy.min()):.4f} to "
                f"{float(energy.max()):.4f} eV)."
            )
        spectra = spectra[pixel_mask, :]
        energy  = energy[pixel_mask]

    # argmax along the pixel axis for each sweep column.
    idx             = np.argmax(spectra, axis=0)
    peak_energies   = energy[idx]
    peak_amplitudes = spectra[idx, np.arange(spectra.shape[1])]
    peak_errors     = np.full_like(peak_energies, np.nan)
    converged       = np.ones(len(peak_energies), dtype=bool)

    sweep_values = scan.sweep_axis
    sweep_label  = scan.sweep_label
    sweep_unit   = scan.sweep_unit

    return PeakTrack(
        sweep_values    = sweep_values,
        sweep_label     = sweep_label,
        sweep_unit      = sweep_unit,
        peak_energies   = peak_energies,
        peak_amplitudes = peak_amplitudes,
        peak_errors     = peak_errors,
        converged       = converged,
        method          = method,
    )


def _resolve_sweep_ranges(
    sweep_values : np.ndarray,
    coord_ranges : list,
    index_ranges : list,
    n_sweeps     : int,
    unit         : str = "",
    param_name   : str = "coord_ranges",
) -> list:
    """
    Build boolean masks for multi-range extraction.

    Accepts either *coord_ranges* (sweep-coordinate windows) or
    *index_ranges* (sweep-index windows), validates each, and warns when
    ranges overlap.

    Parameters
    ----------
    sweep_values : np.ndarray
        Sweep coordinate at each sweep point.  Required when
        *coord_ranges* is supplied; ignored for *index_ranges*.
    coord_ranges : list, optional
        Coordinate-value ranges, both endpoints inclusive.  Each element
        is either a single ``(lo, hi)`` tuple or a list of ``(lo, hi)``
        tuples whose masks are OR'd together — use the list form to
        exclude isolated bad points from a regime without splitting it
        into separate segments.
    index_ranges : list, optional
        Sweep-index ranges, both endpoints inclusive.  Accepts single
        ``(start, stop)`` tuples or lists of tuples, same as
        *coord_ranges*.
    n_sweeps : int
    unit : str
        Unit string for warning messages (e.g. ``"mV/nm"``).
    param_name : str
        The caller's external parameter name for the coordinate ranges,
        used in error and warning messages (e.g. ``"ef_ranges"``).

    Returns
    -------
    list of np.ndarray
        One boolean mask per range, shape ``(n_sweeps,)``.
    """
    if (coord_ranges is None) == (index_ranges is None):
        raise ValueError(
            f"Supply exactly one of {param_name} or index_ranges, not "
            + ("both." if coord_ranges is not None else "neither.")
        )

    unit_suffix = f" {unit}" if unit else ""
    masks = []

    if coord_ranges is not None:
        sv = sweep_values
        finite_sv = sv[np.isfinite(sv)]
        if finite_sv.size > 1:
            step   = float(np.median(np.abs(np.diff(np.sort(finite_sv)))))
            travel = float(np.ptp(finite_sv))
        else:
            step   = 0.0
            travel = 0.0
        threshold = max(0.5 * step, 0.001 * travel)

        for i, element in enumerate(coord_ranges):
            sub_ranges = (
                element if isinstance(element, list) else [element]
            )
            mask = np.zeros(len(sv), dtype=bool)
            for lo, hi in sub_ranges:
                lo, hi = float(lo), float(hi)
                if lo > hi:
                    lo, hi = hi, lo
                sub_mask = (sv >= lo) & (sv <= hi)

                if sub_mask.sum() == 0:
                    below = sv[sv < lo]
                    above = sv[sv > hi]
                    parts = []
                    if below.size:
                        parts.append(f"nearest below: {float(below.max()):.6g}")
                    if above.size:
                        parts.append(f"nearest above: {float(above.min()):.6g}")
                    near_str = "; ".join(parts) if parts else "no finite values"
                    raise ValueError(
                        f"{param_name}[{i}] sub-range ({lo:.6g}, {hi:.6g}) "
                        f"selects no sweep point. {near_str}."
                    )

                sv_in = sv[sub_mask]
                actual_lo = float(sv_in.min())
                actual_hi = float(sv_in.max())
                if abs(actual_lo - lo) > threshold:
                    warnings.warn(
                        f"{param_name}[{i}] lower bound {lo:.6g}{unit_suffix}: "
                        f"nearest sweep point inside the range is at "
                        f"{actual_lo:.6g}{unit_suffix} "
                        f"({abs(actual_lo - lo):.4g}{unit_suffix} away).",
                        UserWarning, stacklevel=3,
                    )
                if abs(actual_hi - hi) > threshold:
                    warnings.warn(
                        f"{param_name}[{i}] upper bound {hi:.6g}{unit_suffix}: "
                        f"nearest sweep point inside the range is at "
                        f"{actual_hi:.6g}{unit_suffix} "
                        f"({abs(actual_hi - hi):.4g}{unit_suffix} away).",
                        UserWarning, stacklevel=3,
                    )
                mask |= sub_mask

            n_in = int(mask.sum())
            if n_in < 2:
                raise ValueError(
                    f"{param_name}[{i}] selects only {n_in} point(s); need "
                    f"at least 2 for a linear fit."
                )

            masks.append(mask)

    else:  # index_ranges
        for i, element in enumerate(index_ranges):
            sub_ranges = (
                element if isinstance(element, list) else [element]
            )
            mask = np.zeros(n_sweeps, dtype=bool)
            for start, stop in sub_ranges:
                start, stop = int(start), int(stop)
                if start > stop:
                    start, stop = stop, start
                if start < 0 or stop >= n_sweeps:
                    raise ValueError(
                        f"index_ranges[{i}] sub-range ({start}, {stop}) "
                        f"is out of bounds for {n_sweeps} sweep points "
                        f"(valid: 0 to {n_sweeps - 1})."
                    )
                mask[start:stop + 1] = True

            n_in = int(mask.sum())
            if n_in < 2:
                raise ValueError(
                    f"index_ranges[{i}] selects only {n_in} point(s); "
                    f"need at least 2 for a linear fit."
                )
            masks.append(mask)

    # Check for overlap between any pair of ranges.
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            overlap = np.sum(masks[i] & masks[j])
            if overlap > 0:
                warnings.warn(
                    f"Ranges {i} and {j} overlap on {overlap} index(es).",
                    UserWarning, stacklevel=3,
                )

    return masks


@dataclass
class _LinearSegment:
    """Raw per-range linear fit result (private, converted by public wrappers)."""
    slope         : float
    slope_err     : float
    intercept     : float
    intercept_err : float
    r_squared     : float
    mask          : np.ndarray


def _extract_linear_fits(
    x_values     : np.ndarray,
    y_values     : np.ndarray,
    y_errors     : np.ndarray,
    coord_ranges,
    index_ranges,
    n_sweeps     : int,
    fit_method   : str = "wls",
    n_bootstrap  : int = 2000,
    rng          = None,
    unit         : str = "",
    param_name   : str = "coord_ranges",
    stacklevel   : int = 3,
) -> list:
    """
    Fit a line within each coordinate or index range.

    This is the mechanical core shared by every public extraction
    function.  It resolves range masks, dispatches to the linear
    fitters, and returns raw slopes — the physics interpretation
    (dipole length, power-law exponent, shift rate) belongs to the
    caller.

    Parameters
    ----------
    x_values : np.ndarray
        Independent variable at each sweep point (field, power, etc.).
    y_values : np.ndarray
        Dependent variable at each sweep point (energy, log-amplitude, etc.).
    y_errors : np.ndarray
        1-sigma on *y_values* (may be all NaN).
    coord_ranges, index_ranges : list or None
        Passed to :func:`_resolve_sweep_ranges`.
    n_sweeps : int
    fit_method : {"wls", "minmax", "bootstrap"}
    n_bootstrap : int
    rng : np.random.Generator, optional
    unit : str
        Unit for warning messages.
    param_name : str
        External parameter name for error messages.
    stacklevel : int
        Extra stack depth for warnings (callers add one per wrapper).

    Returns
    -------
    list of _LinearSegment
    """
    _FIT_METHODS = ("wls", "minmax", "bootstrap")
    if fit_method not in _FIT_METHODS:
        raise ValueError(
            f"fit_method={fit_method!r} is not recognised. "
            f"Choose from {_FIT_METHODS}."
        )

    masks = _resolve_sweep_ranges(
        x_values, coord_ranges, index_ranges, n_sweeps,
        unit=unit, param_name=param_name,
    )

    segments = []
    for i, mask in enumerate(masks):
        x_fit   = x_values[mask]
        y_fit   = y_values[mask]
        sig_fit = y_errors[mask]

        try:
            if fit_method == "wls":
                slope, slope_err, intercept, intercept_err = _dipole_wls(
                    x_fit, y_fit, sig_fit,
                )
            elif fit_method == "minmax":
                slope, slope_err, intercept, intercept_err = _dipole_minmax(
                    x_fit, y_fit, sig_fit,
                )
            else:  # bootstrap
                slope, slope_err, intercept, intercept_err = _dipole_bootstrap(
                    x_fit, y_fit, sig_fit,
                    n_bootstrap=n_bootstrap, rng=rng,
                )

            r_squared = _r_squared(y_fit, slope * x_fit + intercept)
        except Exception as exc:
            warnings.warn(
                f"Linear fit failed for range {i}: {exc}",
                UserWarning, stacklevel=stacklevel,
            )
            slope = slope_err = intercept = intercept_err = np.nan
            r_squared = np.nan

        segments.append(_LinearSegment(
            slope=slope, slope_err=slope_err,
            intercept=intercept, intercept_err=intercept_err,
            r_squared=r_squared, mask=mask,
        ))

    return segments


@dataclass
class MultiDipoleResult:
    """
    Result of a multi-range dipole extraction.

    Holds the shared :class:`PeakTrack` and one :class:`DipoleResult` per
    field range.  Each segment's ``DipoleResult`` carries the full peak
    arrays; its ``converged_mask`` is the boolean mask selecting the sweep
    points that belong to that range (not a convergence mask in the
    lineshape-fitting sense).

    Attributes
    ----------
    track        : PeakTrack
        The peak track shared by all segments.
    segments     : list of DipoleResult
        One per range, in the same order as the ranges were supplied.
    ef_ranges    : list of tuple or None
        The field-value ranges used, or ``None`` when *index_ranges* were
        used instead.
    index_ranges : list of tuple or None
        The sweep-index ranges used, or ``None`` when *ef_ranges* were
        used instead.
    """
    track        : PeakTrack
    segments     : list
    ef_ranges    : list
    index_ranges : list

    def __repr__(self) -> str:
        n = len(self.segments)
        lines = [
            f"MultiDipoleResult ({n} segment{'s' if n != 1 else ''}, "
            f"track method: {self.track.method})"
        ]
        for i, seg in enumerate(self.segments):
            if self.ef_ranges is not None:
                lo, hi = self.ef_ranges[i]
                tag = f"ef: {lo:.4g} to {hi:.4g} mV/nm"
            else:
                lo, hi = self.index_ranges[i]
                tag = f"idx: {lo}–{hi}"

            if np.isfinite(seg.dipole_length):
                lines.append(
                    f"  Segment {i} : "
                    f"d = {seg.dipole_length:.4f} ± "
                    f"{seg.dipole_length_err:.4f} nm  "
                    f"({seg.dipole_length_angstrom:.2f} Å)  "
                    f"R² = {seg.r_squared:.4f}  [{tag}]"
                )
            else:
                lines.append(f"  Segment {i} : fit failed  [{tag}]")
        return "\n".join(lines)


def extract_dipole_lengths(
    source,
    ef_ranges    : list  = None,
    index_ranges : list  = None,
    x_range      : tuple = None,
    track_method : str   = "argmax",
    fit_method   : str   = "wls",
    n_bootstrap  : int   = 2000,
    rng          : np.random.Generator = None,
) -> MultiDipoleResult:
    """
    Extract dipole lengths from multiple linear regimes in a field sweep.

    This is the multi-range counterpart of :func:`extract_dipole_length`.
    The peak emission energy is tracked across the full sweep (via
    :func:`track_peak_energies`), and a separate linear fit is performed
    within each user-defined field range to extract a dipole moment per
    regime.

    Parameters
    ----------
    source : scan or PeakTrack
        Either a sweep object (``AttoCubeSpectralSweep``,
        ``BigTableSpectralSweep``, or similar) or an already-computed
        :class:`PeakTrack`.  When a scan is passed, *x_range* and
        *track_method* are forwarded to :func:`track_peak_energies`;
        when a ``PeakTrack`` is passed they are ignored.
    ef_ranges : list of (lo, hi), optional
        Field-value ranges in mV/nm, both endpoints inclusive.
        Each range defines one dipole regime.  Supply this **or**
        *index_ranges*, not both.
    index_ranges : list of (start, stop), optional
        Sweep-index ranges, both endpoints inclusive.
    x_range : tuple of (E_min, E_max) in eV, optional
        Energy window for peak detection, forwarded to
        :func:`track_peak_energies` when *source* is a scan.
    track_method : {"argmax"}
        Peak-tracking method, forwarded to :func:`track_peak_energies`
        when *source* is a scan.
    fit_method : {"wls", "minmax", "bootstrap"}
        Linear-fit method — see :func:`extract_dipole_length` for
        descriptions.
    n_bootstrap : int
        Number of bootstrap iterations (only for ``fit_method="bootstrap"``).
    rng : np.random.Generator, optional
        Random number generator for reproducibility with bootstrap.

    Returns
    -------
    MultiDipoleResult

    Raises
    ------
    ValueError
        If the sweep is not electric field (unit ``"mV/nm"``), both or
        neither range arguments are given, a range selects fewer than 2
        points, or *fit_method* is not recognised.

    Examples
    --------
    >>> track = track_peak_energies(scan, x_range=(1.43, 1.57))
    >>> result = extract_dipole_lengths(
    ...     track,
    ...     ef_ranges=[(-100, -60), (-60, 60), (60, 100)],
    ... )
    >>> print(result)

    >>> # Or as a one-liner from a scan:
    >>> result = extract_dipole_lengths(
    ...     scan,
    ...     ef_ranges=[(-100, -60), (-60, 60), (60, 100)],
    ...     x_range=(1.43, 1.57),
    ... )
    """
    # --- Resolve or build the peak track ---
    if isinstance(source, PeakTrack):
        track = source
    else:
        track = track_peak_energies(
            source, method=track_method, x_range=x_range,
        )

    if track.sweep_values is None or track.sweep_unit != "mV/nm":
        raise ValueError(
            f"extract_dipole_lengths requires an electric-field sweep "
            f"(unit 'mV/nm'), but the track's sweep is "
            f"{track.sweep_label!r} ({track.sweep_unit!r}). "
            f"Load the scan with sweep='electric_field'."
        )

    # --- Linear fits per range ---
    fits = _extract_linear_fits(
        x_values=track.sweep_values,
        y_values=track.peak_energies,
        y_errors=track.peak_errors,
        coord_ranges=ef_ranges,
        index_ranges=index_ranges,
        n_sweeps=len(track.peak_energies),
        fit_method=fit_method,
        n_bootstrap=n_bootstrap,
        rng=rng,
        unit=track.sweep_unit,
        param_name="ef_ranges",
    )

    # --- Convert raw slopes to dipole lengths ---
    segments = []
    for seg in fits:
        dipole_length = abs(seg.slope) * 1000.0
        dipole_length_err = (
            abs(seg.slope_err) * 1000.0 if np.isfinite(seg.slope_err)
            else np.nan
        )
        segments.append(DipoleResult(
            ef                     = track.sweep_values,
            peak_energies          = track.peak_energies,
            peak_errors            = track.peak_errors,
            slope                  = seg.slope,
            slope_err              = seg.slope_err,
            intercept              = seg.intercept,
            intercept_err          = seg.intercept_err,
            dipole_length          = dipole_length,
            dipole_length_err      = dipole_length_err,
            dipole_length_angstrom = (
                dipole_length * 10.0
                if np.isfinite(dipole_length) else np.nan
            ),
            r_squared              = seg.r_squared,
            peak_model             = track.method,
            converged_mask         = seg.mask,
            method                 = fit_method,
            n_bootstrap            = (
                n_bootstrap if fit_method == "bootstrap" else None
            ),
        ))

    return MultiDipoleResult(
        track        = track,
        segments     = segments,
        ef_ranges    = [tuple(r) for r in ef_ranges] if ef_ranges is not None else None,
        index_ranges = [tuple(r) for r in index_ranges] if index_ranges is not None else None,
    )