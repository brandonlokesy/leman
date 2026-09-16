# leman/diffusion.py
"""
Exciton diffusion cloud analysis.

Provides routines to detect and characterise the exciton diffusion cloud
in real-space PL images: thresholding, boundary tracing, centroid extraction,
and real-space coordinate conversion.

Note: "diffusion" here refers to the spatial spread of the PL cloud as
imaged in real space. The underlying mechanism may include both diffusion
and drift (see doi:10.1038/s41566-023-01198-w); separating them requires
modelling the spatial profiles, which is handled in fitting.py.

The companion plotting helpers live in plotting.py:
  - plot_diffusion_cloud()          single image with boundary + centroid
  - plot_centroid_trajectory()      centroid x/y vs. an external variable
  - DiffusionCloudPanel             AnimationPanel subclass for animate_panels()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage import filters, measure
from skimage.morphology import label
from os import PathLike
from pathlib import Path


from . import processing

from leman.loaders import _ACImg

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DiffusionResult:
    """
    Analysis result for a single real-space PL image.

    Produced by :func:`analyse_diffusion_cloud`.  All coordinate values are
    in **pixel** units unless a *pixel_scale* was supplied to the analysis
    function, in which case the ``x_real`` / ``y_real`` fields are also
    populated.

    Attributes
    ----------
    x_pixel, y_pixel : float
        Intensity-weighted centroid in pixel coordinates.
        ``(0, 0)`` is the top-left corner of the image (``origin="corner"``
        convention).  Use :meth:`centroid_px` / :meth:`centroid_real` for a
        convenient ``(x, y)`` tuple.
    x_real, y_real : float or None
        Centroid in real-space coordinates, in :attr:`scale_units`.
        ``None`` when no scale was supplied.
    area_px2 : float
        Effective cloud area in px².
    area_real : float or None
        Effective cloud area in :attr:`scale_units` squared (``pixel_scale²``).
        ``None`` when no scale was supplied.
    contours : list of np.ndarray
        Boundary contours as ``(N, 2)`` arrays of ``(row, col)`` coordinates,
        one per closed region.  Typically a single contour enclosing the cloud.
    mask : np.ndarray of bool, shape (H, W)
        Binary mask: ``True`` where the image exceeds the threshold.
    image : np.ndarray, shape (H, W)
        The full-frame image actually analysed: background-subtracted when
        *bg_region* was supplied, but before the ROI mask and smoothing
        applied internally for thresholding. This is what a caller should
        display alongside :attr:`mask` / :attr:`contours` — re-deriving the
        image from whatever was originally passed to
        :func:`analyse_diffusion_cloud` risks re-applying a background
        subtraction that already happened here.
    threshold : float
        Threshold value that was actually applied (after Otsu / manual
        fraction conversion).
    pixel_scale : float or None
        µm-per-pixel scale used for conversion (echoed from the call).
    origin : str
        Origin convention used for real-space conversion
        (``"corner"`` | ``"center"`` | ``"image_center"``).
    scale_units : str
        Unit *pixel_scale* was given in, and therefore the unit of
        :attr:`x_real` / :attr:`y_real`; :attr:`area_real` is in its square.
        Carried so that a reader of the result — ``__repr__``, or
        :func:`~leman.plotting.plot_centroid_trajectory`'s axis
        label — can say what the real-space numbers are in. The scale is
        typically calibrated per session against a known laser-spot size, so
        the unit is the caller's choice rather than a property of the
        instrument.
    """

    x_pixel    : float
    y_pixel    : float
    x_real     : float | None
    y_real     : float | None
    area_px2   : float
    area_real  : float | None
    contours   : list
    mask       : np.ndarray
    image      : np.ndarray
    threshold  : float
    pixel_scale: float | None = None
    origin     : str          = "corner"
    roi        : tuple | None = None   # (row_slice, col_slice), echoed from the call
    # Pairs with pixel_scale: the unit that scale is expressed in.  Appended
    # rather than placed next to it, because a dataclass can be constructed
    # positionally and inserting a field silently changes what such a call means.
    scale_units: str          = "µm"

    @property
    def centroid_px(self) -> tuple[float, float]:
        """``(x_pixel, y_pixel)`` as a plain tuple."""
        return (self.x_pixel, self.y_pixel)

    @property
    def centroid_real(self) -> tuple[float | None, float | None]:
        """``(x_real, y_real)`` as a plain tuple (``None`` if no scale)."""
        return (self.x_real, self.y_real)

    def __repr__(self) -> str:
        unit = self.scale_units
        # (label, value) pairs first, so the colons align on the widest label
        # rather than on a hardcoded width -- the real-space labels carry a
        # caller-supplied unit and so have no fixed length.  A real-space row is
        # present only when the conversion actually happened, which is what stops
        # a unit being printed beside a number that is still in pixels.
        rows = [("Centroid (px)", f"({self.x_pixel:.2f}, {self.y_pixel:.2f})")]
        if self.x_real is not None:
            rows.append((
                f"Centroid ({unit})",
                f"({self.x_real:.4g}, {self.y_real:.4g})  [origin='{self.origin}']",
            ))
        rows.append(("Area", f"{self.area_px2:.1f} px²"))
        if self.area_real is not None:
            rows.append((f"Area ({unit}²)", f"{self.area_real:.4g}"))
        rows.append(("Threshold", f"{self.threshold:.4g}"))
        rows.append(("Contours",  f"{len(self.contours)}"))

        width = max(len(label) for label, _ in rows)
        return "\n".join(
            ["DiffusionResult"]
            + [f"  {label:<{width}} : {value}" for label, value in rows]
        )


@dataclass
class DiffusionSequenceResult:
    """
    Analysis results for a sequence of real-space PL images.

    Produced by :func:`analyse_diffusion_sequence`.  Provides convenient
    array access to per-frame scalar quantities so that they can be directly
    passed to a plotting function alongside an external variable axis (time,
    power, electric field, …).

    Attributes
    ----------
    frames : list of DiffusionResult
        Per-frame results, one entry per image in the sequence.
    var_array : np.ndarray or None
        External variable axis (e.g. power in µW, time in s, E-field in
        mV/nm).  Echoed from the call for convenience.
    var_label : str
        Human-readable label for *var_array* (e.g. ``"Power"``).
    var_units : str
        Units string for *var_array* (e.g. ``"µW"``).
    """

    frames    : list            # list[DiffusionResult]
    var_array : np.ndarray | None = None
    var_label : str               = ""
    var_units : str               = ""

    # --- Convenience array accessors --------------------------------------

    @property
    def x_pixel(self) -> np.ndarray:
        """Centroid x in pixel coordinates, shape ``(n_frames,)``."""
        return np.array([f.x_pixel for f in self.frames])

    @property
    def y_pixel(self) -> np.ndarray:
        """Centroid y in pixel coordinates, shape ``(n_frames,)``."""
        return np.array([f.y_pixel for f in self.frames])

    @property
    def x_real(self) -> np.ndarray | None:
        """Centroid x in real space, shape ``(n_frames,)``, or ``None``."""
        vals = [f.x_real for f in self.frames]
        return np.array(vals) if vals[0] is not None else None

    @property
    def y_real(self) -> np.ndarray | None:
        """Centroid y in real space, shape ``(n_frames,)``, or ``None``."""
        vals = [f.y_real for f in self.frames]
        return np.array(vals) if vals[0] is not None else None

    @property
    def area_px2(self) -> np.ndarray:
        """Effective cloud area in px², shape ``(n_frames,)``."""
        return np.array([f.area_px2 for f in self.frames])

    @property
    def area_real(self) -> np.ndarray | None:
        """Effective cloud area in real-space units², or ``None``."""
        vals = [f.area_real for f in self.frames]
        return np.array(vals) if vals[0] is not None else None

    @property
    def scale_units(self) -> str:
        """
        Unit of :attr:`x_real` / :attr:`y_real`; :attr:`area_real` is its square.

        Read off the first frame: one call analyses the whole sequence, so every
        frame carries the same unit.
        """
        return self.frames[0].scale_units

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        return (
            f"DiffusionSequenceResult — {self.n_frames} frames\n"
            f"  x_pixel range : {self.x_pixel.min():.2f} – {self.x_pixel.max():.2f}\n"
            f"  y_pixel range : {self.y_pixel.min():.2f} – {self.y_pixel.max():.2f}\n"
            f"  area range    : {self.area_px2.min():.1f} – {self.area_px2.max():.1f} px²"
        )


# ---------------------------------------------------------------------------
# Core analysis: single image
# ---------------------------------------------------------------------------

def analyse_diffusion_cloud(
    image         : np.ndarray | str | PathLike | "_ACImg",
    threshold     : float | str = "1/e",
    smooth_sigma  : float       = 0.0,
    keep_largest  : bool        = False,
    roi           : tuple[slice, slice] | None = None,
    bg_region     : tuple[slice, slice] | None = None,
    bg_stat       : str         = "median",
    pixel_scale   : float | None = None,
    scale_units   : str         = "µm",
    origin        : str         = "corner",
) -> DiffusionResult:
    """
    Detect the exciton diffusion cloud boundary and centroid in a single image.

    The routine mirrors the encirclement approach used in the original
    ``analysis.m`` MATLAB script: smooth → threshold → trace boundary →
    intensity-weighted centroid.

    Parameters
    ----------
    image : np.ndarray, str, pathlib.Path, or _ACImg
        Background-subtracted PL image. Accepted inputs are:

        - a 2D NumPy array with shape (H, W),
        - a path to a CSV file, or
        - an instance of _ACImg or any of its subclasses.

        Image data should be numeric (float or int).
    threshold : float or ``"otsu"``
        Threshold mode:
        - 1/e threshold (default) — threshold = image.max() / e.
        - ``"otsu"`` — automatic Otsu threshold.
        - ``float`` — fraction of the image maximum, e.g. ``0.15`` means
          15 % of ``image.max()``.
    smooth_sigma : float
        Standard deviation of the Gaussian pre-smoothing kernel in pixels.
        Set ``0`` to skip smoothing.
    keep_largest : bool
        If ``True``, retain only the largest connected region above the
        threshold (suppresses noise blobs).
    pixel_scale : float, optional
        Physical size of one pixel in real-space units (e.g. µm/px).
        When supplied, ``x_real``, ``y_real``, and ``area_real`` are
        populated in the returned :class:`DiffusionResult`.
    scale_units : str
        Unit *pixel_scale* is given in. Carried on the result, which is what
        lets its ``__repr__`` and
        :func:`~leman.plotting.plot_centroid_trajectory` say what
        the real-space numbers are in. Ignored when *pixel_scale* is ``None``,
        since nothing is converted.
    origin : {``"corner"``, ``"center"``, ``"image_center"``}
        Origin convention for real-space coordinates:
        - ``"corner"``       — ``(0, 0)`` is the top-left pixel.
        - ``"center"``       — ``(0, 0)`` is the cloud centroid itself.
        - ``"image_center"`` — ``(0, 0)`` is the dead-centre of the image.

    Returns
    -------
    DiffusionResult
    """
    img = _load_image(image)

    # --- 1. Background subtraction ----------------------------------------
    if bg_region is not None:
        img = processing._apply_bg_region(img, bg_region, bg_stat)

    # --- 2. ROI crop BEFORE smoothing and thresholding --------------------
    # The MATLAB reference (find_em1_perimeter_custom) masks the image *before*
    # computing the maximum and applying the 1/e threshold.  Doing it after
    # means the global maximum (which may sit outside the ROI) drives the
    # threshold level, so the cloud region is either missed entirely or the
    # threshold is set far too high relative to the in-ROI signal.
    img_roi = img  # full image by default
    if roi is not None:
        roi_mask_full = np.zeros(img.shape, dtype=bool)
        roi_mask_full[roi] = True
        img_roi = np.where(roi_mask_full, img, 0.0)

    # --- 3. Smoothing (on the ROI-masked image) ---------------------------
    img_processed = (
        filters.gaussian(img_roi, sigma=smooth_sigma)
        if smooth_sigma > 0
        else img_roi.copy()
    )

    # Zero any smoothed bleed outside the ROI back out so the peak/threshold
    # computation only sees signal inside the ROI.
    if roi is not None:
        img_processed = np.where(roi_mask_full, img_processed, 0.0)

    # --- 4. Threshold → binary mask ---------------------------------------
    # All threshold modes compare against the *in-ROI* maximum, matching the
    # MATLAB approach: [M,I]=max(im); [maxlaser,I2]=max(M); threshold = maxlaser/e.
    peak = img_processed.max()

    if threshold == "1/e":
        threshold_val = peak / np.e

    elif threshold == "otsu":
        # Otsu on the full array is dominated by zeros outside the ROI.
        # Restrict to positive (in-ROI) pixels so the histogram is meaningful.
        in_roi_vals = img_processed[img_processed > 0]
        if in_roi_vals.size > 1:
            threshold_val = filters.threshold_otsu(in_roi_vals)
        else:
            threshold_val = peak / np.e   # fall back to 1/e for degenerate ROI

    elif isinstance(threshold, (int, float)):
        threshold_val = float(threshold) * peak

    else:
        raise ValueError(
            "threshold must be '1/e', 'otsu', or a float, "
            f"got {threshold!r}."
        )

    mask = img_processed >= threshold_val
    threshold = threshold_val  # overwrite name so it is stored in the result

    # --- 5. Keep largest connected region (optional) ----------------------
    if keep_largest:
        mask = _largest_region(mask)

    contours = measure.find_contours(mask.astype(float), level=0.5)
    # Use the background-subtracted full image for intensity weighting so
    # that the centroid reflects true signal, not the zeroed-out ROI surrounds.
    cx_px, cy_px = _intensity_centroid(img, mask)
    area_px2 = _binary_area(mask)

    x_real = y_real = area_real = None
    if pixel_scale is not None:
        x_real, y_real = _pixel_to_realspace(
            cx_px, cy_px,
            scale=pixel_scale,
            origin=origin,
            image_shape=img.shape,
            centroid_px=(cx_px, cy_px),
        )
        area_real = area_px2 * pixel_scale ** 2

    return DiffusionResult(
        x_pixel    = cx_px,
        y_pixel    = cy_px,
        x_real     = x_real,
        y_real     = y_real,
        area_px2   = area_px2,
        area_real  = area_real,
        contours   = contours,
        mask       = mask,
        image      = img,
        threshold  = threshold,
        pixel_scale= pixel_scale,
        origin     = origin,
        roi        = roi,
        scale_units= scale_units,
    )


# ---------------------------------------------------------------------------
# Core analysis: image sequence
# ---------------------------------------------------------------------------

def analyse_diffusion_sequence(
    frames        : list,
    threshold     : float | str = "1/e",
    smooth_sigma  : float       = 1.0,
    keep_largest  : bool        = True,
    roi           : tuple[slice, slice] = None,
    bg_region     : tuple[slice, slice] = None,
    bg_stat       : str         = "median",
    pixel_scale   : float       = None,
    scale_units   : str         = "µm",
    origin        : str         = "corner",
    var_array                   = None,
    var_label     : str         = "",
    var_units     : str         = "",
) -> DiffusionSequenceResult:
    """
    Run :func:`analyse_diffusion_cloud` on every frame of a sequence.

    The *frames* list can be either a list of ``np.ndarray`` images, or any
    object exposing ``load_frame(idx)`` and ``n_frames`` (i.e. an
    :class:`~leman.loaders.ACImgSweep` instance).

    Parameters
    ----------
    frames : list of np.ndarray or ACImgSweep
        Image sequence.
    threshold, smooth_sigma, keep_largest, pixel_scale, scale_units, origin
        Forwarded to :func:`analyse_diffusion_cloud` for every frame.
        Threshold mode:
        - 1/e threshold (default) — threshold = image.max() / e.
        - ``"otsu"`` — automatic Otsu threshold.
        - ``float`` — fraction of the image maximum, e.g. ``0.15`` means
          15 % of ``image.max()``.
    var_array : array-like, optional
        External variable axis (time, power, E-field, …), one value per
        frame.  Stored in the result for downstream plotting.
    var_label : str
        Human-readable label for *var_array* (e.g. ``"Power"``).
    var_units : str
        Unit string for *var_array* (e.g. ``"µW"``).

    Returns
    -------
    DiffusionSequenceResult
    """
    # Support both a raw list of arrays and a scan object.
    # ACImgSweep.load_frame() returns the raw numeric array —
    # background subtraction has NOT happened yet — so we can pass bg_region
    # straight through to analyse_diffusion_cloud.
    #
    # If the caller passes a list of _ACImg instances that were
    # constructed with a bg_region, _load_image will use img_raw (the
    # un-subtracted array) and analyse_diffusion_cloud will apply bg_region
    # once.  There is therefore no double-subtraction risk in either path.
    if hasattr(frames, "load_frame") and hasattr(frames, "n_frames"):
        imgs = [frames.load_frame(i) for i in range(frames.n_frames)]
    else:
        imgs = list(frames)

    results = [
        analyse_diffusion_cloud(
            img,
            threshold    = threshold,
            smooth_sigma = smooth_sigma,
            keep_largest = keep_largest,
            roi          = roi,
            bg_region    = bg_region,
            bg_stat      = bg_stat,
            pixel_scale  = pixel_scale,
            scale_units  = scale_units,
            origin       = origin,
        )
        for img in imgs
    ]

    return DiffusionSequenceResult(
        frames    = results,
        var_array = np.asarray(var_array) if var_array is not None else None,
        var_label = var_label,
        var_units = var_units,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _largest_region(mask: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest connected component."""
    labeled = label(mask)
    if labeled.max() == 0:
        return mask                # nothing above threshold; return as-is
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0                   # exclude background
    return labeled == sizes.argmax()


def _intensity_centroid(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centre of mass of the masked region."""
    weighted = img * mask
    cy, cx = ndimage.center_of_mass(weighted)
    return float(cx), float(cy)   # (x=col, y=row)


def _pixel_to_realspace(
    x_px         : float,
    y_px         : float,
    scale        : float,
    origin       : str,
    image_shape  : tuple,
    centroid_px  : tuple,
) -> tuple[float, float]:
    """
    Convert a pixel coordinate to real space.

    Parameters
    ----------
    x_px, y_px    : pixel coordinates (col, row)
    scale         : µm (or other unit) per pixel
    origin        : ``"corner"`` | ``"center"`` | ``"image_center"``
    image_shape   : ``(n_rows, n_cols)``
    centroid_px   : ``(cx, cy)`` centroid in pixels (used when origin="center")
    """
    if origin == "corner":
        x0, y0 = 0.0, 0.0
    elif origin == "center":
        x0, y0 = centroid_px          # cloud centroid is (0, 0)
    elif origin == "image_center":
        x0 = image_shape[1] / 2.0    # dead-centre of the image
        y0 = image_shape[0] / 2.0
    else:
        raise ValueError(f"origin must be 'corner', 'center', or 'image_center'; got {origin!r}.")

    return (x_px - x0) * scale, (y_px - y0) * scale


def _load_image(
    image: np.ndarray | str | PathLike | _ACImg,
) -> np.ndarray:

    if isinstance(image, (str, PathLike)):
        return np.loadtxt(image, delimiter=",")

    if isinstance(image, _ACImg):
        # Return the RAW image array so that any bg_region passed to
        # analyse_diffusion_cloud is the *only* place subtraction happens.
        # _ACImg.img is already bg-subtracted when bg_region was
        # supplied at construction time; using img_raw here avoids a second
        # subtraction when the caller passes the same bg_region to this function.
        return np.asarray(image.img_raw, dtype=float)

    return np.asarray(image, dtype=float)

def _binary_area(mask: np.ndarray) -> float:
    """
    Approximate MATLAB bwarea for a binary image.

    Returns area in pixel².
    """
    mask = np.asarray(mask, dtype=bool)

    a = mask[:-1, :-1]
    b = mask[:-1, 1:]
    c = mask[1:, :-1]
    d = mask[1:, 1:]

    pattern = (
        a.astype(np.uint8)
        + 2 * b.astype(np.uint8)
        + 4 * c.astype(np.uint8)
        + 8 * d.astype(np.uint8)
    )

    weights = np.zeros(16)

    # 0 foreground pixels
    weights[0] = 0.0

    # 1 foreground pixel
    for p in [1, 2, 4, 8]:
        weights[p] = 0.25

    # 2 foreground pixels
    # Adjacent pair
    for p in [3, 5, 6, 9, 10, 12]:
        weights[p] = 0.5

    # Diagonal pair
    for p in [6, 9]:
        weights[p] = 0.5

    # 3 foreground pixels
    for p in [7, 11, 13, 14]:
        weights[p] = 0.75

    # 4 foreground pixels
    weights[15] = 1.0

    return float(np.sum(weights[pattern]))
