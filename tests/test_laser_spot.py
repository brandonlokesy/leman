"""
Tests for ACLaserRefImg laser-spot localization.

Synthetic images are written to temporary CSVs (the class loads via
``np.loadtxt``).  We check centre/radius recovery for a clean dark-background
spot and, crucially, for a spot sitting on a structured white-light background
where the old row/column-projection fit was biased.
"""

import numpy as np
import pytest

from leman.loaders import ACLaserRefImg

SHAPE = (128, 128)            # (ny, nx)
X0, Y0 = 80.0, 50.0           # true laser centre (col, row)
SIGMA = 6.0                   # true laser sigma
RADIUS = 2 * SIGMA            # true 1/e^2 radius


def _gaussian_spot(amp=180.0, x0=X0, y0=Y0, sigma=SIGMA):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))


def _write_csv(tmp_path, arr, name="laser.csv"):
    path = tmp_path / name
    np.savetxt(path, arr, delimiter=",")
    return path


def _dark_background(seed=0):
    rng = np.random.default_rng(seed)
    return 5.0 + rng.normal(0, 1.0, SHAPE)


def _white_light_background():
    """Large-scale illumination: linear gradient + a broad bright glow."""
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    gradient = 10.0 + 0.3 * xx + 0.2 * yy
    glow = 120.0 * np.exp(-((xx - 30) ** 2 + (yy - 90) ** 2) / (2 * 40.0 ** 2))
    return gradient + glow


# ---------------------------------------------------------------------------
# Dark background
# ---------------------------------------------------------------------------


def test_dark_background_recovers_centre_and_radius(tmp_path):
    img = _dark_background() + _gaussian_spot()
    ref = ACLaserRefImg(
        str(_write_csv(tmp_path, img)),
        expected_radius_px=SIGMA, white_light=False,
    )
    assert abs(ref.center_x - X0) < 1.0
    assert abs(ref.center_y - Y0) < 1.0
    assert abs(ref.radius - RADIUS) < 0.15 * RADIUS


# ---------------------------------------------------------------------------
# White-light background (the case this change targets)
# ---------------------------------------------------------------------------


def test_white_light_background_recovers_centre(tmp_path):
    img = _white_light_background() + _gaussian_spot()
    ref = ACLaserRefImg(
        str(_write_csv(tmp_path, img)),
        expected_radius_px=SIGMA, white_light=True,
    )
    # Centre recovered despite the bright off-centre glow + gradient.
    assert abs(ref.center_x - X0) < 2.5
    assert abs(ref.center_y - Y0) < 2.5
    assert abs(ref.radius - RADIUS) < 0.30 * RADIUS


def test_white_light_off_is_biased_by_background(tmp_path):
    """Sanity check that the white-light handling actually matters here."""
    img = _white_light_background() + _gaussian_spot()
    path = str(_write_csv(tmp_path, img))
    on = ACLaserRefImg(path, expected_radius_px=SIGMA, white_light=True)
    off = ACLaserRefImg(path, expected_radius_px=SIGMA, white_light=False)
    err_on = np.hypot(on.center_x - X0, on.center_y - Y0)
    err_off = np.hypot(off.center_x - X0, off.center_y - Y0)
    assert err_on < err_off            # background suppression improves accuracy
    assert err_on < 2.5


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_radius_scales_with_spot_size(tmp_path):
    img = _dark_background() + _gaussian_spot(sigma=10.0)
    ref = ACLaserRefImg(
        str(_write_csv(tmp_path, img)),
        expected_radius_px=10.0, white_light=False,
    )
    assert abs(ref.radius - 20.0) < 0.20 * 20.0


def test_degenerate_image_does_not_crash(tmp_path):
    img = np.full(SHAPE, 7.0)          # no spot at all
    ref = ACLaserRefImg(
        str(_write_csv(tmp_path, img)), expected_radius_px=SIGMA,
    )
    # Falls back gracefully to sane, in-bounds values.
    assert 0 <= ref.center_x < SHAPE[1]
    assert 0 <= ref.center_y < SHAPE[0]
    assert ref.radius > 0
