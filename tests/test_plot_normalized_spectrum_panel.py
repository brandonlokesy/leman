"""
Tests for plotting.NormalizedSpectrumPanel -- promoted from
example_position_xy_scan.ipynb's notebook-local class once the notebook
needed the identical behaviour a second time, for a different scan.

Forces the Agg backend and does a real draw, per the convention in
test_plotting_laser_circle.py.
"""

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from leman import plotting


class _FakeScan:
    """Minimal stand-in exposing what NormalizedSpectrumPanel actually reads."""

    def __init__(self, peaks=(10.0, 1000.0, 100.0), n_pixels=60):
        self.n_sweeps = len(peaks)
        self.wavelength = np.linspace(700.0, 800.0, n_pixels)
        self.energy = np.linspace(1.55, 1.77, n_pixels)  # ascending, need not be exact hc/wl here
        x = np.linspace(-5, 5, n_pixels)
        gaussian = np.exp(-x**2 / 2)
        self.best_energy_spectra = np.stack([p * gaussian for p in peaks], axis=1)
        self.spectra = self.best_energy_spectra.copy()
        self.scanner_x = np.array([1.0, 2.0, 3.0])[: self.n_sweeps]
        self.scanner_y = np.array([10.0, 20.0, 30.0])[: self.n_sweeps]


def _panel(**kwargs):
    fig, ax = matplotlib.pyplot.subplots()
    panel = plotting.NormalizedSpectrumPanel(_FakeScan(), **kwargs)
    panel.init_artists(ax, panel.n_frames)
    return panel, fig, ax


def test_each_frame_is_normalized_to_its_own_range():
    panel, fig, ax = _panel(smooth_window=None)
    for frame in range(panel.n_frames):
        panel.update(frame)
        y = panel._line.get_ydata()
        assert y.min() == pytest.approx(0.0, abs=1e-9)
        assert y.max() == pytest.approx(1.0, abs=1e-9)


def test_color_reflects_the_global_peak_not_each_frame_own_peak():
    # Peaks are (10, 1000, 100) -- despite every frame being normalized to
    # the same [0, 1] height, the weak frame (10) and the strong one (1000)
    # must get different colours, spanning the *global* 10-1000 range.
    panel, fig, ax = _panel(smooth_window=None)
    panel.update(0)
    color_weak = panel.cmap(panel._norm(panel._peaks[0]))
    panel.update(1)
    color_strong = panel.cmap(panel._norm(panel._peaks[1]))
    assert color_weak != color_strong
    assert panel._norm(panel._peaks[0]) == pytest.approx(0.0, abs=1e-9)
    assert panel._norm(panel._peaks[1]) == pytest.approx(1.0, abs=1e-9)


def test_secondary_axis_only_added_when_requested():
    # A secondary_xaxis lives in ax.child_axes, not fig.axes.
    _, fig_off, ax_off = _panel(secondary_x_axis=False, smooth_window=None)
    _, fig_on, ax_on = _panel(secondary_x_axis=True, smooth_window=None)
    fig_off.canvas.draw()
    fig_on.canvas.draw()
    assert ax_off.child_axes == []
    assert len(ax_on.child_axes) == 1


def test_frame_label_reports_every_requested_sweep_attr():
    panel, fig, ax = _panel(sweep_attrs=["scanner_x", "scanner_y"],
                             sweep_units=["V", "um"], smooth_window=None)
    label = panel.frame_label(1)
    assert "scanner_x = 2" in label
    assert "scanner_y = 20" in label
    assert "V" in label and "um" in label


def test_frame_label_is_none_without_sweep_attrs():
    panel, fig, ax = _panel(smooth_window=None)
    assert panel.frame_label(0) is None


def test_smoothing_can_be_disabled():
    scan = _FakeScan()
    fig, ax = matplotlib.pyplot.subplots()
    panel = plotting.NormalizedSpectrumPanel(scan, smooth_window=None)
    panel.init_artists(ax, panel.n_frames)
    raw_normalized = panel._normalized.copy()

    fig2, ax2 = matplotlib.pyplot.subplots()
    panel2 = plotting.NormalizedSpectrumPanel(scan, smooth_window=5, smooth_poly=2)
    panel2.init_artists(ax2, panel2.n_frames)

    # Smoothing a clean Gaussian barely changes it, but the two arrays should
    # not be bit-identical -- proof smooth_window actually did something.
    assert not np.array_equal(raw_normalized, panel2._normalized)


def test_ylabel_uses_normalized_signal_label():
    panel, fig, ax = _panel(smooth_window=None)
    assert "norm" in ax.get_ylabel().lower()
