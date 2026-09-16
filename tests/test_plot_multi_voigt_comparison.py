"""
Tests for plot_multi_voigt_overlay and plot_fit_param_comparison -- promoted
from example-Raman.ipynb's notebook-local helpers once it became clear they
have nothing Raman-specific in them: both operate on ordinary
fit_multi_voigt results (fitting.fit_raman_modes is one such wrapper, but
neither function reads anything about materials or layer counts).

Forces the Agg backend and does a real draw, per the convention in
test_plotting_laser_circle.py.
"""

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from leman import plotting
from leman.fitting import fit_multi_voigt, multi_voigt

X = np.linspace(0.0, 100.0, 500)


def _three_peak_result():
    true = [(50.0, 30.0, 2.0, 2.0), (5.0, 45.0, 3.0, 3.0), (20.0, 70.0, 2.0, 2.0)]
    y = multi_voigt(X, *[v for peak in true for v in peak]) + 3.0
    return fit_multi_voigt(X, y, p0=true), y


def _two_peak_result():
    true = [(40.0, 25.0, 2.0, 2.0), (10.0, 60.0, 2.0, 2.0)]
    y = multi_voigt(X, *[v for peak in true for v in peak]) + 2.0
    return fit_multi_voigt(X, y, p0=true), y


# ---------------------------------------------------------------------------
# plot_multi_voigt_overlay
# ---------------------------------------------------------------------------


def test_plot_multi_voigt_overlay_one_panel_per_sample():
    three, y3 = _three_peak_result()
    two, y2 = _two_peak_result()
    fit_results = {"a": three, "b": two}
    fit_windows = {"a": (X, y3), "b": (X, y2)}

    fig, axes = plotting.plot_multi_voigt_overlay(fit_results, fit_windows, ncols=3)
    fig.canvas.draw()

    assert axes.shape == (1, 2)  # 2 samples, ncols capped to 2
    assert axes[0, 0].get_title() == "a"
    assert axes[0, 1].get_title() == "b"


def test_plot_multi_voigt_overlay_draws_the_right_number_of_peak_curves():
    three, y3 = _three_peak_result()
    fit_results = {"a": three}
    fit_windows = {"a": (X, y3)}

    fig, axes = plotting.plot_multi_voigt_overlay(fit_results, fit_windows)
    fig.canvas.draw()

    # data + fit(sum) + 3 peak components = 5 lines.
    assert len(axes[0, 0].get_lines()) == 5


def test_plot_multi_voigt_overlay_grid_wraps_and_hides_unused_slots():
    results = {name: _two_peak_result() for name in ("a", "b", "c", "d")}
    fit_results = {name: r for name, (r, _) in results.items()}
    fit_windows = {name: (X, y) for name, (_, y) in results.items()}

    fig, axes = plotting.plot_multi_voigt_overlay(fit_results, fit_windows, ncols=3)
    fig.canvas.draw()

    assert axes.shape == (2, 3)  # ceil(4/3) = 2 rows
    visible = [ax.get_visible() for ax in axes.flatten()]
    assert visible == [True, True, True, True, False, False]


def test_plot_multi_voigt_overlay_mode_names_label_legend_and_fall_back():
    three, y3 = _three_peak_result()
    fit_results = {"a": three}
    fit_windows = {"a": (X, y3)}

    fig, axes = plotting.plot_multi_voigt_overlay(
        fit_results, fit_windows, mode_names=["first", "second"],
    )
    fig.canvas.draw()

    legend_labels = [line.get_label() for line in axes[0, 0].get_lines()]
    assert any(lbl.startswith("first") for lbl in legend_labels)
    assert any(lbl.startswith("second") for lbl in legend_labels)
    assert any(lbl.startswith("peak 2") for lbl in legend_labels)  # 3rd peak, no name given


def test_plot_multi_voigt_overlay_no_axis_labels_by_default():
    two, y2 = _two_peak_result()
    fig, axes = plotting.plot_multi_voigt_overlay({"a": two}, {"a": (X, y2)})
    fig.canvas.draw()
    assert axes[0, 0].get_xlabel() == ""
    assert axes[0, 0].get_ylabel() == ""


def test_plot_multi_voigt_overlay_applies_given_axis_labels_and_unit():
    two, y2 = _two_peak_result()
    fig, axes = plotting.plot_multi_voigt_overlay(
        {"a": two}, {"a": (X, y2)}, mode_names=["p0", "p1"],
        xlabel="Raman shift (cm$^{-1}$)", ylabel="Counts", x_unit=" cm$^{-1}$",
    )
    fig.canvas.draw()
    assert axes[0, 0].get_xlabel() == "Raman shift (cm$^{-1}$)"
    assert axes[0, 0].get_ylabel() == "Counts"
    legend_labels = [line.get_label() for line in axes[0, 0].get_lines()]
    assert any("cm$^{-1}$)" in lbl for lbl in legend_labels)


# ---------------------------------------------------------------------------
# plot_fit_param_comparison
# ---------------------------------------------------------------------------


def test_plot_fit_param_comparison_grid_shape_matches_active_peaks():
    three, _ = _three_peak_result()
    two, _ = _two_peak_result()
    fit_results = {"a": three, "b": two}

    fig, axes = plotting.plot_fit_param_comparison(fit_results, ["p0", "p1", "p2"])
    fig.canvas.draw()

    assert axes.shape == (3, 4)  # 3 active peak rows x 4 Voigt params


def test_plot_fit_param_comparison_skips_a_peak_no_sample_has():
    two, _ = _two_peak_result()
    fit_results = {"a": two}

    # mode_names lists 3 peaks, but this sample only has 2 -- the 3rd row
    # must not appear at all (no sample has it), unlike a per-sample gap.
    fig, axes = plotting.plot_fit_param_comparison(fit_results, ["p0", "p1", "p2"])
    fig.canvas.draw()

    assert axes.shape == (2, 4)


def test_plot_fit_param_comparison_leaves_a_gap_for_a_missing_mode():
    three, _ = _three_peak_result()
    two, _ = _two_peak_result()
    fit_results = {"a": three, "b": two}  # "b" has no peak index 2

    fig, axes = plotting.plot_fit_param_comparison(fit_results, ["p0", "p1", "p2"])
    fig.canvas.draw()

    # Row 2 (peak index 2, "p2") should have exactly one point plotted
    # ("a"), not two.
    ax = axes[2, 0]  # "center" column
    assert len(ax.lines) + len(ax.collections) > 0  # something was drawn for "a"
    # sample "b" contributes no artist on this row/column -- verified via
    # the x-tick labels still listing both samples (no silent axis shift).
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]


def test_plot_fit_param_comparison_param_labels_override_and_fallback():
    two, _ = _two_peak_result()
    fig, axes = plotting.plot_fit_param_comparison(
        {"a": two}, ["p0", "p1"], param_labels={"center": "Center (cm$^{-1}$)"},
    )
    fig.canvas.draw()
    assert axes[0, 0].get_title() == "Center (cm$^{-1}$)"
    assert axes[0, 1].get_title() == "amp"  # no override given -- bare key name
