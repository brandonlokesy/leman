"""
Tests for ``plot_image``'s coordinate mapping and its NaN handling.

Forces the Agg backend and does a real draw, per the convention in
test_plotting_laser_circle.py -- constructing a figure without rendering it
would not catch an imshow() call with a bad kwarg.
"""

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from leman import plotting

DATA = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])


def test_plot_image_default_behaviour_is_unchanged():
    # extent=None, origin="upper" must reproduce plain pixel-index behaviour.
    plot = plotting.plot_image(DATA)
    plot.fig.canvas.draw()
    assert plot.im.get_extent() == [-0.5, 2.5, 1.5, -0.5]  # matplotlib's own default
    assert plot.im.origin == "upper"
    plt.close(plot.fig)


def test_plot_image_extent_and_origin_are_applied():
    plot = plotting.plot_image(
        DATA, extent=(0.0, 3.0, 0.0, 2.0), origin="lower",
        xlabel="X (um)", ylabel="Y (um)",
    )
    plot.fig.canvas.draw()
    assert plot.im.get_extent() == [0.0, 3.0, 0.0, 2.0]
    assert plot.im.origin == "lower"
    assert plot.ax.get_xlabel() == "X (um)"
    plt.close(plot.fig)


def test_origin_moves_the_data_not_the_axis():
    # With an explicit extent the y-axis numbers increase upward for both
    # origins, so origin chooses which data row is drawn at the bottom.  A map
    # whose row 0 holds its smallest Y needs "lower", or it is drawn mirrored
    # against correct axis labels.  Read back rendered pixels rather than the
    # origin attribute, which would only echo the argument.
    rows = np.array([[0.0, 0.0], [255.0, 255.0]])   # row 0 dark, row 1 bright
    ylims, dark_ends = [], []
    for origin in ("upper", "lower"):
        plot = plotting.plot_image(
            rows, extent=(0.0, 2.0, 0.0, 2.0), origin=origin,
            colorbar=False, show_axes=False, clim=(0.0, 255.0), figsize=(1, 1),
        )
        plot.fig.subplots_adjust(0, 0, 1, 1)
        plot.fig.canvas.draw()
        # Red channel is enough: the colormap is monotonic in it either way.
        buf = np.asarray(plot.fig.canvas.buffer_rgba())[..., 0].astype(float)
        half = buf.shape[0] // 2
        ylims.append(plot.ax.get_ylim())
        dark_ends.append("top" if buf[:half].mean() < buf[half:].mean() else "bottom")
        plt.close(plot.fig)

    assert ylims[0] == ylims[1] == (0.0, 2.0)   # axis direction identical
    assert dark_ends == ["top", "bottom"]       # row 0 moved, the axis did not


def test_plot_image_masks_nan_instead_of_coloring_it():
    plot = plotting.plot_image(DATA)
    plot.fig.canvas.draw()
    arr = plot.im.get_array()
    assert np.ma.is_masked(arr[0, 2])
    assert not np.ma.is_masked(arr[0, 0])
    plt.close(plot.fig)


def test_rescale_img_with_a_nan_does_not_blank_the_panel():
    # rescale_intensity(in_range="image") reads its limits off the array's min
    # and max, so one NaN makes both NaN and every pixel comes back NaN.  The
    # mask therefore has to be applied first.  A fit-parameter map with
    # not-computed pixels is the case that hits this.
    plot = plotting.plot_image(DATA, rescale_img=True)
    plot.fig.canvas.draw()
    arr = plot.im.get_array()
    assert np.ma.is_masked(arr[0, 2])                 # the NaN, still masked
    finite = arr.compressed()
    assert finite.size == 5                           # the other five survive
    assert np.isfinite(finite).all()
    assert (finite.min(), finite.max()) == (0.0, 1.0)  # genuinely rescaled
    plt.close(plot.fig)


def test_plot_image_all_nan_still_renders_via_clim():
    # An explicit clim must keep all-NaN data from raising.
    plot = plotting.plot_image(np.full((2, 2), np.nan), clim=(0.0, 1.0))
    plot.fig.canvas.draw()
    plt.close(plot.fig)
