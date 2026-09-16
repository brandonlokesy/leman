"""
Tests for ``frame_source=`` on the real-space viewers (audit B1).

``bg_region`` on ``ACImgSweep`` was stored and never read, so no
viewer could show a background-subtracted frame.  The correction now lives in
``load_frame_bg``, leaving ``load_frame`` as the file's own counts, and the
viewers choose between them.

Every assertion here reads the array *off the axes* rather than checking that the
call succeeded: forwarding a parameter and then ignoring it is exactly the defect
B1 was, so a test that only builds the figure would not have caught it.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import numpy as np
import pytest

from leman import plotting
from leman.loaders import ACImgSweep

SHAPE = (8, 10)                        # (ny, nx) — non-square, so a transpose shows

# Signal-free corner of [1, 1, 1, 21]: median 1, so a subtracted frame peaks one
# count below a raw one.  Small and exact, not a tolerance.
BG_REGION   = (slice(0, 2), slice(0, 2))
BG_MEDIAN   = 1.0
SIGNAL_FILL = 100.0
N_FRAMES    = 3


def _scan(tmp_path, **kwargs):
    """A three-frame scan on a pedestal, each frame offset so frames differ."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(N_FRAMES):
        img = np.full(SHAPE, SIGNAL_FILL + i)
        img[BG_REGION] = [[1.0, 1.0], [1.0, 21.0]]
        np.savetxt(tmp_path / f"pl_iter_{i}.csv", img, delimiter=",")
    return ACImgSweep(tmp_path, prefix="pl_", **kwargs)


def _drawn(ax) -> np.ndarray:
    """The array currently on *ax*'s image artist."""
    return np.asarray(ax.images[0].get_array())


# ---------------------------------------------------------------------------
# plot_real_space_PL_map
# ---------------------------------------------------------------------------


def test_single_frame_default_honours_the_loaders_bg_region(tmp_path):
    scan = _scan(tmp_path, bg_region=BG_REGION)
    _, ax = plotting.plot_real_space_PL_map(scan)

    assert np.allclose(_drawn(ax).max(), SIGNAL_FILL - BG_MEDIAN)


def test_single_frame_raw_shows_the_files_counts(tmp_path):
    scan = _scan(tmp_path, bg_region=BG_REGION)
    _, ax = plotting.plot_real_space_PL_map(scan, frame_source="raw")

    assert np.allclose(_drawn(ax).max(), SIGNAL_FILL)


def test_single_frame_bg_raises_without_a_bg_region(tmp_path):
    scan = _scan(tmp_path)
    with pytest.raises(ValueError, match="not available on this scan"):
        plotting.plot_real_space_PL_map(scan, frame_source="bg")


# ---------------------------------------------------------------------------
# animate_real_space_PL_map
# ---------------------------------------------------------------------------


def test_animation_raw_and_best_differ_on_every_frame(tmp_path):
    # The whole point of the toggle: animating one scan both ways shows what the
    # subtraction removed.  Checked past frame 0, since update() is a separate
    # code path from init.
    scan = _scan(tmp_path, bg_region=BG_REGION)

    raw_fig, raw_anim = plotting.animate_real_space_PL_map(scan, frame_source="raw")
    bg_fig,  bg_anim  = plotting.animate_real_space_PL_map(scan, frame_source="best")

    for frame in range(N_FRAMES):
        # FuncAnimation exposes its per-frame callable only as _func; calling it is
        # what advances the artist without rendering a whole GIF.
        raw_anim._func(frame)
        bg_anim._func(frame)
        assert np.allclose(_drawn(raw_fig.axes[0]).max(), SIGNAL_FILL + frame)
        assert np.allclose(_drawn(bg_fig.axes[0]).max(),
                           SIGNAL_FILL + frame - BG_MEDIAN)


# ---------------------------------------------------------------------------
# ImageSequencePanel
# ---------------------------------------------------------------------------


def test_panel_update_honours_its_frame_source(tmp_path):
    import matplotlib.pyplot as plt

    scan  = _scan(tmp_path, bg_region=BG_REGION)
    panel = plotting.ImageSequencePanel(scan, frame_source="raw")
    _, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))

    assert np.allclose(_drawn(ax).max(), SIGNAL_FILL)      # frame 0, via init
    panel.update(2)
    assert np.allclose(_drawn(ax).max(), SIGNAL_FILL + 2)  # frame 2, via update
    plt.close("all")


def test_panel_defaults_to_the_corrected_frame(tmp_path):
    import matplotlib.pyplot as plt

    scan  = _scan(tmp_path, bg_region=BG_REGION)
    panel = plotting.ImageSequencePanel(scan)
    _, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))
    panel.update(1)

    assert np.allclose(_drawn(ax).max(), SIGNAL_FILL + 1 - BG_MEDIAN)
    plt.close("all")


# ---------------------------------------------------------------------------
# preview_image
# ---------------------------------------------------------------------------


def test_preview_image_honours_frame_source(tmp_path):
    import matplotlib.pyplot as plt

    scan = _scan(tmp_path, bg_region=BG_REGION)
    _, ax_best = scan.preview_image(0)
    _, ax_raw  = scan.preview_image(0, frame_source="raw")

    assert np.allclose(_drawn(ax_best).max(), SIGNAL_FILL - BG_MEDIAN)
    assert np.allclose(_drawn(ax_raw).max(),  SIGNAL_FILL)
    plt.close("all")
