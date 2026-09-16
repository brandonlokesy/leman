"""
Regression tests for the laser-spot annotation in ``plotting`` (audit A3).

``animate_real_space_PL_map`` used to hand ``set_path_effects`` the
``matplotlib.patheffects`` *module* instead of a path-effect instance.
``set_path_effects`` validates nothing — it only stores the list — so nothing
failed until the renderer walked that list and called ``draw_path`` on each
entry. The animation object therefore built cleanly and died later, on render or
save, which is how the bug survived in the file.

Every test here consequently forces a **real draw**: constructing the figure
cannot catch this class of defect. ``test_module_as_path_effect_fails_on_draw``
pins that mechanism, so if a future matplotlib starts validating eagerly (or
stops raising), the premise of the other tests is flagged rather than silently
weakened.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle
from matplotlib.patheffects import AbstractPathEffect

from leman import plotting

SHAPE    = (32, 40)      # (ny, nx) — non-square, so a transposed frame would show
N_FRAMES = 3


class _FakeLaserRef:
    """Stand-in for ACLaserRefImg: only these three are read."""

    center_x = 18.0
    center_y = 12.0
    radius   = 5.0


class _FakeScan:
    """
    Duck-typed ACImgSweep. The plotting functions touch only
    ``load_frame``, ``n_frames`` and ``laser_ref``, so no files are needed and
    the test stays hermetic.
    """

    def __init__(self, laser_ref=None, n_frames=N_FRAMES):
        self.n_frames  = n_frames
        self.laser_ref = laser_ref
        # A ramp plus a per-frame offset, so set_data() changes something
        # observable from frame to frame.
        yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
        self._frames = [xx + yy + 10.0 * i for i in range(n_frames)]

    def load_frame(self, idx: int):
        return self._frames[idx]


@pytest.fixture(autouse=True)
def _close_figures():
    """Plotting functions never close their figures; don't leak them across tests."""
    yield
    plt.close("all")


def _circles(ax):
    """Every Circle patch on *ax* — the laser annotation is the only one drawn."""
    return [p for p in ax.patches if isinstance(p, Circle)]


# ---------------------------------------------------------------------------
# The mechanism the bug relied on
# ---------------------------------------------------------------------------


def test_module_as_path_effect_fails_on_draw():
    """
    The old call reproduced exactly: it is accepted, then fails at draw.

    This is the guard for the tests below — it asserts both halves of why they
    have to render, namely that the bad value passes the setter and that the
    renderer is what rejects it.
    """
    fig, ax = plt.subplots()
    circle = Circle((5, 5), radius=2, facecolor="none", edgecolor="red")
    circle.set_path_effects([path_effects])          # module, not an instance
    ax.add_patch(circle)

    with pytest.raises(AttributeError, match="draw_path"):
        fig.canvas.draw()


# ---------------------------------------------------------------------------
# The shared helper
# ---------------------------------------------------------------------------


def test_draw_laser_circle_halo_is_a_path_effect_instance():
    fig, ax = plt.subplots()
    circle = plotting._draw_laser_circle(ax, _FakeLaserRef())

    effects = circle.get_path_effects()
    assert effects, "halo=True should install a path effect"
    assert all(isinstance(e, AbstractPathEffect) for e in effects)
    fig.canvas.draw()                                # the assertion that matters


def test_draw_laser_circle_returns_the_added_artist():
    fig, ax = plt.subplots()
    circle = plotting._draw_laser_circle(ax, _FakeLaserRef())

    # Returned so callers can restyle without a parameter per property.
    assert circle in ax.patches
    assert circle.get_center() == (_FakeLaserRef.center_x, _FakeLaserRef.center_y)
    assert circle.get_radius() == _FakeLaserRef.radius


def test_draw_laser_circle_without_halo_sets_no_effects():
    fig, ax = plt.subplots()
    circle = plotting._draw_laser_circle(ax, _FakeLaserRef(), halo=False)

    assert circle.get_path_effects() == []
    fig.canvas.draw()


# ---------------------------------------------------------------------------
# animate_real_space_PL_map — the A3 call site
# ---------------------------------------------------------------------------


def test_animation_with_laser_annotation_draws():
    scan = _FakeScan(laser_ref=_FakeLaserRef())
    fig, anim = plotting.animate_real_space_PL_map(scan)
    ax = fig.axes[0]

    fig.canvas.draw()                                # A3: this used to raise

    (circle,) = _circles(ax)
    assert all(isinstance(e, AbstractPathEffect) for e in circle.get_path_effects())
    assert circle.get_linestyle() in ("--", "dashed")   # call site keeps its dashes
    assert anim is not None


def test_animation_renders_every_frame(tmp_path):
    """
    End-to-end via ``save``: the operation that failed in real use. Pillow is
    used rather than ffmpeg so the test has no external-binary dependency.
    """
    scan = _FakeScan(laser_ref=_FakeLaserRef())
    fig, anim = plotting.animate_real_space_PL_map(
        scan, var_array=[0.0, 1.0, 2.0], var_label="E-field",
    )

    out = tmp_path / "anim.gif"
    anim.save(str(out), writer=PillowWriter(fps=2))
    assert out.stat().st_size > 0


def test_no_circle_when_annotation_disabled():
    scan = _FakeScan(laser_ref=_FakeLaserRef())
    fig, _ = plotting.animate_real_space_PL_map(scan, laser_annotation=False)
    fig.canvas.draw()

    assert _circles(fig.axes[0]) == []


def test_no_circle_when_scan_has_no_laser_ref():
    scan = _FakeScan(laser_ref=None)
    fig, _ = plotting.animate_real_space_PL_map(scan, laser_annotation=True)
    fig.canvas.draw()

    assert _circles(fig.axes[0]) == []


# ---------------------------------------------------------------------------
# plot_image — the axes-accepting call site
# ---------------------------------------------------------------------------


class _FakeImage:
    """
    Duck-typed single-image loader. ``plot_image`` reads only ``img`` and
    ``laser_ref``, so no CSV is needed.
    """

    def __init__(self, laser_ref=None):
        yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
        self.img       = (xx + yy).astype(float)
        self.laser_ref = laser_ref


def test_plot_image_annotates_from_the_image_reference():
    plot = plotting.plot_image(
        _FakeImage(laser_ref=_FakeLaserRef()), laser_annotation=True,
    )
    fig, ax, circle = plot.fig, plot.ax, plot.circle
    fig.canvas.draw()

    assert _circles(ax) == [circle]
    assert circle.get_center() == (_FakeLaserRef.center_x, _FakeLaserRef.center_y)
    assert circle.get_radius() == _FakeLaserRef.radius
    # Dashes match the animation call site, so static and animated agree.
    assert circle.get_linestyle() in ("--", "dashed")


def test_plot_image_explicit_ref_overrides_the_image_reference():
    class _Other:
        center_x, center_y, radius = 3.0, 4.0, 1.5

    plot = plotting.plot_image(
        _FakeImage(laser_ref=_FakeLaserRef()),
        laser_annotation = True,
        laser_ref        = _Other(),
    )
    fig, circle = plot.fig, plot.circle
    fig.canvas.draw()

    assert circle.get_center() == (_Other.center_x, _Other.center_y)
    assert circle.get_radius() == _Other.radius


def test_plot_image_no_circle_when_annotation_disabled():
    plot = plotting.plot_image(_FakeImage(laser_ref=_FakeLaserRef()))
    fig, ax, circle = plot.fig, plot.ax, plot.circle
    fig.canvas.draw()

    assert circle is None
    assert _circles(ax) == []


def test_plot_image_laser_ref_alone_draws_nothing():
    """
    ``laser_annotation`` is the only switch — ``laser_ref`` selects which
    reference, it does not enable the overlay. Pinned so the documented gating
    rule is not quietly "fixed" into an implicit enable.
    """
    plot = plotting.plot_image(
        _FakeImage(), laser_ref=_FakeLaserRef(),
    )
    fig, ax, circle = plot.fig, plot.ax, plot.circle
    fig.canvas.draw()

    assert circle is None
    assert _circles(ax) == []


def test_plot_image_bare_array_with_annotation_does_not_raise():
    """A plain ndarray has no ``laser_ref``; the getattr fallback must hold."""
    plot = plotting.plot_image(
        np.zeros(SHAPE), laser_annotation=True,
    )
    fig, ax, circle = plot.fig, plot.ax, plot.circle
    fig.canvas.draw()

    assert circle is None
    assert _circles(ax) == []


def test_plot_image_draws_into_a_supplied_grid_axes():
    """
    The case the parameter exists for: six panels in one figure, each carrying
    its own circle, with no stray figure created along the way.
    """
    fig, axes = plt.subplots(2, 3)
    n_figs_before = len(plt.get_fignums())

    circles = [
        plotting.plot_image(
            _FakeImage(laser_ref=_FakeLaserRef()), ax=ax,
            colorbar=False, show_axes=False, laser_annotation=True,
        ).circle
        for ax in axes.ravel()
    ]
    fig.canvas.draw()

    assert len(plt.get_fignums()) == n_figs_before      # no extra figure
    # Each circle landed on its own panel, not all on the first.
    for ax, circle in zip(axes.ravel(), circles):
        assert _circles(ax) == [circle]


# ---------------------------------------------------------------------------
# ImageSequencePanel — the animation-panel call site
# ---------------------------------------------------------------------------
#
# This panel used to hand-build its own Circle, so it was the copy an author of a
# new AnimationPanel would have copied from. It now calls the shared helper, and
# these are the first tests to reach the branch at all: the two tests that
# construct an ImageSequencePanel elsewhere (test_plotting_frame_source.py) use a
# scan with no laser_ref, so the annotation never ran.


def _panel_on_axes(scan, **kwargs):
    """Build a panel and run init_artists on a real axes, then draw."""
    panel = plotting.ImageSequencePanel(scan, **kwargs)
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))
    fig.canvas.draw()
    return fig, ax, panel


def test_panel_draws_one_circle_and_keeps_it():
    """
    The artist is reachable, which is what makes refusing further style
    parameters honest rather than merely restrictive.
    """
    fig, ax, panel = _panel_on_axes(_FakeScan(laser_ref=_FakeLaserRef()))

    (circle,) = _circles(ax)
    assert panel.laser_circle is circle
    assert circle.get_center() == (_FakeLaserRef.center_x, _FakeLaserRef.center_y)
    assert circle.get_radius() == _FakeLaserRef.radius


def test_panel_circle_is_none_before_init():
    panel = plotting.ImageSequencePanel(_FakeScan(laser_ref=_FakeLaserRef()))

    assert panel.laser_circle is None


def test_panel_draws_no_circle_when_annotation_disabled():
    fig, ax, panel = _panel_on_axes(_FakeScan(laser_ref=_FakeLaserRef()),
                                    laser_annotation=False)

    assert _circles(ax) == []
    assert panel.laser_circle is None


def test_panel_draws_no_circle_when_scan_has_no_laser_ref():
    fig, ax, panel = _panel_on_axes(_FakeScan(laser_ref=None),
                                    laser_annotation=True)

    assert _circles(ax) == []
    assert panel.laser_circle is None


def test_panel_halo_is_a_path_effect_instance():
    """The A3 defect, at this call site: the halo must be an instance, not a module."""
    fig, ax, panel = _panel_on_axes(_FakeScan(laser_ref=_FakeLaserRef()))

    effects = panel.laser_circle.get_path_effects()
    assert effects
    assert all(isinstance(e, AbstractPathEffect) for e in effects)


def test_panel_without_halo_sets_no_effects():
    fig, ax, panel = _panel_on_axes(_FakeScan(laser_ref=_FakeLaserRef()),
                                    laser_halo=False)

    assert panel.laser_circle.get_path_effects() == []


def test_panel_keeps_its_solid_default_line():
    """
    Solid survives GIF palette quantization where a thin dashed line does not, so
    this panel's default must not drift to the dashes the static plots use.
    """
    fig, ax, panel = _panel_on_axes(_FakeScan(laser_ref=_FakeLaserRef()))

    assert panel.laser_circle.get_linestyle() in ("-", "solid")
