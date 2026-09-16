"""
Tests for the animation engine — ``animate_panels`` and the ``AnimationPanel`` protocol.

This surface had no tests at all (defect E18). Nothing called ``animate_panels``, so the
frame-count minimum, the shared-title composition and the ``frame_label`` hook were
entirely unpinned, and the one thing that class of code gets wrong is *late* failure: an
animation builds cleanly and then dies on render, or renders the wrong frames while the
caption claims otherwise. Constructing the figure cannot catch either.

So the rule here is the same one ``test_plotting_laser_circle.py`` follows: anything that
only manifests during rendering is tested by **actually rendering**, via
``anim.save(..., PillowWriter)`` into ``tmp_path``. ``_ProbePanel`` records every frame
index it is handed, which is what makes "did it animate the frames it said it would?"
an assertion rather than an inspection.
"""

import warnings

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import HTMLWriter, PillowWriter
from matplotlib.text import Text
from PIL import Image

from leman import plotting, processing
from leman.loaders import ACSpectralSweep

from test_loaders import make_spectral_csv
from test_loaders_nesting import RASTER


SHAPE = (12, 16)      # (ny, nx) — non-square, so a transposed frame would show

# Matplotlib warns from Animation.__del__ when an animation is collected having never
# drawn. Many tests here deliberately build a figure and assert on it without
# rendering, so the warning is expected noise for them rather than a signal — and a
# noisy suite hides real warnings. It appears at all only because the engine does not
# blit: setting up blitting used to force an init draw, which marked the animation as
# started as a side effect. Scoped to this one message so anything else still surfaces.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning"
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Plotting functions never close their figures; don't leak them across tests."""
    yield
    plt.close("all")


class _ProbePanel(plotting.AnimationPanel):
    """
    A minimal AnimationPanel that records what the engine asked of it.

    ``seen`` is the frame index passed to every ``update`` call, in order, which is how
    the tests below check *which* frames were animated rather than merely how many.
    ``init_frames`` records the frame sequence handed to ``init_artists``.
    """

    def __init__(self, n_frames=4, title="", label=None, top_axis=False):
        self._n_frames   = n_frames
        self.title       = title
        self.label       = label
        self.top_axis    = top_axis
        self.seen        = []
        self.init_frames = None
        self._line       = None

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def init_artists(self, ax, frames) -> None:
        self.init_frames = list(frames)
        x = np.linspace(1.5, 2.5, 32)
        (self._line,) = ax.plot(x, np.sin(x))
        ax.set_xlabel("Energy (eV)")
        if self.title:
            ax.set_title(self.title)
        if self.top_axis:
            # A conjugate wavelength axis, drawn the way a spectrum panel would.
            # Only its vertical footprint matters here.
            def _conv(e):
                e = np.asarray(e, float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    return 1239.84193 / e
            secax = ax.secondary_xaxis("top", functions=(_conv, _conv))
            secax.set_xlabel("Wavelength (nm)")

    def update(self, frame: int) -> tuple:
        self.seen.append(frame)
        return (self._line,)

    def frame_label(self, frame: int):
        if self.label is None:
            return None
        return f"{self.label} {frame}"


class _LateLabelPanel(_ProbePanel):
    """
    A panel whose ``frame_label`` only works once ``init_artists`` has run.

    ``DiffusionCloudPanel`` behaves this way — it resolves its swept-variable array in
    ``init_artists`` — and ``animate_panels`` carries a load-bearing comment about
    evaluating the suptitle afterwards for exactly this reason. Without a panel that
    actually depends on the ordering, that comment is unenforced.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ready = False

    def init_artists(self, ax, frames) -> None:
        super().init_artists(ax, frames)
        self._ready = True

    def frame_label(self, frame: int):
        if not self._ready:
            return None
        return f"resolved {frame}"


def _shared_title(fig) -> Text:
    """
    The engine's shared title, or ``None``.

    A real ``fig.suptitle``, so it lives on the figure. That is what makes the layout
    engine reserve room for it, which is what keeps it clear of whatever the panels
    draw on top (E20).
    """
    return fig._suptitle if fig.texts else None


def _render(fig, anim, tmp_path, name="anim.gif"):
    """Drive every frame through a real writer and return the output path."""
    out = tmp_path / name
    anim.save(str(out), writer=PillowWriter(fps=4))
    return out


def _advanced_through(seen):
    """
    The frame sequence with consecutive repeats collapsed.

    Building the animation and priming the writer both draw frame 0 — measured as
    ``[0, 0, 0, 0, 1, 2, 3]`` for a four-frame animation — so a raw comparison against
    ``[0, 1, 2, 3]`` pins matplotlib's priming behaviour rather than ours. Collapsing
    runs keeps the assertion on the thing we control: which frames were visited, in
    what order, exactly once each.
    """
    return [f for i, f in enumerate(seen) if i == 0 or f != seen[i - 1]]


# ---------------------------------------------------------------------------
# Frame count
# ---------------------------------------------------------------------------


def test_no_panels_is_refused():
    with pytest.raises(ValueError, match="at least one panel"):
        plotting.animate_panels([])


def test_every_frame_is_animated_by_default():
    panels = [_ProbePanel(n_frames=4), _ProbePanel(n_frames=4)]
    fig, anim = plotting.animate_panels(panels)

    assert all(p.init_frames == [0, 1, 2, 3] for p in panels)
    assert list(anim.new_frame_seq()) == [0, 1, 2, 3]


def test_panels_that_disagree_on_frame_count_are_refused():
    """
    The old behaviour took the minimum and said nothing, so a figure built from a scan
    and an image sequence that do not correspond rendered happily and looked right.
    """
    panels = [_ProbePanel(n_frames=4), _ProbePanel(n_frames=10)]
    with pytest.raises(ValueError, match="disagree on how many frames"):
        plotting.animate_panels(panels)


def test_the_refusal_names_the_counts():
    """A message that does not say which panel is which sends you back to the data."""
    panels = [_ProbePanel(n_frames=4), _ProbePanel(n_frames=10)]
    with pytest.raises(ValueError) as excinfo:
        plotting.animate_panels(panels)

    message = str(excinfo.value)
    assert "has 4" in message and "has 10" in message
    assert "white-light" in message      # names the usual cause of an off-by-one


def test_an_explicit_selection_allows_panels_of_differing_length():
    """
    Refusing is about the *default*, where the engine would have to pick for you.
    Naming the frames yourself removes the guess, so long as every index is valid
    for every panel — which is how the white-light off-by-one gets handled upstream.
    """
    short, long_ = _ProbePanel(n_frames=4), _ProbePanel(n_frames=10)
    fig, anim = plotting.animate_panels([short, long_], frames=range(4))

    assert short.init_frames == [0, 1, 2, 3]
    assert long_.init_frames == [0, 1, 2, 3]


def test_a_window_selects_its_own_frames():
    panel = _ProbePanel(n_frames=10)
    fig, anim = plotting.animate_panels([panel], frames=range(4, 8))

    assert panel.init_frames == [4, 5, 6, 7]
    assert list(anim.new_frame_seq()) == [4, 5, 6, 7]


def test_a_stride_is_just_another_selection():
    """One parameter covers a window, a stride and a single frame."""
    panel = _ProbePanel(n_frames=10)
    fig, anim = plotting.animate_panels([panel], frames=range(0, 10, 3))

    assert panel.init_frames == [0, 3, 6, 9]


def test_every_frame_is_animated_in_order(tmp_path):
    """
    Rendered, not inspected: ``update`` is only driven by the writer, so a frame-sequence
    bug is invisible until something actually saves.
    """
    panel = _ProbePanel(n_frames=4)
    fig, anim = plotting.animate_panels([panel])
    out = _render(fig, anim, tmp_path)

    assert out.stat().st_size > 0
    assert _advanced_through(panel.seen) == [0, 1, 2, 3]


def test_panels_advance_in_lock_step(tmp_path):
    a, b = _ProbePanel(n_frames=4), _ProbePanel(n_frames=4)
    fig, anim = plotting.animate_panels([a, b])
    _render(fig, anim, tmp_path)

    assert a.seen == b.seen


def test_update_receives_the_frames_own_index(tmp_path):
    """
    The whole reason the engine passes native indices: a panel reads its data at the
    index it is handed, with no window offset to carry. Getting this wrong is silent —
    the animation plays real frames while the title names different ones — so it is
    checked through an actual render rather than by inspecting the setup.
    """
    panel = _ProbePanel(n_frames=10)
    fig, anim = plotting.animate_panels([panel], frames=range(6, 9))
    _render(fig, anim, tmp_path)

    assert _advanced_through(panel.seen) == [6, 7, 8]


def test_a_single_frame_selection_animates_one_frame(tmp_path):
    panel = _ProbePanel(n_frames=10)
    fig, anim = plotting.animate_panels([panel], frames=[7])
    _render(fig, anim, tmp_path)

    assert panel.init_frames == [7]
    assert set(panel.seen) == {7}


# ---------------------------------------------------------------------------
# Refusing a selection that cannot mean what it says
# ---------------------------------------------------------------------------


def test_an_empty_selection_is_refused():
    with pytest.raises(ValueError, match="at least one frame"):
        plotting.animate_panels([_ProbePanel(n_frames=4)], frames=[])


def test_a_frame_past_the_end_is_refused():
    """Otherwise this fails inside a writer, mid-render, naming no panel."""
    with pytest.raises(ValueError, match="but the panels have 4 frames"):
        plotting.animate_panels([_ProbePanel(n_frames=4)], frames=range(2, 6))


def test_a_fractional_frame_is_refused():
    with pytest.raises(TypeError, match="whole frame indices"):
        plotting.animate_panels([_ProbePanel(n_frames=4)], frames=[1.5])


def test_a_negative_frame_is_refused():
    """
    Negative indices would work by accident on some panels and silently reorder the
    animation when mixed with positives, so they are refused with the idiom instead.
    """
    with pytest.raises(ValueError, match="counted from 0"):
        plotting.animate_panels([_ProbePanel(n_frames=4)], frames=[-1])


def test_a_bare_count_is_refused():
    """
    ``n_frames=`` used to take a count and sat in this position. A stale positional
    call must not be read as a frame selection.
    """
    with pytest.raises(TypeError):
        plotting.animate_panels([_ProbePanel(n_frames=4)], 3)
    with pytest.raises(TypeError):
        plotting.animate_panels([_ProbePanel(n_frames=4)], n_frames=3)
    with pytest.raises(TypeError, match="sequence of frame indices"):
        plotting.animate_panels([_ProbePanel(n_frames=4)], frames=3)


# ---------------------------------------------------------------------------
# Encoding each frame's peak as the trace's colour
# ---------------------------------------------------------------------------


class _FakeSweep:
    """
    A spectral sweep with known per-frame peaks.

    ``SpectrumLinePanel`` reads its data through ``_resolve_spectra(scan, "best", …)``
    and ``_resolve_x_axis``, both of which only need attributes, so a stand-in can set
    the peaks exactly — which is what makes "the colour scale spans the whole scan"
    an assertion about numbers rather than about shape.
    """

    #: Column maxima, deliberately not monotonic so a window can exclude the extremes.
    PEAKS = np.array([10.0, 90.0, 50.0, 30.0, 70.0])

    def __init__(self):
        n_px = 8
        base = np.linspace(0.0, 1.0, n_px)[:, None]     # (n_px, 1)
        # (n_px, 1) profile scaled by each column's peak -> column max is that peak.
        self.best_energy_spectra = base * self.PEAKS[None, :]
        self.best_spectra        = self.best_energy_spectra
        self.energy              = np.linspace(1.5, 2.5, n_px)
        self.wavelength          = np.linspace(500.0, 700.0, n_px)
        self.n_sweeps            = len(self.PEAKS)
        self.scanner_y           = np.arange(len(self.PEAKS), dtype=float)
        self.signal_name         = "PL intensity"
        self.signal_unit         = "counts"


def test_no_colorbar_by_default():
    """An encoding is a claim about the data, so it is asked for, not assumed."""
    panel = plotting.SpectrumLinePanel(_FakeSweep(), show_sweep_title=False)
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))

    assert panel.colorbar is None
    assert panel.mappable is None
    assert fig.axes == [ax]          # nothing stole axes space


def test_the_colour_scale_spans_the_whole_scan_not_the_animated_frames():
    """
    The colour then means "this frame's brightness", so two clips of one scan agree.
    A window-derived scale would give the same frame different colours in different
    clips, and a one-frame window a degenerate scale.
    """
    panel = plotting.SpectrumLinePanel(_FakeSweep(), cmap="viridis",
                                       show_sweep_title=False)
    fig, ax = plt.subplots()
    panel.init_artists(ax, [2, 3])   # excludes both the dimmest and the brightest

    assert panel.mappable.norm.vmin == _FakeSweep.PEAKS.min()
    assert panel.mappable.norm.vmax == _FakeSweep.PEAKS.max()


def test_two_windows_of_one_scan_agree_on_colour():
    panels = []
    for frames in ([0, 1], [3, 4]):
        panel = plotting.SpectrumLinePanel(_FakeSweep(), cmap="viridis",
                                           show_sweep_title=False)
        fig, ax = plt.subplots()
        panel.init_artists(ax, frames)
        panels.append(panel)

    a, b = panels
    assert a.mappable.norm.vmin == b.mappable.norm.vmin
    assert a.mappable.norm.vmax == b.mappable.norm.vmax


def test_the_line_colour_tracks_each_frames_peak():
    panel = plotting.SpectrumLinePanel(_FakeSweep(), cmap="viridis",
                                       show_sweep_title=False)
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))

    for frame, peak in enumerate(_FakeSweep.PEAKS):
        panel.update(frame)
        assert panel.line.get_color() == panel.mappable.to_rgba(peak)


def test_the_artists_are_reachable_from_the_panel():
    """The return contract, at the only altitude a panel has: its attributes."""
    panel = plotting.SpectrumLinePanel(_FakeSweep(), cmap="viridis",
                                       show_sweep_title=False)
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))

    assert panel.line in ax.lines
    assert panel.colorbar is not None
    assert panel.mappable is not None


def test_cmap_and_color_together_are_refused():
    """cmap overwrites the colour every frame, so color= would be a silent no-op."""
    with pytest.raises(ValueError, match="either color= or cmap="):
        plotting.SpectrumLinePanel(_FakeSweep(), cmap="viridis", color="k")


def test_reinitialising_does_not_stack_colorbars():
    panel = plotting.SpectrumLinePanel(_FakeSweep(), cmap="viridis",
                                       show_sweep_title=False)
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))
    panel.init_artists(ax, range(panel.n_frames))

    # One panel axes plus exactly one colour-bar axes.
    assert len(fig.axes) == 2


def test_the_colormap_goes_through_get_cmap():
    """
    Every other cmap entry point in the module accepts the full ColormapLike
    vocabulary; a raw ScalarMappable would take only registered names.
    """
    panel = plotting.SpectrumLinePanel(
        _FakeSweep(), cmap=["#1b9e77", "#d95f02", "#7570b3"], show_sweep_title=False,
    )
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))

    assert panel.mappable.cmap.N == 3


# ---------------------------------------------------------------------------
# Selecting frames by coordinate
# ---------------------------------------------------------------------------
#
# frame_window turns coordinates into the frame indices animate_panels takes. The
# lookup is the scan's own, so these mostly check that its policies arrive intact
# rather than re-testing them: an ambiguous coordinate refused, a distant one warned
# about, a nest refused.


@pytest.fixture
def sweep(tmp_path):
    """Sweep points 0, 1, 2 at Scanner Y = 7.0, 7.5, 8.0 V."""
    path = tmp_path / "flat.csv"
    make_spectral_csv(path)
    return ACSpectralSweep(str(path), spectra_type="PL", sweep="piezo_y")


@pytest.fixture
def nested_sweep(tmp_path):
    path = tmp_path / "raster.csv"
    make_spectral_csv(path, params=RASTER)
    return ACSpectralSweep(str(path), spectra_type="PL",
                                 fast_sweep="piezo_x", slow_sweep="piezo_y")


def test_a_coordinate_window_gives_frame_indices(sweep):
    assert plotting.frame_window(sweep, 7.0, 8.0) == range(0, 3)


def test_both_endpoints_are_inclusive(sweep):
    """A caller who names two points is asking to see both of them."""
    assert plotting.frame_window(sweep, 7.0, 7.5) == range(0, 2)


def test_an_omitted_endpoint_runs_to_the_edge(sweep):
    assert plotting.frame_window(sweep, 7.5) == range(1, 3)      # to the last frame
    assert plotting.frame_window(sweep, end=7.5) == range(0, 2)  # from the first


def test_the_whole_scan_when_neither_endpoint_is_given(sweep):
    assert plotting.frame_window(sweep) == range(0, sweep.n_sweeps)


def test_one_coordinate_twice_is_a_single_frame(sweep):
    assert plotting.frame_window(sweep, 7.5, 7.5) == range(1, 2)


def test_a_named_axis_is_honoured(sweep):
    """
    Any axis the scan can look up, not only the one the sweep was declared with. A raw
    row label, because a role name like "top_voltage" needs gates= declared first —
    that refusal reaches through frame_window too, which is the point of not
    reimplementing the lookup.
    """
    assert plotting.frame_window(sweep, 0.0, 0.5, axis="V_A") == range(0, 2)


def test_reversed_endpoints_are_refused(sweep):
    """
    Not swapped. A reversed pair is far more often a typo than a request to play
    backwards, and swapping silently would make the typo invisible.
    """
    with pytest.raises(ValueError, match="does not do reverse playback"):
        plotting.frame_window(sweep, 8.0, 7.0)


def test_the_refusal_names_the_reverse_playback_idiom(sweep):
    """
    The refusal is only reasonable because the alternative is one slice away, so the
    message has to carry it — otherwise this reads as a missing feature.
    """
    with pytest.raises(ValueError) as excinfo:
        plotting.frame_window(sweep, 8.0, 7.0)

    message = str(excinfo.value)
    assert "[::-1]" in message
    assert "start must not come after end" in message
    # The suggested call has the endpoints the right way round, so it can be copied.
    assert "frame_window(scan, 7.0, 8.0)[::-1]" in message


def test_reverse_playback_is_still_reachable(sweep):
    """
    What makes refusing cost nothing: `frames=` takes any order, and a range slices,
    so the reversal is one slice away — and visible at the call site.
    """
    window = plotting.frame_window(sweep, 7.0, 8.0)
    assert list(window[::-1]) == [2, 1, 0]

    panel = _ProbePanel(n_frames=3)
    fig, anim = plotting.animate_panels([panel], frames=window[::-1])
    assert list(anim.new_frame_seq()) == [2, 1, 0]


def test_a_distant_coordinate_warns_but_still_resolves(sweep):
    """The scan's own policy, arriving through frame_window unchanged."""
    with pytest.warns(UserWarning):
        window = plotting.frame_window(sweep, 7.0, 40.0)
    assert window.stop == sweep.n_sweeps      # clamped to the nearest, the last frame


def test_a_nest_is_refused(nested_sweep):
    """
    A contiguous index range across a raster is a snake through acquisition order, not
    a region, so a bare coordinate cannot mean what it looks like it means.
    """
    with pytest.raises(ValueError):
        plotting.frame_window(nested_sweep, 0.0, 5.0)


def test_the_window_feeds_animate_panels(sweep):
    panel = plotting.SpectrumLinePanel(sweep, sweep_attr="scanner_y",
                                       show_sweep_title=False)
    window = plotting.frame_window(sweep, 7.5, 8.0)
    fig, anim = plotting.animate_panels([panel], frames=window)

    assert list(anim.new_frame_seq()) == [1, 2]


# ---------------------------------------------------------------------------
# The conjugate top axis
# ---------------------------------------------------------------------------


def _panel_with_twin(x_axis):
    panel = plotting.SpectrumLinePanel(_FakeSweep(), x_axis=x_axis, twin_axis=True,
                                       show_sweep_title=False)
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))
    return fig, ax, panel


def test_no_conjugate_axis_by_default():
    panel = plotting.SpectrumLinePanel(_FakeSweep(), show_sweep_title=False)
    fig, ax = plt.subplots()
    panel.init_artists(ax, range(panel.n_frames))

    assert panel.ax_twin is None
    assert ax.child_axes == []


@pytest.mark.parametrize("x_axis, expected", [
    ("energy",     "Wavelength (nm)"),
    ("wavelength", "Energy (eV)"),
])
def test_the_conjugate_axis_shows_the_other_unit(x_axis, expected):
    """
    Both directions, because a wavelength panel wants eV on top just as much — and
    an unreachable branch would be dead code.
    """
    fig, ax, panel = _panel_with_twin(x_axis)

    assert panel.ax_twin is not None
    assert panel.ax_twin.get_xlabel() == expected


def test_the_conversion_is_the_packages_own_constant():
    """
    Guards against a hardcoded 1239.84 drifting from constants.HC_EV_NM, which is what
    every other energy/wavelength conversion in the package goes through.

    Read off the resulting limits rather than the transform callables, which are a
    matplotlib private. Sorted because the reciprocal reverses the ordering: the low
    energy edge is the long-wavelength one.
    """
    fig, ax, panel = _panel_with_twin("energy")
    fig.canvas.draw()

    expected = processing.energy_to_wavelength(np.asarray(ax.get_xlim()))
    assert sorted(panel.ax_twin.get_xlim()) == pytest.approx(sorted(expected))


def test_the_conjugate_axis_emits_no_divide_warning():
    """
    Matplotlib evaluates the transform across the axis, including 0, where the
    reciprocal is undefined. Without the errstate guard every draw warns.
    """
    fig, ax, panel = _panel_with_twin("energy")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig.canvas.draw()


def test_the_conjugate_axis_ticks_are_round_in_its_own_unit():
    """
    The reason for secondary_xaxis over twiny with relabelled ticks: matplotlib picks
    the ticks in the displayed unit, so the nm labels land on round wavelengths rather
    than wherever the eV ticks happened to fall.
    """
    fig, ax, panel = _panel_with_twin("energy")
    fig.canvas.draw()

    shown = [t for t in panel.ax_twin.get_xticks()
             if panel.ax_twin.get_xlim()[0] <= t <= panel.ax_twin.get_xlim()[1]]
    assert shown, "the secondary axis placed no ticks in range"
    # Round in nm — the whole point. A relabelled eV tick would be 495.9, 550.4, …
    assert all(float(t).is_integer() for t in shown)


def test_the_conjugate_axis_clears_the_shared_title():
    """
    The combination E20 was fixed for: before that, the panel title was lifted above
    the top axis' decorations and straight through the shared title.
    """
    panel = plotting.SpectrumLinePanel(_FakeSweep(), twin_axis=True, cmap="viridis")
    fig, anim = plotting.animate_panels([panel], figsize=(10, 4))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_top     = fig.axes[0].title.get_window_extent(renderer).y1
    shared_bottom = _shared_title(fig).get_window_extent(renderer).y0
    assert shared_bottom >= title_top


# ---------------------------------------------------------------------------
# The AttoCube's trailing white-light frame
# ---------------------------------------------------------------------------
#
# The export writes one more white-light frame than PL frames; the last one has no
# PL frame to pair with. animate_panels cannot fix this — it sees a row of
# ImageSequencePanels and cannot tell which is the white light — so the rule lives in
# animate_wl_pl_spectra, where the argument is named `wl`. These tests build real
# directories because ACImgSweep is type-checked, not duck-typed.


def _image_dir(root, prefix, n_frames):
    """A directory of numeric-grid frame CSVs, named the way the exporter names them."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        np.savetxt(root / f"{prefix}iter_{i}.csv",
                   np.full(SHAPE, float(i)), delimiter=",")
    return root


def test_one_extra_white_light_frame_is_dropped_with_a_warning(tmp_path):
    wl = _image_dir(tmp_path / "wl", "wl_", 5)     # N + 1
    pl = _image_dir(tmp_path / "pl", "pl_", 4)     # N

    with pytest.warns(UserWarning, match=r"takes 4 images out of a possible 5"):
        fig, anim, panels = plotting.animate_wl_pl_spectra(
            wl=(str(wl), "wl_"), pl=(str(pl), "pl_"),
        )

    assert list(anim.new_frame_seq()) == [0, 1, 2, 3]


def test_the_warning_names_the_export_quirk(tmp_path):
    """A bare count would read as a bug in the data rather than a known export habit."""
    wl = _image_dir(tmp_path / "wl", "wl_", 5)
    pl = _image_dir(tmp_path / "pl", "pl_", 4)

    with pytest.warns(UserWarning) as record:
        plotting.animate_wl_pl_spectra(wl=(str(wl), "wl_"), pl=(str(pl), "pl_"))

    assert any("white-light" in str(w.message) for w in record)


def test_a_larger_mismatch_is_refused_not_trimmed(tmp_path):
    """
    Only *exactly one* extra frame is the documented quirk. Two is something else, and
    quietly dropping both would hide whatever it is.
    """
    wl = _image_dir(tmp_path / "wl", "wl_", 6)     # N + 2
    pl = _image_dir(tmp_path / "pl", "pl_", 4)

    with pytest.raises(ValueError, match="disagree on how many frames"):
        plotting.animate_wl_pl_spectra(wl=(str(wl), "wl_"), pl=(str(pl), "pl_"))


def test_fewer_white_light_frames_than_pl_is_refused(tmp_path):
    """The quirk is one-directional; the other way round is not it."""
    wl = _image_dir(tmp_path / "wl", "wl_", 3)
    pl = _image_dir(tmp_path / "pl", "pl_", 4)

    with pytest.raises(ValueError, match="disagree on how many frames"):
        plotting.animate_wl_pl_spectra(wl=(str(wl), "wl_"), pl=(str(pl), "pl_"))


def test_matching_counts_animate_without_a_warning(tmp_path):
    wl = _image_dir(tmp_path / "wl", "wl_", 4)
    pl = _image_dir(tmp_path / "pl", "pl_", 4)

    # Recorded rather than escalated to an error: turning every UserWarning into an
    # exception also catches matplotlib's unrelated "deleted without rendering" one,
    # which is raised from __del__ and so surfaces as an unraisable-exception warning
    # attributed to whichever test happens to trigger collection.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, anim, panels = plotting.animate_wl_pl_spectra(
            wl=(str(wl), "wl_"), pl=(str(pl), "pl_"),
        )

    assert not [w for w in caught if "out of a possible" in str(w.message)]
    assert list(anim.new_frame_seq()) == [0, 1, 2, 3]


def test_an_explicit_selection_overrides_the_trim(tmp_path):
    """The rule is a default, not a policy: naming frames yourself wins."""
    wl = _image_dir(tmp_path / "wl", "wl_", 5)
    pl = _image_dir(tmp_path / "pl", "pl_", 4)

    fig, anim, panels = plotting.animate_wl_pl_spectra(
        wl=(str(wl), "wl_"), pl=(str(pl), "pl_"), frames=range(2),
    )
    assert list(anim.new_frame_seq()) == [0, 1]


# The wrapper builds its panels and hands them back, because a panel's artists are
# how it is restyled — reaching them through fig.axes means guessing which entry is
# whose. spectrum_style is the one route to the spectrum panel's own options, and
# mirrors laser_style, which does the same job for the two image panels.


def _spectra_csv(tmp_path):
    """A three-sweep spectral CSV, matching a three-frame image directory."""
    path = tmp_path / "spectra.csv"
    make_spectral_csv(path)
    return str(path)


def test_the_panels_come_back_in_construction_order(tmp_path):
    pl = _image_dir(tmp_path / "pl", "pl_", 3)

    fig, anim, panels = plotting.animate_wl_pl_spectra(
        pl=(str(pl), "pl_"), spectra=_spectra_csv(tmp_path),
    )

    assert [type(p).__name__ for p in panels] == ["ImageSequencePanel",
                                                  "SpectrumLinePanel"]


def test_an_omitted_panel_is_absent_rather_than_none(tmp_path):
    """The length follows the arguments, so panels[-1] is the spectrum whenever
    spectra was passed."""
    pl = _image_dir(tmp_path / "pl", "pl_", 3)

    _, _, image_only = plotting.animate_wl_pl_spectra(pl=(str(pl), "pl_"))
    _, _, with_spec  = plotting.animate_wl_pl_spectra(
        pl=(str(pl), "pl_"), spectra=_spectra_csv(tmp_path))

    assert len(image_only) == 1
    assert len(with_spec) == 2
    assert isinstance(with_spec[-1], plotting.SpectrumLinePanel)


def test_spectrum_style_reaches_the_panel(tmp_path):
    """twin_axis was unreachable through this wrapper before spectrum_style."""
    pl = _image_dir(tmp_path / "pl", "pl_", 3)

    fig, anim, panels = plotting.animate_wl_pl_spectra(
        pl=(str(pl), "pl_"), spectra=_spectra_csv(tmp_path),
        spectrum_style={"twin_axis": True},
    )
    fig.canvas.draw()

    assert panels[-1].ax_twin is not None
    assert panels[-1].ax_twin.get_xlabel() == "Wavelength (nm)"   # energy is default


def test_no_spectrum_style_leaves_the_panel_at_its_defaults(tmp_path):
    pl = _image_dir(tmp_path / "pl", "pl_", 3)

    fig, anim, panels = plotting.animate_wl_pl_spectra(
        pl=(str(pl), "pl_"), spectra=_spectra_csv(tmp_path),
    )

    assert panels[-1].ax_twin is None


def test_an_unknown_spectrum_style_key_raises_from_the_panel(tmp_path):
    """
    The dict is forwarded, not validated here, so the error names the constructor
    that owns the keyword rather than this wrapper.
    """
    pl = _image_dir(tmp_path / "pl", "pl_", 3)

    with pytest.raises(TypeError, match="twin_axes"):
        plotting.animate_wl_pl_spectra(
            pl=(str(pl), "pl_"), spectra=_spectra_csv(tmp_path),
            spectrum_style={"twin_axes": True},
        )


def test_laser_style_reaches_both_image_panels(tmp_path):
    """
    The pattern spectrum_style copies, previously unpinned anywhere. Both image
    panels get the same dict, so a style set once applies to both.
    """
    wl = _image_dir(tmp_path / "wl", "wl_", 3)
    pl = _image_dir(tmp_path / "pl", "pl_", 3)

    fig, anim, panels = plotting.animate_wl_pl_spectra(
        wl=(str(wl), "wl_"), pl=(str(pl), "pl_"),
        laser_style={"laser_annotation": False, "laser_color": "cyan"},
    )

    assert [p.laser_annotation for p in panels] == [False, False]
    assert [p.laser_color for p in panels] == ["cyan", "cyan"]


# ---------------------------------------------------------------------------
# The shared title
# ---------------------------------------------------------------------------


def test_the_frame_counter_is_shown_by_default():
    fig, anim = plotting.animate_panels([_ProbePanel(n_frames=4)])
    assert _shared_title(fig).get_text() == "Frame 0/4"


def test_the_frame_counter_format_is_overridable():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4)], frame_count_fmt="{frame} of {n_frames}",
    )
    assert _shared_title(fig).get_text() == "0 of 4"


def test_the_counter_names_the_frames_index_in_the_scan():
    """
    A windowed animation captions its frames with the indices they have in the scan,
    so a still lifted from one can be traced back to a file, and two clips of the same
    scan are not captioned identically.
    """
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=100)], frames=range(20, 24),
    )
    assert _shared_title(fig).get_text() == "Frame 20/100"


def test_the_counter_can_report_position_within_the_selection():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=100)], frames=range(20, 24),
        frame_count_fmt="{position}/{n_shown}",
    )
    assert _shared_title(fig).get_text() == "0/4"


def test_panel_labels_join_the_counter():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4, label="V ="), _ProbePanel(n_frames=4, label="P =")],
        suptitle_sep=" | ",
    )
    assert _shared_title(fig).get_text() == "Frame 0/4 | V = 0 | P = 0"


def test_a_panel_without_a_label_contributes_nothing():
    """A ``None`` label is dropped, so the separator never dangles."""
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4, label=None), _ProbePanel(n_frames=4, label="P =")],
        suptitle_sep=" | ",
    )
    assert _shared_title(fig).get_text() == "Frame 0/4 | P = 0"


def test_no_title_artist_when_there_is_nothing_to_say():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4, label=None)], show_frame_count=False,
    )
    assert _shared_title(fig) is None


def test_the_shared_title_tracks_the_frame(tmp_path):
    """
    Blitting only repaints axes artists, so a figure-level title freezes on frame 0
    under ``blit=True`` — verified frozen in both the notebook slider and MP4. The
    engine therefore redraws in full, and this renders to prove the title moves.
    """
    fig, anim = plotting.animate_panels([_ProbePanel(n_frames=4)])
    title = _shared_title(fig)
    _render(fig, anim, tmp_path)

    assert title.get_text() == "Frame 3/4"
    assert title in fig.texts


def test_the_engine_does_not_blit(tmp_path):
    """
    Pins the reason, not just the effect. If this is ever flipped back to ``True``,
    the shared title silently freezes in the notebook slider and in MP4 — the two
    paths that matter most here, and the two a GIF-only check would not catch.
    """
    fig, anim = plotting.animate_panels([_ProbePanel(n_frames=4)])
    assert anim._blit is False


def test_the_header_updates_in_the_notebook_player(tmp_path):
    """
    The workflow this package is actually used through: ``to_jshtml``'s player, whose
    slider steps frame by frame.

    Tested behaviourally rather than by checking the blit flag, because this is the
    path that silently broke — with blitting on, every slider frame carried the
    frame-0 header while the GIF looked fine. ``HTMLWriter`` is what ``to_jshtml``
    uses; ``embed_frames=False`` makes it write the frames as real PNGs, which are
    the pixels the slider shows.
    """
    fig, anim = plotting.animate_panels([_ProbePanel(n_frames=3)], figsize=(6, 3))
    anim.save(str(tmp_path / "player.html"),
              writer=HTMLWriter(fps=4, embed_frames=False))

    headers = []
    for png in sorted(tmp_path.rglob("frame*.png")):
        with Image.open(png) as img:
            img.load()
            headers.append(np.asarray(img.convert("L"))[:40, :].copy())

    assert len(headers) == 3
    # Every frame's header strip must differ from the first: the counter changed.
    assert all(not np.array_equal(h, headers[0]) for h in headers[1:])


def test_a_label_resolved_in_init_artists_still_reaches_the_title():
    """
    Guards the ordering: the suptitle is built after every panel is initialised, so a
    panel that resolves its label in ``init_artists`` is not silently dropped.
    """
    fig, anim = plotting.animate_panels(
        [_LateLabelPanel(n_frames=4)], show_frame_count=False,
    )
    assert _shared_title(fig).get_text() == "resolved 0"


# ---------------------------------------------------------------------------
# Layout — the shared title against the panels' own decorations
# ---------------------------------------------------------------------------


def test_the_shared_title_clears_a_panel_title():
    fig, anim = plotting.animate_panels(
        [_ProbePanel(n_frames=4, title="Panel A")], figsize=(10, 4),
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_top = fig.axes[0].title.get_window_extent(renderer).y1
    shared_bottom = _shared_title(fig).get_window_extent(renderer).y0
    assert shared_bottom >= title_top


@pytest.mark.parametrize("n_panels", [1, 2, 3, 4])
@pytest.mark.parametrize("figsize", [(10, 4), (5, 4), (20, 4), (10, 2.5)])
def test_the_shared_title_clears_a_secondary_top_axis(n_panels, figsize):
    """
    E20's regression guard, parametrised because the old bug was scale-dependent.

    A panel that draws a secondary top axis is the case that broke: the layout engine
    shrinks the panel rather than growing the figure, so a title positioned as a
    fraction of the panel slid down into the panel's own title — measured 20.1 px of
    overlap. A real suptitle is reserved space instead, and clears by ~8 px at every
    panel count and figure size tried.
    """
    panels = [
        _ProbePanel(n_frames=4, title=f"Panel {i}", top_axis=True)
        for i in range(n_panels)
    ]
    fig, anim = plotting.animate_panels(panels, figsize=figsize)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_top = max(ax.title.get_window_extent(renderer).y1 for ax in fig.axes[:n_panels])
    shared_bottom = _shared_title(fig).get_window_extent(renderer).y0
    assert shared_bottom >= title_top
