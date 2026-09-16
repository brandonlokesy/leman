"""
Tests for the colormap specification accepted throughout ``plotting``.

``get_cmap`` once took a name and nothing else, and the first thing it did with
that name was ``hasattr(cmc, name)`` — which requires a string, so any colormap
*object* died with ``TypeError: attribute name must be string`` before reaching
``plt.get_cmap``, which has accepted ``Colormap`` instances all along.  That
bare-name lookup is gone: names now go through Matplotlib's single registry,
which the optional packages register into under the ``cmc.`` and ``cmo.``
prefixes.  The shadowing tests below are why — bare lookup silently outranked
Matplotlib for names both sides define.

Seaborn is not a dependency of this package, so the sequence-of-colours cases
here use plain lists.  That is the same code path: ``sns.color_palette(...)``
without ``as_cmap=True`` returns ``_ColorPalette``, a ``list`` subclass of RGB
tuples.  With ``as_cmap=True`` it returns a ``ListedColormap``, covered by the
Colormap cases.
"""

import matplotlib
matplotlib.use("Agg", force=True)      # headless: render without a display

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

from leman import plotting

try:
    from cmcrameri import cm as cmc
    HAS_CRAMERI = True
except ImportError:
    HAS_CRAMERI = False

try:
    from cmocean import cm as cmo
    HAS_CMOCEAN = True
except ImportError:
    HAS_CMOCEAN = False

needs_crameri = pytest.mark.skipif(not HAS_CRAMERI, reason="cmcrameri not installed")
needs_cmocean = pytest.mark.skipif(not HAS_CMOCEAN, reason="cmocean not installed")


def _lut(cmap):
    """
    The colormap's colours, sampled over its full range.

    Colormaps are compared this way rather than by identity or ``.name``:
    Matplotlib's registry hands out a fresh copy per lookup, so ``is`` never
    holds, and every colliding name below is spelled identically on both sides
    (Matplotlib's "gray" and cmocean's "gray" both report ``name == "gray"``).
    The colours are the only thing that differs.
    """
    return cmap(np.linspace(0, 1, 256))


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def test_matplotlib_name_resolves():
    assert isinstance(plotting.get_cmap("viridis"), Colormap)


def test_unknown_name_raises_value_error():
    with pytest.raises(ValueError):
        plotting.get_cmap("nosuchcolormap")


# ---------------------------------------------------------------------------
# Third-party names: prefixed, resolved through Matplotlib's one registry
# ---------------------------------------------------------------------------

@needs_crameri
def test_crameri_prefixed_name_resolves():
    assert np.allclose(_lut(plotting.get_cmap("cmc.vik")), _lut(cmc.vik))


@needs_cmocean
def test_cmocean_prefixed_name_resolves():
    assert np.allclose(_lut(plotting.get_cmap("cmo.thermal")), _lut(cmo.thermal))


def test_importing_plotting_is_enough_to_register():
    """
    Registration is an import side effect, and ``plotting`` imports both
    packages so callers need not.  An import that looks unused is exactly the
    kind a future tidy-up removes, which would break every prefixed name.
    """
    registered = plt.colormaps()
    if HAS_CRAMERI:
        assert "cmc.vik" in registered
    if HAS_CMOCEAN:
        assert "cmo.thermal" in registered


@pytest.mark.parametrize("bare", [
    pytest.param("vik", marks=needs_crameri),
    pytest.param("thermal", marks=needs_cmocean),
])
def test_bare_third_party_name_does_not_resolve(bare):
    """Neither package registers unprefixed names, and nor does this module."""
    with pytest.raises(ValueError):
        plotting.get_cmap(bare)


# ---------------------------------------------------------------------------
# Shadowing — the reason names go through one registry
# ---------------------------------------------------------------------------

@needs_crameri
@pytest.mark.parametrize("name", ["berlin", "managua", "vanimo"])
def test_crameri_does_not_shadow_matplotlib(name):
    """
    Matplotlib 3.10 added these three from the same source as cmcrameri, so both
    define the bare name — and the two versions are not numerically identical.
    Resolving bare third-party names would hand cmcrameri's back to a caller who
    asked for Matplotlib's, under a name that reads the same either way.
    """
    assert np.allclose(_lut(plotting.get_cmap(name)), _lut(plt.get_cmap(name)))
    assert not np.allclose(_lut(plotting.get_cmap(name)), _lut(getattr(cmc, name)))


@needs_cmocean
def test_cmocean_does_not_shadow_matplotlib_gray():
    """
    cmocean and Matplotlib both define "gray", and ``animate_wl_pl_spectra``
    defaults ``wl_cmap="gray"`` — so shadowing here would silently re-colour the
    white-light panel with cmocean's perceptually-uniform version.
    """
    assert np.allclose(_lut(plotting.get_cmap("gray")), _lut(plt.get_cmap("gray")))
    assert not np.allclose(_lut(plotting.get_cmap("gray")), _lut(cmo.gray))


# ---------------------------------------------------------------------------
# Defaults must not need an optional package
# ---------------------------------------------------------------------------

def _default_cmap_names():
    """
    Every string default of a ``cmap``-ish parameter across the module.

    Collected by introspection rather than listed, so a new plotting function
    with a colormap default is covered the moment it is written.  Classes are
    walked too — the animation panels take their colormap on ``__init__``.
    """
    found = []
    for obj_name, obj in vars(plotting).items():
        if obj_name.startswith("_"):
            continue
        target = obj if inspect.isfunction(obj) else (
            obj.__init__ if inspect.isclass(obj) else None
        )
        if target is None:
            continue
        try:
            params = inspect.signature(target).parameters
        except (ValueError, TypeError):        # C-level callables have no signature
            continue
        for p_name, p in params.items():
            if p_name.endswith("cmap") and isinstance(p.default, str):
                found.append(pytest.param(p.default, id=f"{obj_name}.{p_name}"))
    return found


@pytest.mark.parametrize("default", _default_cmap_names())
def test_defaults_need_no_optional_package(default):
    """
    cmcrameri and cmocean are optional extras, so a default naming one of their
    colormaps would make a plotting call fail on a bare install — the default is
    then not really a default.  Both register only under a dotted prefix, so an
    undotted name that Matplotlib resolves cannot have come from either.
    """
    assert "." not in default, f"{default!r} is a third-party prefixed name"
    assert isinstance(plotting.get_cmap(default), Colormap)


def test_introspection_actually_found_the_defaults():
    """Guards the test above: an empty parametrize list would pass vacuously."""
    assert len(_default_cmap_names()) >= 8


# ---------------------------------------------------------------------------
# Colormap objects
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmap", [
    plt.get_cmap("viridis"),
    ListedColormap(["#000000", "#ffffff"]),
    LinearSegmentedColormap.from_list("test", ["red", "blue"]),
])
def test_colormap_passthrough_is_identity(cmap):
    """Returned unchanged, not copied or re-wrapped."""
    assert plotting.get_cmap(cmap) is cmap


# ---------------------------------------------------------------------------
# Sequences of colours
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("colours", [
    ["#1b9e77", "#d95f02", "#7570b3"],          # hex
    ["red", "white", "blue"],                   # named
    [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)],   # RGB tuples
    np.linspace(0, 1, 9).reshape(3, 3),         # (n, 3) array
])
def test_colour_sequence_becomes_listed_colormap(colours):
    cmap = plotting.get_cmap(colours)
    assert isinstance(cmap, ListedColormap)
    assert cmap.N == 3          # one discrete band per colour, no interpolation


def test_bands_use_the_given_colours_verbatim():
    """
    The endpoints of the normalised range map onto the first and last colour, so
    a palette is reproduced rather than resampled.
    """
    colours = [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)]
    cmap    = plotting.get_cmap(colours)
    assert np.allclose(cmap(0.0)[:3], colours[0])
    assert np.allclose(cmap(1.0)[:3], colours[-1])


def test_empty_sequence_raises_value_error():
    with pytest.raises(ValueError, match="empty sequence"):
        plotting.get_cmap([])


@pytest.mark.parametrize("bad", [5, None, {"vik": 1}])
def test_non_colour_input_raises_type_error(bad):
    with pytest.raises(TypeError, match="colormap name"):
        plotting.get_cmap(bad)


def test_sequence_of_non_colours_raises_type_error():
    with pytest.raises(TypeError, match="sequence of colours"):
        plotting.get_cmap(["not a colour"])


# ---------------------------------------------------------------------------
# Reaches an actual plot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmap", [
    "viridis",
    ListedColormap(["#000000", "#ff0000", "#ffffff"]),
    ["#000000", "#ff0000", "#ffffff"],
])
def test_plot_image_accepts_every_form(cmap):
    img          = np.random.default_rng(0).random((8, 10))
    plot = plotting.plot_image(img, cmap=cmap)
    assert isinstance(plot.im.get_cmap(), Colormap)
    plt.close(plot.fig)
