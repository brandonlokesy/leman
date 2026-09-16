"""
Tests for the small duck-typed adapters that let ImageSequencePanel and
NormalizedSpectrumPanel work with data that isn't a full AttoCube loader
object: TrimmedImageSequence, GridImageSequence, GridSweep.
"""

import numpy as np
import pytest

from leman import plotting


class _FakeImageScan:
    def __init__(self, frames, laser_ref=None):
        self._frames = frames
        self.laser_ref = laser_ref

    @property
    def n_frames(self):
        return len(self._frames)

    def load_frame(self, idx):
        return self._frames[idx]


# ---------------------------------------------------------------------------
# TrimmedImageSequence
# ---------------------------------------------------------------------------


def test_trimmed_image_sequence_subsets_a_contiguous_range():
    base = _FakeImageScan([np.full((2, 2), i) for i in range(5)])
    trimmed = plotting.TrimmedImageSequence(base, range(3))
    assert trimmed.n_frames == 3
    assert [trimmed.load_frame(i)[0, 0] for i in range(3)] == [0, 1, 2]


def test_trimmed_image_sequence_reorders_by_arbitrary_permutation():
    base = _FakeImageScan([np.full((2, 2), i) for i in range(4)])
    reordered = plotting.TrimmedImageSequence(base, [3, 1, 0, 2])
    assert [reordered.load_frame(i)[0, 0] for i in range(4)] == [3, 1, 0, 2]


def test_trimmed_image_sequence_carries_laser_ref_through():
    base = _FakeImageScan([np.zeros((2, 2))], laser_ref="a laser ref")
    trimmed = plotting.TrimmedImageSequence(base, [0])
    assert trimmed.laser_ref == "a laser ref"


def test_trimmed_image_sequence_defaults_laser_ref_to_none_if_absent():
    class _NoLaserRef:
        def load_frame(self, idx):
            return np.zeros((2, 2))
    trimmed = plotting.TrimmedImageSequence(_NoLaserRef(), [0])
    assert trimmed.laser_ref is None


# ---------------------------------------------------------------------------
# GridImageSequence
# ---------------------------------------------------------------------------


def test_grid_image_sequence_wraps_a_prebuilt_stack():
    stack = np.stack([np.full((2, 3), i) for i in range(4)], axis=-1)  # (2, 3, 4)
    seq = plotting.GridImageSequence(stack, laser_ref="ref")
    assert seq.n_frames == 4
    assert seq.laser_ref == "ref"
    for i in range(4):
        assert np.all(seq.load_frame(i) == i)


# ---------------------------------------------------------------------------
# GridSweep
# ---------------------------------------------------------------------------


class _FakeBaseScan:
    energy = np.linspace(1.5, 1.8, 10)
    wavelength = np.linspace(700.0, 750.0, 10)


def test_grid_sweep_exposes_energy_and_wavelength_unchanged():
    spectra = np.zeros((10, 5))
    sweep = plotting.GridSweep(_FakeBaseScan(), best_energy_spectra=spectra)
    assert np.array_equal(sweep.energy, _FakeBaseScan.energy)
    assert np.array_equal(sweep.wavelength, _FakeBaseScan.wavelength)
    assert sweep.n_sweeps == 5


def test_grid_sweep_exposes_arbitrary_sweep_attrs():
    spectra = np.zeros((10, 3))
    sweep = plotting.GridSweep(
        _FakeBaseScan(), best_energy_spectra=spectra,
        scanner_x=np.array([1.0, 2.0, 3.0]), scanner_y=np.array([4.0, 5.0, 6.0]),
    )
    assert np.array_equal(sweep.scanner_x, [1.0, 2.0, 3.0])
    assert np.array_equal(sweep.scanner_y, [4.0, 5.0, 6.0])


def test_grid_sweep_accepts_spectra_alone_for_wavelength_x_axis():
    spectra = np.zeros((10, 4))
    sweep = plotting.GridSweep(_FakeBaseScan(), spectra=spectra)
    assert sweep.n_sweeps == 4
    assert not hasattr(sweep, "best_energy_spectra")
    assert np.array_equal(sweep.spectra, spectra)


def test_grid_sweep_raises_without_either_spectra_array():
    with pytest.raises(ValueError, match="n_sweeps"):
        plotting.GridSweep(_FakeBaseScan())


def test_grid_sweep_works_with_normalized_spectrum_panel():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    spectra = np.tile(np.exp(-((np.arange(10) - 5) ** 2) / 4), (3, 1)).T * [1.0, 2.0, 3.0]
    sweep = plotting.GridSweep(
        _FakeBaseScan(), best_energy_spectra=spectra, scanner_x=np.array([0.0, 1.0, 2.0]),
    )
    panel = plotting.NormalizedSpectrumPanel(sweep, sweep_attrs="scanner_x", smooth_window=None)
    fig, ax = plt.subplots()
    panel.init_artists(ax, panel.n_frames)
    panel.update(1)
    assert panel.frame_label(1) == "scanner_x = 1 V"
