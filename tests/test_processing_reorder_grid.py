"""
Tests for processing.reorder_grid -- flattens a (..., n_slow, n_fast) grid
(as built by ACSpectralSweep.as_grid) into a flat sequence in a
chosen traversal order and axis direction.
"""

import numpy as np
import pytest

from leman.processing import reorder_grid

N_SLOW, N_FAST = 3, 4
# grid[slow, fast] = the flat index that position came from -- i.e. exactly
# what as_grid() would produce from np.arange(N_SLOW * N_FAST), since a
# sweep is written fast-inner (flat index = slow * n_fast + fast).
FLAT = np.arange(N_SLOW * N_FAST)
GRID = FLAT.reshape(N_SLOW, N_FAST)


def test_default_reproduces_the_original_flat_order():
    assert np.array_equal(reorder_grid(GRID), FLAT)


def test_inner_axis_slow_visits_every_slow_position_before_advancing_fast():
    result = reorder_grid(GRID, inner_axis="slow")
    # First N_SLOW entries must be one full pass over slow at fast=0:
    # flat indices 0, N_FAST, 2*N_FAST, ...
    assert list(result[:N_SLOW]) == [f * N_FAST for f in range(N_SLOW)][:N_SLOW]
    # Reshaping back to (n_fast, n_slow) must match grid.T exactly.
    assert np.array_equal(result.reshape(N_FAST, N_SLOW), GRID.T)


def test_reverse_fast_alone():
    result = reorder_grid(GRID, reverse_fast=True)
    expected = np.flip(GRID, axis=1).reshape(-1)
    assert np.array_equal(result, expected)


def test_reverse_slow_alone():
    result = reorder_grid(GRID, reverse_slow=True)
    expected = np.flip(GRID, axis=0).reshape(-1)
    assert np.array_equal(result, expected)


def test_reverse_both():
    result = reorder_grid(GRID, reverse_fast=True, reverse_slow=True)
    expected = np.flip(np.flip(GRID, axis=1), axis=0).reshape(-1)
    assert np.array_equal(result, expected)


def test_reverse_flags_compose_with_inner_axis_slow():
    result = reorder_grid(GRID, inner_axis="slow", reverse_fast=True, reverse_slow=True)
    flipped = np.flip(np.flip(GRID, axis=1), axis=0)
    expected = flipped.T.reshape(-1)
    assert np.array_equal(result, expected)


def test_leading_dimensions_are_preserved():
    # A spectral-array-shaped grid: (n_pixels, n_slow, n_fast).
    n_pixels = 5
    grid = np.broadcast_to(GRID, (n_pixels, N_SLOW, N_FAST)).copy()
    result = reorder_grid(grid)
    assert result.shape == (n_pixels, N_SLOW * N_FAST)
    for row in result:
        assert np.array_equal(row, FLAT)

    # And an image-shaped grid: (height, width, n_slow, n_fast).
    h, w = 6, 7
    image_grid = np.broadcast_to(
        GRID, (h, w, N_SLOW, N_FAST),
    ).copy()
    image_result = reorder_grid(image_grid, inner_axis="slow")
    assert image_result.shape == (h, w, N_SLOW * N_FAST)


def test_unknown_inner_axis_raises():
    with pytest.raises(ValueError, match="inner_axis"):
        reorder_grid(GRID, inner_axis="sideways")


def test_default_is_a_true_identity_up_to_reshape():
    # Calling reorder_grid with every default should be indistinguishable
    # from just flattening the grid directly -- no copy-related surprises.
    assert np.array_equal(reorder_grid(GRID), GRID.reshape(-1))
