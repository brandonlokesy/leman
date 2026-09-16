# TODO

## Refactoring
- ~~ACPLVabScan needs a rename~~ → `ACSpectralSweep` (2026-07-30).
  Old name kept as a deprecated subclass emitting `FutureWarning`.
- ~~ACPLVabScan needs a rewrite~~ (2026-07-30)
    - ~~accept PL, R, RC etc. spectra~~ (2026-07-30) — `spectra_type=` is
      required, from `constants.SPECTROSCOPY_TYPES`, and drives `signal_label`.
      **R/RC now genuinely load**: reflectance uses the identical 4-column layout
      as PL, so it needed no parser work — only `reference=` and the contrast
      maths (`processing.spectral_contrast`). **TRPL loads too**, via
      `ACTRPLSweep`. Absorption/cavity/BFP still have no loader.
    - ~~check format/dimension of spectra files~~ — `_validate_payload`.
    - ~~verify only one gate is being used~~ — `gate_mode` reports
      dual/top-only/bottom-only and, for dual, whether the two are
      anti-correlated (field-like) or correlated (doping-like); shown in
      `__repr__`. Backed by `varying_parameters()`.
    - ~~accept any independent parameter~~ — one `sweep=` argument taking a
      `_SWEEP_TYPES` key *or* any raw CSV row label. Undeclared means the sweep
      index, never a guessed axis.
    - ~~`roi=1` vs `2`~~ — answered: two spatial ROIs on the CCD, the excitation
      spot and a remote spatially-filtered spot (two-spot galvo scans). Both are
      always loaded; the loader warns when the *selected* one is all zeros.
- Still open on this class:
    - ~~Nested sweeps~~ **done 2026-08-06 (E14).** A 2-D raster arrives as **one
      flattened file** (41×51 in the reflectance example) and a TRPL sweep as a
      **directory**. `sweep_grid()` detects; `fast_sweep=`/`slow_sweep=` declares,
      and `as_grid()` reshapes. TRPL inherits the machinery but has no
      `get_decay_at` yet, and no nested TRPL sweep has been seen.
    - Field × power (a genuinely 2-parameter sweep) still unseen in a real file —
      though it is now expressible, and is what the E14 fixtures model.
- HDF5 export/import: done (`hdf5.py`, `scan.to_hdf5()`, `.h5` accepted by the
  loader, both axis kinds).
- ~~`dev/hdf5`'s `converters.py` remains unmerged~~ **image half landed 2026-08-21.**
  `converters.py` converts real-space image CSVs to TIFF, one file per frame or one
  multi-page stack, and carries the `tmdc-convert` command. It reuses
  `loaders._classify_csv` and `loaders._order_by_iter`; the branch's own copies of
  both reimplemented A9 and A7. Output goes to a `converted/` folder
  (`dev/decisions/0034-converted-files-land-in-a-converted-folder.md`, superseding
  0032 on the name only — `processed/` is for what an analysis extracts, a
  conversion decides nothing). With `out=` a directory run mirrors the source tree
  beneath it (`dev/decisions/0035-out-is-an-output-root-and-the-tree-is-mirrored.md`).
    - ~~Still owed: bulk spectral CSV → HDF5 on the command line~~ **done 2026-08-21.**
      `convert_spectral_csv_to_hdf5` loads with `ACSpectralSweep` and writes
      with `hdf5.write_sweep`, so the package has one archive format and a converted
      sweep reopens in the loader — unlike the branch's private layout.
      `convert_trpl_dir_to_hdf5` collapses a directory of decays into one archive.
      Measured: 4.59 MB → 0.142 MB spectral (32×), and 11.57 MB → 0.070 MB for
      TRPL (165×, the 11.14 MB parameter-table companion being what the archive
      makes unnecessary).
        - `--spectra-type` is the only measurement fact declared at conversion time;
          sweep, gates and geometry are read-time arguments, since the loader takes
          each from its argument and only falls back to stored metadata.
        - A TRPL directory converts only when named, never when reached by
          `--recursive`: `dev/decisions/0033-a-trpl-directory-converts-only-when-named.md`.
- ACImgSweep ...
    - Select the range of frames of interest -> work on a subset
    - ~~**File ordering is lexicographic** (A7)~~ **fixed 2026-08-07.**
      `_order_by_iter` moved out of `ACTRPLSweep` to a module-level helper and
      is now called by both loaders, gap warning included. Escaped notice only
      because the committed exports are zero-padded — and the padding *width* varies
      between them (4 digits in `position-scan/PL`, 6 in `position-scan/wl`), so it
      was never something to rely on. The helper also now warns when two files claim
      the same index (A12) — two acquisitions in one directory, which needs no
      malformed file to happen. ~~**A9** (`_is_image_csv` takes a two-row spectrum for
      an image)~~ **fixed 2026-08-10** — `_classify_csv` replaces it, requires three
      rows for a frame, and defers a headed file to `_read_block_layout` so a TRPL
      export is named `temporal` rather than `spectral`. ~~**B1** (`bg_region` ignored
      in `load_frame`)~~ **fixed 2026-08-10** — `load_frame` still returns the file's
      counts and `load_frame_bg` carries the correction, so `diffusion` keeps
      subtracting exactly once; the viewers pick with
      `frame_source={"best","raw","bg"}`. That completes this class's pass.
- Position in scans (x,y).
    - ~~for x (..) for y (...) can be flipped to for y (...) for x (..)~~ **settled
      2026-07-31, built 2026-08-06 (E14).** The flip is settled *by statement*, not
      read off the data: `fast_sweep=` names the inner loop. A swapped declaration
      is refused, and the message names the swap.
    - ~~Reshape a raster sweep~~ **done (E14)**: `as_grid()` gives
      `(n_points, n_slow, n_fast)` as a view.
    - **Still open: plot it as a map.** `as_grid()` plus `nesting.fast_axis` /
      `slow_axis` are the inputs a `plot_spatial_map` needs; nothing draws one yet.
      Not covered by 0028: that gave `plot_spectral_map` nest pinning, so it draws
      spectra along *one line* of the grid. This entry wants a fitted quantity —
      peak position, intensity — over the whole real-space x/y plane.
- TRPL follow-ups (see E12):
    - No plotting: `_resolve_x_axis` knows only energy/wavelength. Let it ask the
      scan for its own axis instead of adding a third string.
    - No lifetime fitting: `fitting.py` has no exponential model. The baseline
      machinery is already model-agnostic; only `_fit_single_peak` hardcodes the
      3-parameter peak shape.
    - Time unit (ns) is inferred, not confirmed — `_TRPL_TIME_UNIT`. Any fitted
      lifetime inherits it.
- Reference class
    - Format spectra to return array, not per value sweeps

## Fitting
- Multiple ROIs for diffusion E.g. 2-3 diffusion spots. Track each for area and centre of mass
- Multiple ROIs for dipole length. E.g. in hybrid intralayer excitons in homobilayers with switchable dipole lengths based on the transitions

## Documentation
- Add mathematical formulas to functions that assume certain physics. E.g. the calculation of the electric field within the heterostructure stack.
