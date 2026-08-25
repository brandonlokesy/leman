# Defect register

Findings register for `tmdc_optics_tools`, from a full read of the package.

**What belongs here:** something that is *wrong* — a crash, a silently wrong number, a
dead parameter, documentation that contradicts the code, duplication. The diagnosis is
kept even after the fix, because it is the record of why the code changed.

**What does not:** the reasoning behind a design choice. That is a record in
`dev/decisions/`, and entries below point at one wherever the fix required a decision.
Rules live in `.claude/CLAUDE.md`; physical quantities in `dev/physics-conventions.md`;
the export format in `dev/instruments/attocube.md`.

Items marked **[verified by running]** were reproduced in the interpreter.

**Name the symbol, not the line.** Entries point at functions, methods and classes —
`plot_diffusion_cloud`, `DiffusionCloudPanel._resolve_var` — because line numbers rot on
the first unrelated edit above them. They had rotted wholesale by 2026-08-13: `plotting.py`
had moved by several hundred lines, so nearly every number in this file pointed at
something else, and the entries most likely to be picked up next (**E16**, **E17**, **E11**)
sent a reader into the wrong function. A number is worth keeping only where it identifies
something a symbol cannot — a specific import line, one call site among several.

Resolved findings keep their number and stay where they are, tagged
**[FIXED — date, commit]** — never deleted, never struck through. `Suggested order`
below tracks what is left.


## Already addressed

- **Selectable fit baselines** (July 2026). All `fit_*` functions now take
  `baseline={"constant"|"linear"|"none"}`, default `"constant"`. Peak models
  previously decayed to zero in their wings, so an un-subtracted dark-count
  pedestal was absorbed by inflating amplitude and FWHM. On the example
  Stark-shift data the offset-free fits were unusable (median FWHM 2401 meV in a
  100 meV window, 42/61 converged, Stark R² = −0.03); with `"constant"` the same
  fits give 17.7 meV and 60/61. `baseline="none"` reproduces the old behaviour.

---

## A. Broken — crashes or silently wrong numbers

*(A1–A3 fixed; A4 deferred; **A5 fixed 2026-08-07**, in the PR #19 merge rather than by a
change aimed at it; **A6 fixed 2026-07-30**; **A8 fixed 2026-08-06**
with E14; **A10 and A7 fixed 2026-08-07**; **A9 and B1 both fixed 2026-08-10**, which
closes the `AttoCubePLScanRealSpace` pass; **A19–A21 fixed 2026-08-17/18** on
`fix/nest-level-separation`; **A23 fixed 2026-08-18**; **A25 and A29 fixed
2026-08-25**, the two Raman-loader entries, taken as two changes.
**A18** and **A22** are the live bugs left in this section, joined by **A24** and
**A26–A28** from the PR #19 review.)*

**A1. `processing.remove_cosmic_rays` cannot be called at all.** **[FIXED — 2026-07-28, e77fabf]**
`remove_cosmic_rays`'s replacement loop referenced `cosmic_mask`, which was never defined — the
variable built at the top is `cr_mask`. Every call raised `UnboundLocalError`, so
the function had never run once. Two further problems in the same loop: `working`
was rebuilt from the raw `spectra` on every pass, which made the iteration a no-op
(a multi-pixel spike has a near-zero Laplacian across its flat top, reachable only
after the edges have been replaced and the Laplacian recomputed), and a bare
`print()` sat inside a library function.

*Fixed:* one mask variable; `working` carried across iterations with replacement
medians taken from it; print dropped; degenerate all-flagged case guarded.

*Beyond the original sketch* — the fix made the function usable for the first time,
so its shape was still open:

- Accepts `(n_pixels, n_sweeps)` with `axis=`, detecting one column at a time. A
  cosmic ray is a single-exposure event, and the MAD noise estimate has to be
  per-exposure because PL intensity moves by an order of magnitude across a gate
  sweep. Results are identical whether spectra are passed singly or batched.
- `cross_sweep_veto=` (default `False`) drops detections that recur at the same
  pixel across sweeps: a cosmic ray cannot repeat, so those are hot pixels or real
  narrow spectral features (Raman, laser leakage, sharp emitters), which the
  3-point Laplacian cannot distinguish from a spike. It only ever removes flags.
- Under the default, warns when a pixel is flagged in more than
  `PERSISTENT_FLAG_FRACTION` of the sweeps, so the conservative default cannot
  remove a real feature silently. See the opt-in design principle in CLAUDE.md.

*Test:* `tests/test_processing_cosmic_rays.py`, 18 cases. The 3-px spike test pins
the iteration fix — at `max_iter=1` only the spike edges are found, pixel 301 is
missed, which is exactly how the pre-fix code behaved on every pass.

**A2. `DeviceGeometry.eps_hs` and `__repr__` raise `AttributeError`.** **[FIXED — 2026-07-30, eca60d4]** **[verified by running]**
`DeviceGeometry.eps_hs` calls `_slabs()`, which returns `(thickness, eps)` **tuples**,
then iterates them as if they were `StackLayer` objects (`layer.thickness`,
`layer.eps`). Because `__repr__` prints `eps_hs`, `print(geom)` is broken for every
geometry.

*Fixed:* unpack the tuples. Tuples are the right representation and `_slabs()` should
keep returning them — an hBN gate flake is ~50 nm rather than *n* monolayers and its ε
lives in `EPS_HBN` not `EPS_TMDC`, so wrapping one in `StackLayer` would mean
asserting a false `n_layers`. Verified on four stack shapes: symmetric hBN,
heterobilayer, single-sided, and bare TMDC.

*Beyond the original sketch* — the fix sketch asked for a decision on
`optical_thickness`, and answering it required tracing the formula back through the
group's MATLAB to the senior's thesis, which settled three further things:

Renamed in the same pass, since "HS" was used with **opposite scopes** in one
package — `eps_hs` meant TMDC + hBN while `EPS_TMDC["HS"]` means TMDC-only. The
geometry properties now follow **one rule: `eps_`/`d_` prefix is the quantity,
`_2d`/`_stack` suffix is the scope**:

| | TMDC layers only | whole gate stack (incl. hBN) |
|---|---|---|
| dielectric constant | `eps_2d` | `eps_stack`  (was `eps_hs`) |
| thickness, nm | `d_2d`  (new) | `d_stack`  (was `heterostructure_thickness`) |

`d_2d` was added because a Stark-derived dipole length is routinely compared against
the physical layer separation; it has no in-repo caller, which is the wrong test for
a read-only accessor on a geometry class. No aliases: a transitional `d_tot` wrapper
was considered and dropped, since a silent synonym doubles the public surface for no
capability and has no mechanism for removal — pre-adoption, the rename is free (cf.
CLAUDE.md on preferring a rename over a compatibility shim). `docs/api/loaders.md`
uses a bare `:::` directive so it needed no change; README §1 updated.

`optical_thickness` was **deleted** rather than renamed. It computed `d_tot × ε_2D`
— full-stack thickness times a TMDC-only ε — behind a local named `d_2d` that
actually held `d_tot`, which is what made it read as a TMDC quantity. It is not an
optical path length (that is `Σ nᵢdᵢ`, and these static out-of-plane ε values cannot
supply `n`: TMDC in-plane `n ≈ 4–5` near resonance, not `√7.2 ≈ 2.7`), and not a
capacitive thickness either (`C ∝ ε/d`, so `d/ε` is the series-additive combination,
never `d·ε`). It was the pre-fix `electric_field` denominator with `ε_hBN` not yet
divided out — `722.16 = 185.17 × 3.9` — i.e. a fragment of one arrangement, with no
callers anywhere. Deleted because the name would have misled the planned
reflectance/cavity work, which needs its own dispersive `n(λ)`.

**`electric_field` was made exact in the same pass** (numerator `ε_hBN` → `ε_stack`),
per Brandon's own derivation from D-continuity. Fields are now ~0.6% higher than all
prior MATLAB and Python results. See `dev/physics-conventions.md` §2
for the two wrong forms and why the old MATLAB's `eps_hs` typo was load-bearing.
The `ValueError` guard for a device with no hBN at all was **kept**, though the exact
form no longer needs it: both-`None` is far more likely a forgotten constructor
argument than a deliberately ungated device, and it would otherwise silently return
a ~100× too-large field.

**A3. Laser circle in `animate_real_space_PL_map` passes a module as a path effect.** **[FIXED — 2026-07-30, 6f1b73a]** **[verified by running]**
In `animate_real_space_PL_map`, `circle.set_path_effects([path_effects])` handed matplotlib the
imported *module* instead of an `AbstractPathEffect`; it fails when the artist is
drawn. The correct call already existed twice elsewhere, `_draw_laser_circle` among them.
*Fix:* delete the line and call the shared `_draw_laser_circle` helper (see D2).

*Fixed:* the eleven-line hand-rolled block is now
`_draw_laser_circle(ax, scan.laser_ref, ls="--")` — one of D2's three drawers gone,
rather than a repaired line. Reproduced first: the old line raises
`AttributeError: module 'matplotlib.patheffects' has no attribute 'draw_path'` on
`fig.canvas.draw()`, because `set_path_effects` only stores the list and the renderer
later calls `draw_path` on each entry. So the failure was at draw time, not at call
time — `laser_annotation=True` returned an `anim` object fine and died on save/render.

Two incidental changes, both from adopting the helper's defaults: `zorder` 10 → 4
(images are `zorder=0`, so the circle still sits above the map, and the helper's 4 is
already proven in `ImageSequencePanel`'s animations — **wrong, corrected 2026-08-18:**
that panel drew at 3, not 4, until it was migrated onto the helper under **D2**; the
conclusion stands on the image being at 0), and the legend label becomes
`"Laser 1/e² (r px)"`. Nothing in this function calls `ax.legend()`, so the label is
only reachable if the caller asks for one. `ls="--"` is passed to keep the dashed
appearance this call site had; the remaining default divergence between the drawers is
D2's to settle.

*Test:* `tests/test_plotting_laser_circle.py`, 8 cases — the first plotting tests in
the suite, so they force `matplotlib.use("Agg")`. Every case renders, because
`set_path_effects` accepts anything and a test that only builds the figure passes
against the broken code. Confirmed by reverting the fix: two cases fail with the
original `AttributeError`, the other six are green either way and cover the helper
plus the two negative paths. `test_module_as_path_effect_fails_on_draw` reproduces
the old call and asserts it raises, so a matplotlib that validated in the setter
would fail loudly rather than leave the regression tests toothless. Hermetic —
duck-typed scan stand-ins, and `PillowWriter` for the end-to-end `save`, so no
`ffmpeg`.

**A4. `diffusion._binary_area` has a wrong `bwarea` weight table.** — *deferred, see below*
`diffusion._binary_area`. MATLAB's `bwarea` weights are 0, ¼, ½ (adjacent pair),
**¾ (diagonal pair)**, **⅞ (three pixels)**, 1. The code puts the diagonal patterns
`6` and `9` in the *adjacent* list at 0.5, then re-assigns them to 0.5 again in a
no-op loop, and gives three-pixel patterns 0.75 instead of 0.875. Every reported
cloud area is biased low, systematically.
*Deferred by decision* — pending a call on whether `bwarea` semantics are wanted at
all, or a plain pixel count would do. Do not fix unprompted.

**A5. `plot_diffusion_cloud` can subtract the background twice.**
**[FIXED — 2026-08-07, `640a108`]** **[verified by running]**
`plot_diffusion_cloud` pulls `image.img` — which is *already* background-subtracted when
the loader was constructed with `bg_region` — and then passes `bg_region` on to
`analyse_diffusion_cloud`, which subtracts again. The guard in
`diffusion._load_image` (which deliberately uses `img_raw`) is bypassed because a
bare ndarray is handed in, not the image object.
*Fix:* pass the image *object* straight through to `analyse_diffusion_cloud` and let
`_load_image` apply its existing rule; use the returned data for display.

*Fixed exactly as sketched*, and it needed one addition to be usable: displaying "the
returned data" requires the result to carry it, so `DiffusionResult` gained an `image`
field holding the array that was actually analysed — background-subtracted where a region
was given, before the ROI mask and the smoothing applied internally for thresholding.
`plot_diffusion_cloud` passes `image` where it used to pass `img`, and draws
`result.image` where it used to draw its own local copy. Both halves are needed: without
the field, every caller re-derives the displayed array from whatever it passed in, which
is the mechanism of this defect one level up.

Landed inside the PR #19 merge (Raman loaders, PL peak fitting, TRPL lifetime fitting)
rather than in a change aimed at this entry, which is why it went unmarked here until
2026-08-20. That merge also carried **A7**, **A8** and **A9**, all three of which had
already been fixed on `main`.

*Test:* `tests/test_diffusion.py`, 5 cases — the repo's first `diffusion` tests. A
second subtraction would drive the background patch to `-BG_LEVEL` instead of leaving it
at ~0, which is asserted in both directions; a bare array still gets exactly one
subtraction, so the fix cannot have been a blanket "never subtract"; and the plot's
displayed array is read back **off the axes** and compared with `result.image`, for both
the computed and the precomputed-`result` paths.

Still open on the same function: **E11** — ~30 parameters and a `result` return instead
of its artists. **A18** and **E19** are the two `DiffusionCloudPanel` items, and both now
have a test file to grow into.

**A6. Zero-filled blocks were loaded as real sweep points.** **[FIXED — 2026-07-30]**
**[verified against the real file]** Found only once a reflectance export existed to
try (E9). The acquisition software over-allocates its header *and fills the surplus*:
`examples/data/reflectance-contrast/sample_26_07_24_17_55_47_iter_0.csv` — the raw
299 MB export, gitignored because GitHub will not take it — declares 4182 blocks of
which only the first 2091 are measurements; blocks 2092–4182 hold literal
`0.0000000E+0` in Par, Wavelength and both ROIs. Those columns are numeric, not
blank, so the all-NaN padding strip never touched them. (Don't write this as
`sample_*.csv`: that glob also matches the committed `sample_truncated_…csv`, which
is a much smaller cut and declares 53 blocks, not 4182.)

The file therefore loaded as `n_sweeps=4182` with 2091 identically-zero trailing
spectra. Every map, fit, `varying_parameters()` and min/max over that scan was wrong,
silently — `Scanner Y` read `0 → 100` only because of the phantom zeros.

*Fixed* by `_drop_unwritten_blocks`, using **the axis column being identically zero**
as the sentinel (sound for both layouts: a spectrometer axis is never all-zero, and a
time axis has only its first bin at zero). Reported via `n_declared_sweeps` and a
`__repr__` line rather than silently. Interleaved zeros — which would mean the format
model is wrong — warn loudly and nothing is dropped. Two subtleties worth keeping:
`NaN != 0` is `True`, so the written-block test must check finiteness or a trailing
NaN row marks every block as written; and in the interleaved case the axis must still
be taken from the first *written* block, or the wavelength axis is all zeros and the
energy axis becomes infinite.

Dropping these is **decoding, not a correction** under *corrections are opt-in*:
there is no data in them to preserve, and keeping them fabricates 2091 measurements
that were never taken. Covered synthetically by `make_spectral_csv(zero_blocks=…)`,
and on real data by the committed `sample_truncated_26_07_24_17_55_47_iter_0.csv`:
a 53-block cut of the raw export — its first 50 measurements plus 3 of its own
zero-filled blocks, renumbered to 50–52 so they stay *trailing* — which loads as
`53 declared, 3 zero-filled and dropped`. Keep the padding trailing if that file is
ever regenerated; interleaved zeros take the warn-and-drop-nothing branch instead.

**A7. `AttoCubePLScanRealSpace` orders its files lexicographically.**
**[FIXED — 2026-08-07]** **[verified against real data]** `AttoCubePLScanRealSpace.__init__`
did `sorted(Path(path).glob(f"{prefix}*.csv"))` and nothing reordered afterwards, so a
sequence numbered past 9 came back as `iter_0, iter_1, iter_10, iter_11, iter_2, …`.
Every frame was then paired with the wrong index — animations play out of order, and
`analyse_diffusion_sequence(var_array=…)` silently mislabels each frame's variable.
Only bites at ≥11 frames; the committed examples escape because they are
**zero-padded**, which is why this never showed up.

The padding is not a property to rely on, and that is the argument for fixing a
dormant bug: **the width varies between exports.** `examples/data/position-scan/PL`
uses four digits (`pl_iter_0000`) and `position-scan/wl` beside it uses six
(`wl_iter_000000`). Alphabetical order is correct only for a fixed width, so the
package was depending on a filename convention it does not control and has already
seen vary. Committed frame counts are 11, 18, 46, 47 and 57 — all past the 10 where
this begins.

**Second caller, 2026-08-21.** `converters.convert_image_dir_to_tiff_stack` orders
its pages with the same `_order_by_iter`, rather than the `sorted(glob(...))` the
unmerged `dev/hdf5` branch used. A stack built in filename order carries the sweep
scrambled inside a single file, where it is harder to notice than a mislabelled
animation. `tests/test_converters.py::test_unpadded_stack_is_in_acquisition_order`
pins it with **unpadded** names, which is the case the branch's own fixture could not
exhibit.

*Fixed* by **moving** `AttoCubeTRPLSweep._order_by_iter` and `_ITER_INDEX` to a
module-level helper (after `_drop_unwritten_blocks`, where the other
"what is this file" logic lives) and calling it from both loaders — not copying it,
which would have made a third near-duplicate of the D-section kind. Three changes to
the moved code:

- **Takes `list[Path]`, not `list[(Path, layout)]`.** Paths are the common
  denominator; the function only ever reads `f.stem`. TRPL re-attaches its layouts by
  dict lookup in two lines, so `_assemble` and `_cross_check_companion` are untouched.
  A `key=` callable was rejected: a parameter existing only because one caller wraps
  its paths.
- **`stacklevel` is a required keyword-only parameter.** The two callers sit at
  different depths below the researcher, so no single value is right for both.
  Real-space passes 3 (correct); TRPL passes 4 **verbatim**, so its warning location
  is unchanged and its four existing tests describe it accurately. See the new item
  below for why 4 is itself wrong and why that was not fixed here.
- Two words in the gap message widened from sweep to export wording, since the helper
  now serves an image sequence too. Both TRPL tests match only the message head, so
  they pass unchanged.

Covered by `tests/test_loaders_real_space.py` (8 tests; 5 fail against the pre-fix
code, and the 3 that pass either way are the did-not-regress ones). Verified on real
data beyond the synthetic tests: the 11 committed padded frames copied to a scratch
directory renamed **unpadded** load as 0…10, and their per-frame sums match the padded
originals element-for-element — so the pairing is right, not merely the filename list.
A9 and B1 followed on 2026-08-10, completing that pass.

**A8. A declared 1-D sweep on a raster gives a sawtooth axis, silently.**
**[FIXED — 2026-08-06]** Reported 2026-07-31. `sweep="piezo_x"` on the reflectance raster **succeeded** and
returns `Scanner X` as the sweep axis: 0→80, repeated 51 times, non-monotonic.
`__repr__` reads innocently (`Sweep : piezo_x — 0 → 80 V`) because it prints only
min and max. Anything plotted against it overplots 51 times, and any fit against it
treats 51 different rows as repeat measurements of one position. (Reported as
`position_x` in µm, before E7's units were settled.)

Worse than an error, because nothing announces it. `sweep_grid()` already detects the
raster (`Scanner X (41) × Scanner Y (51) = 2091`), so the two facts needed to catch
this are both in hand and simply never compared.

*Fix:* at load, warn when the resolved sweep axis is non-monotonic **and**
`sweep_grid()` found a grid, naming the grid and pointing at it. Not a hard error:
one axis of a raster is a legitimate thing to ask for when slicing a single row, and
under *corrections are opt-in* the loader should not refuse a declaration the caller
made. Cheap — both values are already computed in `__init__`.

The proper fix is the deferred grid API (reshape to `(n_points, n_y, n_x)`), which
should also take an explicit `grid=("Scanner X", "Scanner Y")` declaration,
inner-axis first: detection needs `n_inner × n_outer == n_sweeps` exactly and so
fails on an aborted scan with a partial final row, and declaring it settles the x/y
loop-order flip in `dev/TODO.md` by statement rather than inference. **Note that a
raster is not a `_SWEEP_TYPES` entry** — that registry answers "which 1-D array of
length `n_sweeps` is the axis", and a raster has two, so `sweep_axis` has nothing
single to return. It needs the separate mechanism, not a new key.

*Fixed 2026-08-06, with **E14**; widened the same day.* The first fix asked *is
the sweep axis non-monotonic, and is there a nest?* That caught the reported case
and missed two others, because non-monotonicity is a side effect rather than the
cause. `_warn_if_sweep_axis_repeats` asks the cause instead: **does the declared
sweep axis give each point its own value?** A map positions its spectra along that
axis, so points sharing a value land on top of each other and only one is drawn.

What the first version missed:

- **The outer quantity of the same raster.** `sweep="piezo_y"` gives a *staircase*
  — `0 0 0 0 5 5 5 5 10 10 10 10`. It never decreases, so the monotonicity test
  passed it. It is the worse failure of the two: of the eleven strips a
  `pcolormesh` draws between neighbouring points, the sawtooth reverses 2 while the
  staircase collapses **9** to zero height.
- **Nests that are not rasters.** A field × power scan has the identical staircase
  in its power column. Both quantities of *any* nest repeat — that is what nesting
  is — so this was never about rasters.
- **Why the nest guard also went.** `sweep_grid()` searches raw rows for two whose
  distinct counts multiply out. On field × power both gate rows hold `n_fast ×
  n_slow` distinct values, so it finds nothing and returns `None`; a guarded check
  would have stayed silent on exactly the measurement E14 exists for. The check now
  fires on repeats alone. Deliberate repeat measurements at each setting therefore
  warn too, which is correct — their map collapses the same way.

The grid API A8 calls "the proper fix" is E14, which arrived with a different
spelling: `fast_sweep=` / `slow_sweep=`, not `grid=(inner, outer)`. Reasons there.

*Test:* `tests/test_loaders_nesting.py` — both quantities of the raster warn
(parametrised, each naming its own count), field × power warns while
`sweep_grid()` is `None`, an undeclared sweep and a genuinely unique axis are
silent, and `__repr__` distinguishes a declared nest from a detected one.

**A10. `varying_parameters` overflows on a zero-mean row, which is what an
anti-symmetric gate sweep is.** **[FIXED — 2026-08-07]** Reported 2026-08-06.
**[verified by running]** It ranked rows by span relative to their own magnitude:

```python
scale = max(abs(float(np.mean(finite))), np.finfo(float).tiny)
found.append((span / scale, label, ...))
```

`np.finfo(float).tiny` (~2.2e-308) is there so a zero-mean channel does not divide
by zero — but dividing by it instead overflows. For `V_A = tile(linspace(-3, 3, 4),
3)`, whose mean is *exactly* 0.0, the load emits
`RuntimeWarning: overflow encountered in scalar divide` and the row's rank is `inf`.
Where float dust leaves a mean of ~1e-16 rather than 0 there is no warning and the
rank is ~4e16, which is the same failure without the signal.

An anti-symmetric gate pair sweeping the displacement field at fixed density is a
routine measurement here, so this is the ordinary case, not a corner. Two
consequences, both silent:

- **The ordering stops meaning anything.** Both gates rank `inf`, so they cannot be
  separated from each other and every genuinely-varying row sorts below them.
- **`sweep_grid()` inherits it.** Its candidate loop iterates `varying_parameters()`
  in order and returns the *first* pair that verifies, so on a raster taken during
  an anti-symmetric sweep it reports `V_B (4) × Scanner Y (3)` rather than the
  scanner pair — both nest identically, and the meaningless ranking picks.

Neither is fatal now that a nest is **declared** rather than detected (E14), which
is why this is reported rather than folded into that pass: `sweep_grid()` is a
diagnostic, and its answer here is true, merely not the useful one.

*Surfaced by* the E14 fixtures, which used an anti-symmetric pair because that is
the case E14 exists to support.

*Fixed 2026-08-07 — scale by RMS.* `sqrt(mean(x**2))` is a magnitude rather than a
centroid, so it cannot vanish for a row that is not identically zero, and a row
straddling zero is measured by how large it is rather than by how nearly it
cancels. The `max(..., tiny)` floor stays as belt and braces; a zero-RMS row has
zero span and never reaches the division. Threshold and ranking use the one scale,
so they cannot disagree about what counts as noise.

It preserves the distinction the comment describes, which is the test that
mattered:

| row | span | old `span/scale` | new | verdict |
|---|---|---|---|---|
| 10 V gate, 1 mV wobble | 1e-3 | 1.0e-4 | 1.0e-4 | noise, unchanged |
| 2 mV channel, 1 mV wobble | 1e-3 | 0.400 | 0.392 | sweep, unchanged |
| `[-3,-1,1,3]`, mean exactly 0 | 6 | **`inf`** | 2.683 | sweep, now finite |
| `linspace(-3,3,6)`, mean 1.5e-16 | 6 | **4.05e16** | 2.928 | sweep, now sane |
| static | 0 | — | — | excluded by span, unchanged |

*What this does **not** fix, stated because the diagnosis above over-claimed.*
`sweep_grid()` still reports `V_B (4) × Scanner Y (3)` on that raster. The ranking
is no longer meaningless — the gates really did sweep their whole range while the
scanners stepped through part of theirs, so 2.683 above 1.604 is the metric working
— but two pairs verify equally well and the loop returns the first. That residue is
inherent: nothing in the rows says which pair the experiment was *about*, which is
the whole reason a nest is declared rather than detected (**E14**). `sweep_grid()`
is a diagnostic and its docstring says so.

*Test:* `tests/test_loaders.py`, 2 cases — an exactly-zero-mean pair ranks finite
and in order under `simplefilter("error")`, so a numpy overflow would fail the test;
and the wobble-versus-sweep distinction is pinned on both sides.

**A9. `_is_image_csv` accepts a two-row spectrum as an image.**
**[FIXED — 2026-08-10, 679ebf9]** **[verified against real data]** Reported
2026-07-31. `loaders.py:_is_image_csv` tested only whether the first line parses as
floats, which a `SingleSpectrum` CSV does (its first row is the wavelength axis). So
`AttoCubePLScanRealSpace` pointed at a directory of single spectra loaded each as a
2×N "image" — a two-row frame that every downstream diffusion and animation routine
would happily process into nonsense.

Note the contrast with `_read_block_layout`, which was given exactly this
discrimination on 2026-07-31 — a bare grid of two rows is named as `SingleSpectrum`,
more than two as an image sequence — so the rule to copy already existed in the
package.

**Second caller, 2026-08-21.** `converters` selects frames with the same
`_classify_csv`, so a spectrum sitting beside the frames is reported as skipped
rather than written out as a two-pixel-tall TIFF, and `convert_image_csv_to_tiff`
refuses one outright using the existing `_CSV_KIND_REASON` wording. The helper moved
from a `staticmethod` on `AttoCubePLScanRealSpace` to module level in that change so
a second caller could reach it. Pinned by
`tests/test_converters.py::test_spectrum_beside_frames_is_not_converted`.

**Fixed by replacing the predicate with a classifier.** `_is_image_csv` is gone;
`_classify_csv` returns the kind, because a bool cannot carry the reason a file was
skipped and *saying what was skipped* was half the fix. Six kinds: `image`,
`spectrum`, `too_short`, `spectral`, `temporal`, `unrecognised`, `unreadable`.

Three decisions worth recording, in the order they were made:

- **A numeric grid needs three rows.** The row-count peek is bounded — one line plus
  at most two — so a classifier run over a directory costs the same on a 300 MB
  export as on a small one. The bounded peek is now `_n_rows_upto`, module-level and
  *shared* with `_read_block_layout` rather than copied out of it; the D-section rule
  says the second copy is the bug.
- **A headed file defers to `_read_block_layout`.** The first draft had `_classify_csv`
  return `"spectral"` for anything with a text header. Brandon caught it: `"spectral"`
  and `"temporal"` are the two `_BLOCK_LAYOUTS` entries, so a wider meaning for the
  same word forks the vocabulary and mislabels a TRPL export. Delegating also lets the
  message name the right loader via `_CLASS_FOR_KIND`, and gives `"unrecognised"` for
  a header matching no known layout — a different acquisition version, which the
  format-fixed 57-row record says is worth naming rather than dropping.
- **Only surprising skips warn.** An AttoCube export beside the frames is what an
  acquisition writes every time — every committed example directory has one — so
  warning there would fire on every legitimate load and would have broken
  `test_committed_sequence_loads_without_warnings`. `_EXPECTED_NON_IMAGE_KINDS` is
  therefore `frozenset(_CLASS_FOR_KIND)`, and everything outside it is named. When
  nothing survives, the raise lists each candidate with its reason and its loader.

*Test:* `tests/test_loaders_real_space.py`, 8 new cases (10 → 18): the three-row
boundary, the three header kinds, a directory of single spectra raising with
`SingleSpectrum` named, a temporal export naming `AttoCubeTRPLSweep`, a stray spectrum
among frames warned by name, an export among frames staying silent, and
`caught[0].filename == __file__` for the new `stacklevel=2` — measured, per A11,
rather than read off the call stack. The two committed-data tests pass unchanged.

Verified on real data: over `examples/data`, the committed
`ref_single_spectrum_26_07_01_14_42_47.csv` now classifies `spectrum` where it used to
classify as an image, the TRPL files classify `temporal` rather than `spectral`, and
the 11 stark-shift frames plus the laser and white-light references all classify
`image`.

**B1** followed the same day and completed this class's pass. In the end it did *not*
change what `load_frame` returns — the correction went into a sibling accessor, which
is what kept `diffusion` correct.

**A11. Warning locations across `loaders.py` are unverified.** — *open, reported
2026-08-07* All 15 `warnings.warn` calls hand-tune `stacklevel`, spread across the
values 2, 3, 4 and 5, and not one has a test pinning where it points. The one chain
that has been traced is wrong: for `AttoCubeTRPLSweep`, `__init__` →
`_decode_and_describe` → `_decode` → `_decode_dir` → `_order_by_iter` needs **6** to
reach the researcher's line, and passes 4, which lands inside
`_decode_and_describe` instead.

Two consequences, and the second is the one that bites. The line number is useless —
a notebook loading eight scans in a loop cannot tell which one complained. And
because Python's default filter shows a warning **once per location**, eight scans
warning the same way from one library line print **one** message, not eight; pointing
at the caller's line instead makes each of them a distinct location. A wrong
`stacklevel` therefore suppresses repeats, which is a diagnostics failure rather than
a cosmetic one.

Not folded into A7 by decision: A7 passes TRPL's existing 4 verbatim so that its
warning location is unchanged and its tests stay honest. *Fix:* one deliberate pass —
trace each of the 15 chains, set each value, and pin each with a
`caught[0].filename` assertion of the kind `test_warning_points_at_the_caller` now
uses. Not a character changed in passing.

*Second confirmed chain, and the mechanism behind it (2026-08-10).* `_nearest`'s
out-of-range warning reaches the caller correctly from `nearest_index` (`depth=4`)
but not from `get_spectrum_at`, where `_index_for_value` passes `depth=5` and the
warning lands on the `self._sweep_selector(...)` call inside `get_spectrum_at`
itself. **The uncounted frame is the `locate` lambda** inside `_sweep_selector`:
`_sweep_selector` wraps `_index_for_value` in a closure before calling it, so the
chain is `_nearest` → `_index_for_value` → *lambda* → `_sweep_selector` →
`get_spectrum_at` → caller, which needs **6**. Measured, not inferred — a
`catch_warnings(record=True)` harness over all three entry points reports
`nearest_index` → the caller's own line, `get_spectrum_at` → a line inside the loader rather than the caller's.

Worth keeping because a lambda is invisible when the chain is traced by reading
`def` lines, which is how the other 14 values will be checked. Anything routing
through `_sweep_selector` inherits it and needs one more again:
`plotting.plot_spectrum` grew value-based selection on 2026-08-10 via a
`_select_sweep_point` helper, so its chain needs **7** and it lands inside
`_select_sweep_point` itself. That site is listed here rather than fixed locally — correcting
one chain by hand while the shared `depth=` stays wrong would just move the defect.

**A12. `_order_by_iter` could not see duplicate iteration indices.**
**[FIXED — 2026-08-07]** **[verified by running]** The gap check,
`set(range(seen[0], seen[-1] + 1)) - set(seen)`, is silent on duplicates by
construction — it compares against the *set* of indices, which a repeat leaves
unchanged — and CPython's stable sort leaves a colliding pair in glob order.

**This was first written up as unreachable, and that was wrong.** The claim was that a
duplicate could only arrive if a classifier had already failed (`_read_block_layout`
for TRPL, the image classifier now called `_classify_csv`), which would have made a guard here a
symptom-catcher for **A9**. It takes no classifier failure at all: **two acquisitions
copied into one directory** is enough, and every file in that directory is a
legitimate export that the classifier is right to accept. Demonstrated —
`pl_iter_0..2` beside `pl_run2_iter_0..2` loaded as six frames interleaved
`0, 100, 1, 101, 2, 102` with **no warning of any kind**. That is precisely the
mispairing A7 exists to prevent, arriving by another route, so the two belong
together.

*Fixed* by grouping the sorted files by index and warning when any index has more than
one, naming the colliding files rather than counting them — with two runs merged, the
filenames are what identify the cause and "narrow the prefix" is the actual remedy.
Nothing is dropped and no winner is picked: choosing which duplicate survives is a
decision about the data taken on the researcher's behalf, which *corrections are
opt-in* forbids, and it would hide the merge instead of reporting it. Refusing was
considered and rejected — the sibling conditions in this same function warn, as does
`_drop_unwritten_blocks`'s interleaved-zeros branch, and unlike the `gates=` case
nothing here is being guessed: the files are unambiguous, the directory is.

Independent of the gap warning in both directions, and both can fire on one load.
Covered by `test_two_runs_in_one_directory_warn_on_collision`,
`test_collision_and_gap_both_warn` (real-space) and
`test_duplicate_iteration_index_warns` (TRPL). No committed export triggers it: the
TRPL companion collides on `iter_0` with the first decay but is a *spectral* file
separated out by content before the sort, and the timestamped spectral export in each
real-space directory is excluded by `_classify_csv` — both pinned by the existing
no-warning tests.

**A13. A declared nest was refused when a level was wider than the axis tolerance.**
**[FIXED — 2026-08-07]** **[verified against real data]** Reported 2026-08-07 from the
uncommitted export `PL_Vbot_power_sweep_test_3_…_iter_0.csv` — a 5-point `V_A` sweep
inside 3 power levels, which
`fast_sweep="bottom_voltage", slow_sweep="power"` refused with *"the counts multiply
correctly, but the values do not repeat in a regular nest"*.

`_nest_shape` used **two different definitions of "the same grid point"** and applied
one tolerance to both. `_count_distinct` clusters by *consecutive sorted gaps*, and
got the 3 power levels right. The hold-still check was
`allclose(slow_grid, slow_grid[:, :1], atol=slow_atol)` — distance to the row's
**first element**. A cluster whose internal spread exceeds `atol` still counts as one
level and then fails the `allclose`.

The scale is the reason it bites in practice: `_axis_atol` is `_NEST_RTOL ×` the
axis's **full span**, but what the check measures is **within-level scatter**.
Excitation power is read back from a meter, so the scatter is roughly a fixed
fraction of each reading while the tolerance is fixed by the largest — on a
log-spaced power sweep the topmost level always fails first, and more decades makes
it worse. Measured on that file: top level max deviation from its first element
0.0338 µW against an `atol` of 0.0303 µW, over by 1.12×, while the levels were
separated by **239×** their own scatter. Loosening `_NEST_RTOL` would have papered
over it at the wrong scale.

Second-order but worse to debug: `_nesting_failure`'s swap branch calls the same
`_nest_shape`, so it failed too. The reporter's declaration *was* inverted — power is
the slow axis — and the tolerance defect suppressed exactly the message that would
have said so, leaving the generic mismatch text.

*Fixed* by reducing each axis to level indices once, in `_level_labels`, and deciding
the nest on those: the structural test is then exact integer comparison and `atol`
is applied in one place. `_count_distinct` is now a wrapper over the same helper, so
the two definitions cannot drift apart again. Non-finite entries label `-1`, which
reproduces the old `equal_nan=True` tiling and the old refusal of a non-finite slow
coordinate. Covered by `test_a_level_wider_than_the_axis_tolerance_still_nests` and
`test_a_wobbling_level_still_names_the_swap`; the fixture's top level is deliberately
wider than the tolerance while every gap inside it stays narrower, and the old
algorithm run against it returns `None`, so neither test passes for free.

Known and unchanged: the clustering is single-linkage, which `_count_distinct` has
always been. An axis whose scatter approaches its level spacing will chain levels
together — the nest then refuses loudly on a wrong `n_fast` rather than verifying
something wrong.

**A14. `x_axis=` accepted anything at four of six entry points.**
**[FIXED — 2026-08-11]** **[verified by running]** Found 2026-08-10 while reviewing
`pixel_slice`. Only `plotting._resolve_x_axis` and `pixel_slice` refused an
unrecognised value. `loaders._resolve_spectra` and `fitting.fit_scan_peak` both spelt
it `... if x_axis == "energy" else <wavelength>`, so the wavelength branch was the
`else` of a two-way test on a free-form string: **anything that was not exactly
`"energy"` selected the wavelength array**. `scan.get_spectrum_at(2.5, x_axis="eV")`
returned wavelength-space counts and `fit_scan_peak(x_axis="eV")` fitted the
wavelength axis, both without a warning. `get_spectrum_by_index` inherited it through
`_resolve_spectra`.

Two properties made it worth fixing rather than noting. The failure is *silent and
plausible*: the arrays have the same shape and the same dtype, and PL counts against
nm look like PL counts against eV until the numbers are read. And the value is
typically a literal typed at the call site, so `"eV"`, `"nm"`, `"Energy"` — the three
spellings a researcher actually reaches for — were all accepted as "wavelength". The
five plotting sites that branch the same way were never exposed, because each calls
`_resolve_x_axis` first; that is why the defect sat in the two modules with no such
call.

*Fixed* by making the vocabulary data. `constants.X_AXES` holds the two axes as
`(name, unit)` — the same shape as `SIGNAL_LABELS`, so a label composes the same way —
and `_x_axis_name_unit` is the one check, with the message derived from the table so a
key added there cannot go unmentioned in it. Wired to all four sites plus
`_resolve_x_axis`, which now composes its two labels from the table instead of
hardcoding them. Each caller still picks its own arrays: the mapping is not uniform
(`fit_scan_peak` pairs the energy axis with `best_energy_spectra` and the wavelength
axis with raw `spectra`), and `getattr(scan, x_axis)` would have degraded the refusal
to an `AttributeError` on a duck-typed object.

Covered by `tests/test_x_axis_vocabulary.py`: the table's rows, the derived message,
an unhashable axis, and all five entry points refusing `x_axis="eV"`. The message
wording is unchanged from `pixel_slice`'s own, so the assertion in
`tests/test_loaders_pixel_slice.py` still holds at its original site.

Deliberately not covered: **a source named explicitly is served on its own axis**, so
`get_spectrum_at(source="raw", x_axis="energy")` still only *warns*. `x_axis` resolves
what `"best"` means; it is not a claim about a named source, and
`_resolve_spectra`'s docstring said it raised there when it does not. Corrected, and
pinned by a test, so the hole is recorded rather than implied.

**A15. `fit_scan_peak(x_range=)` masked instead of slicing, and refused nothing.**
**[FIXED — 2026-08-11]** **[verified by running]** Found 2026-08-10 alongside A14. The
window was a boolean mask, `(x >= x_range[0]) & (x <= x_range[1])`, with three
consequences:

- **An empty window died inside numpy.** `x` became shape `(0,)`, and the first sweep
  reached `_peak_amplitude_p0`'s `y.max()` — `ValueError: zero-size array to reduction
  operation maximum which has no identity`, preceded by four un-suppressed
  `RuntimeWarning`s (`Mean of empty slice`, `invalid value encountered in scalar
  divide`) from the baseline seed. Nothing in that names `x_range`, the axis, or the
  span the window missed, and it aborts mid-loop rather than up front.
- **A reversed pair was the same crash.** `(1.42, 1.30)` gives an all-False mask, so
  the most likely typo landed on the least informative error.
- **A bound outside the axis was clipped in silence**, so a window chosen for one scan
  and reused on another with a narrower axis fitted a different range with no notice.

*Fixed* by routing the window through `processing._window_slice` — the helper
`pixel_slice` already uses — so the refusal, the clip warning and the
order-insensitivity are the same as everywhere else, and the window is a `slice`
rather than a mask. `slice(None)` covers the no-window path, so both paths are one
line of code. Note that this makes `FitResult.x_fit` a **view** into the scan's axis
where it used to be a copy; documented on the `Returns`, consistent with `pixel_slice`
and the accessors. It removes one allocation per sweep, not two — `.astype(float)`
copies the counts regardless — so the change is worth making for the refusal and the
single spelling, not for memory.

`stacklevel` is the part worth reading before touching this. `fit_scan_peak` is
reachable at two depths — directly, and through `extract_dipole_length` →
`_prepare_dipole_data` — and the example notebook uses the second, so one hardcoded
value would have blamed `fitting.py` on the documented workflow. The public function is
now a wrapper over `_fit_scan_peak`, which takes a **required** `stacklevel`, following
`_order_by_iter` and `_window_slice`: 4 from the wrapper, 5 from the dipole chain.
Both **measured** — `_window_slice` is frame 1 and `_fit_scan_peak` frame 2, verified
by calling at depths 2 through 5 and reading the blamed file, since 2 blames
`fitting.py` and 4 and 5 land on `sys:1` when called directly. This is a new site got
right at both depths, not one of A11's 15; A11 is untouched.

A window of 1 to 3 pixels is non-empty and so passes the helper, but still fails: with
fewer points than free parameters `curve_fit` raises `TypeError`, which
`_fit_single_peak`'s `except RuntimeError` does not catch, so it escapes as neither a
fit nor a non-converged placeholder. Now refused in `_fit_scan_peak`, naming both
counts and the model. The minimum belongs to the *model* rather than the axis, which is
why it is not a `min_points=` argument on `_window_slice` — `pixel_slice` returns
positions and rightly has no minimum. No advice to drop the baseline: that is a model
term, not a knob to trade for a window.

Covered by `tests/test_fitting_scan_window.py`, the first test file for `fitting.py`:
the window fits exactly the pixels inside it and gives the same numbers as a
hand-masked `fit_lorentzian` (the invariant the mask spelling used to provide), the
view is asserted with `shares_memory`, and both warning depths assert
`filename == __file__`. Its fixture carries a synthetic Lorentzian because the suite's
default `roi1` is a monotone ramp on which nothing converges, with a little
fixed-seed noise so an exact fit does not give the vanishing covariance the WLS
weights divide by.

**A16. A declared cosmic-ray repair never reached the wavelength axis.**
**[FIXED — 2026-08-11]** **[verified by running]** Found 2026-08-10 while auditing the
`x_axis` branches for A14. Every energy-space array is built from `spectra_cr`
(`signal = ...` in `__init__`), and E13 says the repair feeds the fits — but four
consumers chose their array with
`best_energy_spectra if x_axis == "energy" else spectra`, so on the wavelength axis they
served the file's own counts and the repair was dropped: `plot_spectral_map`,
`plot_spectrum`, `SpectrumLinePanel.init_artists` and `fit_scan_peak`. The same scan
therefore showed the repair on one axis and the spikes on the other, and
`fit_scan_peak(x_axis="wavelength")` fitted the spikes.

**Root cause was an asymmetry in the class, not four slips.** `AttoCubeSpectralSweep`
had `best_energy_spectra` and no wavelength-space counterpart, so each of those
expressions names a *property* on one side and a *specific array* on the other — they
read as symmetric and are not. `plot_single_spectrum` is the control: it spells the same
line as `spectrum.best_spectra` and was correct, because `SingleSpectrum` has that
property. The class with the accessor had correct callers.

The comment that shipped with the oldest copy is worth keeping in mind, because it was
**true when written**: *"raw spectra for wavelength axis (BG correction is a loader
concern)"*. Wavelength space genuinely had nothing better to offer — the
background-subtracted wavelength array is a local (`corrected`) that is never stored.
E13 then added a wavelength-space correction that does exist. `_resolve_spectra` was
updated for it; the hand-rolled copies were not, which is the duplication cost stated in
CLAUDE.md — the second copy is the bug, not the maintenance burden.

*Fixed* by adding `best_spectra` (repair if declared, else the file's counts), pointing
`_resolve_spectra`'s `"best"` branch at the pair — which also stops it passing over a
`SingleSpectrum`'s `spectra_bg`, contrary to its own docstring — and routing the three
plotting sites through `_resolve_spectra`, which removes the signal-array `x_axis`
ternary from `plotting.py` entirely. `fitting` uses the two properties directly: it must
not import from `loaders`, for the layering reason in the `_window_slice` move.

Covered by `tests/test_cosmic_rays_downstream.py`, which reuses the loader tests' spike
recipe. Its first test asserts the planted spike is *actually flagged*, because
`spectra_cr` would otherwise equal `spectra` and every later assertion would pass while
testing nothing — the same hazard that let this defect live: no fixture anywhere else
declares `cosmic_rays=` alongside a plot or a fit, so the bug was untested in both
directions and no shipped notebook or README combines the two.

Incidental find, unrelated to the repair but visible from the test: `plot_spectral_map`'s
`median_kernel` still defaults to **3**, so its mesh is smoothed across sweeps unless
told otherwise. The test passes `median_kernel=1` to compare arrays. Still the open item
in CLAUDE.md's *Known issues*.

*Update, 2026-08-21.* That default is now `1`, under **E3**, and the CLAUDE.md entry is
gone. `_map_column` in `tests/test_cosmic_rays_downstream.py` still passes
`median_kernel=1` explicitly, which is now redundant but harmless — it states what the
comparison needs rather than relying on a default.

**Not fixed, and its own change:** there is no background-corrected wavelength-space
array, so `bg_region_nm=` shows on the energy axis only, and `best_spectra` therefore
means "repaired" on a sweep and "background-subtracted" on a `SingleSpectrum`. Closing
that means storing `corrected` as `spectra_bg`, which adds a public array and changes
wavelength-axis numbers for every scan loaded with a background — see the array-count
decision in `dev/decisions/0001-cosmic-ray-repair-at-load-time.md` before doing it.

**A17. The two spectral axes carried different sets of arrays.**
**[FIXED — 2026-08-11]** **[verified by running]** Asked for 2026-08-11, closing what
A16 left open. Two constructs caused it, and each denied one axis an array the other
had:

```python
signal    = self.spectra if self.spectra_cr is None else self.spectra_cr   # :3441
corrected = signal                                                        # :3469, a LOCAL
```

The first folded the repair into the array `energy_spectra` was built from, so
`energy_spectra` meant *raw or repaired depending on a constructor argument* and there
was no way to reach the file's own counts on the energy axis. The second built the
background-subtracted wavelength array, stored only its energy-space transform, and
discarded it — so `bg_region_nm=` was visible on the energy axis alone. `SingleSpectrum`
meanwhile builds `energy_spectra` from raw always, so the two classes disagreed about
what that attribute meant whenever a repair was declared.

*Fixed* by making it one cumulative ladder per axis, mirrored: `spectra` / `spectra_cr`
/ `spectra_bg` and `energy_spectra` / `energy_spectra_cr` / `energy_spectra_bg`, each
rung the one above it plus one correction, `None` when that correction was not
requested. Five stored arrays become seven. Both `best_*` become the same three-way
preference and pick the **same rung** — the invariant whose absence produced A16.

Three consequences worth stating plainly:

- **`best_spectra`'s values moved.** It never saw a background before, so every
  wavelength-axis consumer routing through `"best"` — the three plotting entry points,
  `fit_scan_peak(x_axis="wavelength")`, both accessors — now shifts by the pedestal.
  That is the point of the change; `best_energy_spectra` returns the same numbers as
  before.
- **`energy_spectra` changed meaning** for a scan loaded with `cosmic_rays=`: it is now
  the file's counts. Reachable directly and, until the source vocabulary was re-keyed,
  as `spectra_source="energy"`.
- **`energy_spectra_pre_jacobian` stays a single array**, of the first rung. It cannot
  become a property: it is a `spectra_source` key, and `get_spectrum_at` promises a
  *view*, which a freshly-built array cannot honour. It needs no `_cr` / `_bg` variants
  either — the wavelength rungs hold exactly those values, which is precisely what
  adding them buys.

**This amends E13 and the *Rejected* section of
`dev/decisions/0001-cosmic-ray-repair-at-load-time.md`**, which refused "a third flag with
its own energy-space arrays" and closed with "no new energy-space names". The refusal
was of an independent flag **crossed with** the other two — a 2×2×2 lattice, ten names,
`best_*` having to choose a branch. This keeps one fixed order (`raw → cr → bg`), so
`energy_spectra_cr` is not a new branch but the energy view of a rung that already
existed, the count goes 5 → 7, and `best_*` stays a total order. What that refusal got wrong is
that "no new energy-space names" was not free: it was paid for by letting one attribute
name two different quantities depending on a load-time argument, which is the same
defect class as A16 — a name that reads as symmetric and is not. The CLAUDE.md bullet
was edited rather than outgrown.

Covered by `tests/test_spectra_ladder.py`, parametrised over the four declarations
(neither correction, each alone, both) so the table is pinned as a table: which rungs
exist, that each energy rung is its wavelength rung argsorted, that both `best_*` pick
the same rung, that the first rung is the file's counts even with a repair declared, and
that `spectra_bg` is the pedestal taken off the *repaired* counts. `tests/test_
jacobian_background.py` needed no change — it already asserted the pre-Jacobian array
tracks `spectra`, so it pinned the post-change semantics.

**The source vocabulary was re-keyed in the same pass**, and that is what made the
`energy_spectra` change safe rather than silent: `spectra_source="energy"` would
otherwise have kept working and quietly served a different array. `_SPECTRA_SOURCES` now
maps a correction state to `(wavelength attr, energy attr)` and `x_axis` picks the
column, so `"raw"` / `"cr"` / `"bg"` / `"contrast"` each reach either axis and the two
arguments cannot disagree. Retired: `"energy"`, `"energy_bg"`, `"contrast_wavelength"`,
and `"energy_pre_jacobian"` → `"pre_jacobian"`, which is energy-only and now *raises* on
the wavelength axis naming the three states that live there.

Three things fell out of that:

- **The raw-on-energy-axis warning is deleted.** It caught "you named a wavelength-space
  array and are plotting it against energy", a combination that is now unrepresentable.
  A condition that cannot arise needs no warning, and the one mismatch that survives
  refuses with actionable advice instead.
- **An absent source distinguishes two failures.** `getattr(scan, attr, None)` could not
  tell *the class has no such correction* from *you did not ask for it* — and a
  `SingleSpectrum` takes no `cosmic_rays=`, so the old message would have advised passing
  an argument that does not exist. `hasattr` first, then `_SOURCE_REQUIRES`, which names
  the exact argument rather than guessing between two.
- **`_SPECTRA_SOURCE_LABELS` is deleted** — imported by `plotting` and read nowhere, a
  second drifting copy of the vocabulary. `plotting`'s `startswith("contrast")` becomes
  an equality check, the prefix having existed only because one quantity had two
  spellings.

Not closed: `SingleSpectrum` still has no `energy_spectra_pre_jacobian` and no repair
rung (it takes no `cosmic_rays=`, deliberately). Nothing reaches the former.

**A18. `DiffusionCloudPanel` never checks `var_array` against the frame count, so a short
one crashes mid-animation.** `_resolve_var` (`DiffusionCloudPanel._resolve_var`) does
`np.asarray(arr)[:n_frames]`, which truncates a *long* array silently and accepts a
*short* one without comment. `_frame_title` then indexes `self._var_array[frame]`, so an
array shorter than the animation raises `IndexError` partway through rendering — after
`init_artists` has succeeded and, when saving, after some frames are already written.

The array reaches the panel from two directions (an explicit `var_array=`, or
`seq_result.var_array` forwarded from `analyse_diffusion_sequence`), and neither path knows
the frame count, so nothing upstream can catch it either.

*Fix:* validate in `_resolve_var`, where both paths meet and `n_frames` is already in
scope — refuse a length that is neither `n_frames` nor longer, naming both numbers. A long
array is a legitimate truncation (a windowed or `min()`-limited animation), so that case
stays silent; only "too short to finish" is an error. Cheap to pin: `diffusion` has no
tests at all today, so this wants the first one.

**A19. A measured nest axis could not be resolved at all, in either orientation.**
**[FIXED — 2026-08-17]** **[verified against real data]** Reported 2026-08-14 from the
uncommitted export `PL_Vbot_power_sweep_test_26_08_07_16_25_51_iter_0.csv` — 11 `V_A`
steps inside 6 power levels, 66 spectra, which
`fast_sweep="bottom_voltage", slow_sweep="power"` refused.

This is the case **A13 named and left open**: *"an axis whose scatter approaches its
level spacing will chain levels together"*. A13 made the two definitions of "the same
grid point" agree; it did not change the fact that one tolerance per axis has to fit
between the scatter *within* a level and the step *between* levels, and on a measured
axis no such value need exist. Two properties of a power sweep close the window: meter
scatter grows with the reading, so it is set by the top of the axis, and a power series
is log-spaced, so the smallest step to resolve is a tiny slice of the span.

Measured on that file: worst within-level scatter 3.06 µW against an `_axis_atol` of
0.42 µW — short by 7.3×. Synthetic clean sweeps fail as easily: 6 levels 100 µW apart
need only 0.6 µW of meter noise to shatter. Raising the constant was tested and
rejected — at `1e-2` a 101-step gate sweep collapses to **one** level, because the
tolerance must also stay under the step between levels (about span ÷ levels).

Worse in the swapped orientation, which the reporter also hit: with power as the *fast*
axis the miscount becomes `n_fast`, so `n_sweeps % n_fast` fails and the structural
checks never run at all. A clean 6-level log sweep gave `n_fast = 21` for 66 points.

*Fixed* by deciding the nest on **level separation instead of a tolerance**
(`_levels_separate`). Only the divisors of `n_sweeps` can be shapes, since the exporter
runs the fast axis fastest; for each, both axes are reshaped and each level's
lowest-to-highest range must clear its neighbour's. The comparison is always local — one
level's own scatter against the step to the level beside it — so unequal spacing and
value-proportional scatter both stop mattering, and the same test serves either
orientation (rows for a slow axis, columns for a fast one). Exactly one shape may
survive; several is refused by listing them rather than picked between.

Measured against an independent criterion (spectra whose reading is closer to some other
level than the one they were taken at), over 400 trials per condition: where nothing is
misattributed it now resolves 100% of the time and previously resolved 0%; where
readings genuinely interleave it still refuses. A single stray reading is tolerated up
to 0.8 of a step and refused beyond, i.e. once that reading has reached the neighbouring
setting. Trimming outliers before comparing was tested and rejected: it accepted 92% of
trials in which spectra really were mixed up, and accepted a reading three full steps
out of place. A false refusal is visible and recoverable; a false acceptance silently
files spectra under the wrong setting.

`_NEST_RTOL` is renamed `_AXIS_RTOL`, since it no longer governs nesting —
`_axis_atol`, `_level_labels` and `_count_distinct` remain, serving
`_warn_if_sweep_axis_repeats` and the accessor coordinate match.

`_nesting_failure` was rewritten with it. The old message reported
`n_fast × n_slow ≠ n_sweeps` using counts the algorithm never used to decide anything,
which on the reported file printed *"11 × 9 = 99, not 66"* — an arithmetic story about a
structural failure. It now anchors on whichever axis *does* separate, since that pins
the shape the file actually has, and quotes the overlapping ranges of the axis that does
not: *"At 11 × 6, fast_sweep='bottom_voltage' holds apart but slow_sweep='power' does
not: one covers 834.345–836.925 µW and the next 835.474–837.247 µW"*. Anchoring on an
arbitrary divisor instead was written first and caught in review — it blamed a healthy
gate axis at a 6 × 2 shape nobody had declared.

Covered by `test_a_scattered_level_nests_when_its_levels_stay_apart`,
`test_a_scattered_axis_nests_as_the_fast_axis_too`,
`test_overlapping_levels_are_refused_and_quoted` and
`test_a_non_finite_reading_refuses_the_nest`, each with a fixture pinned to reach its
case rather than pass for free. `test_aborted_raster_raises_naming_both_counts` is
renamed `…_naming_the_sweep_total`, and it plus
`test_swapped_declaration_raises_and_names_the_swap` had assertions on the removed
`(N distinct)` wording replaced. The assertion and grouping paths add 21 more, including
`test_transposed_counts_warn_about_the_clean_axis`,
`test_a_level_coordinate_is_the_median_not_the_first_reading` and
`test_a_grouped_nest_survives_a_round_trip`.

Two bugs were caught in review while writing this and are worth recording, since both
would have been quiet. `_nesting_failure` first anchored its diagnosis on the first
divisor where exactly one axis separated, which blamed a healthy gate axis at a 6 × 2
shape nobody had declared; it now anchors on whichever axis *does* separate, because that
is what pins the shape the file actually has. And the HDF5 writer's new loop used `for
kind in ("fast", "slow")`, shadowing the `kind` dict the axis dataset is named from a few
lines later — caught by nine round-trip tests failing with `TypeError: string indices must
be integers`.

*Also fixed, in the same pass* (decision `0017`): the file can now be loaded with the
axis labelled in µW while its **structure** comes from the commanded row —
`slow_sweep="power", slow_group_by="Fianium_Select_A4"` resolves 11 × 6 and gives a slow
axis of 416.9, 825.2, 835.6, 836.3, 836.1, 722.3 µW. Each level's coordinate is the
**median** of its readings rather than the first one, which was biased low by up to
1.3 µW on the two levels that drift while the laser settled (+0.31 and +0.23 µW per point,
r = 0.89 and 0.83). `SweepNesting.slow_spread` carries the peak-to-peak range beside each
coordinate — 0.4, 3.06, 2.58, 1.77, 1.29, 1.61 µW here — so a level that is not flat is
visible instead of averaged away. A shape that no row can establish can be asserted with
`n_fast=` / `n_slow=`. Both declarations round-trip through HDF5 (`FORMAT_VERSION` 2.2),
conditionally, so a resolved nest still re-resolves on read.

**Still not fixed, because it is not a code problem:** the measured power axis remains
*physically* unusable on this file — not monotonic, three levels within 1 µW. The load now
warns saying exactly that, and `0004` §5 still refuses an ambiguous lookup against it.
Why the laser plateaued, and why `Fianium_Select_A4 = 160000` reads 417 µW here while
`161000` reads 0.83 µW in `PL_Vbot_power_sweep_26_08_10_…` under identical logged
settings, is a lab question.

**Also found here, fixed separately as A20.** `_axis_atol` could not recognise an axis
that never moved. The sketch left above — *"borrow `varying_parameters`' RMS scale as a
floor"* — was **measured to be wrong** and was not taken; see A20 for what a floor does
to a fine sweep on a large offset.

**A20. A sweep axis that never moved was read as one setting per spectrum.**
**[FIXED — 2026-08-18]** **[verified against real data]** Found 2026-08-17 while closing
A19. `_axis_atol` answers *how far apart may two readings be and still be the same
instrument setting*, and answered it with `1e-3 ×` the axis's own span. That is circular
when the axis did not move: the span is then the instrument's own scatter, so the
tolerance lands a thousandth of the scatter it exists to absorb, and a **quieter**
instrument does not help — the tolerance shrinks in the same proportion. Measured across
four scatter levels on a held 500 µW setting: 59, 62, 58 and 64 settings found, never 1.

Not a cosmetic miscount. Two documented safeguards failed, neither loudly. On 12 spectra
taken at one power setting:

- `_warn_if_sweep_axis_repeats` **said nothing**. It compares settings against sweep
  points, got 12 of 12, and concluded each spectrum had its own position — while its own
  docstring names repeat measurements at one setting as part of what it catches.
- `nearest_index(500.0)` returned index 5 of 12 equally good points **with no warning**,
  and `get_spectrum_at` did not refuse, against decision `0004` §5.

*Fixed* by `_axis_driven`, which asks whether a row was driven on **two independent
signs, either sufficient**: its span exceeds `rtol ×` its RMS magnitude, or its readings
step through that span a small part at a time *in acquisition order*. Each is blind where
the other sees — the first misses a fine sweep on a large offset (300.0→300.2 K travels
0.07% of its readings), the second misses a coarse sweep whose few points are as far
apart as scatter would be. A row is called held only when both fail, so a false collapse
needs both wrong at once. `varying_parameters` now shares the same helper, so the report
and the grouping cannot contradict each other; its ranking by span/RMS is unchanged, and
its membership is unchanged on every committed file.

**The reach is narrower than it first appears, and is pinned by a test.** Because the
first sign fires as soon as scatter exceeds `rtol` of the reading, a held setting is
recognised only while the read-back is stable to better than that. Measured: 100% up to
1e-4 relative scatter, 97%/94%/70% at 2e-4 for n = 12/20/66, and **0% from 1e-3**. So a
source-meter holding a gate (20 µV on 5 V) is recognised; a power meter holding a power
(3 µW on 800 µW) is not, and still reads as many settings exactly as before. Closing
that would need a rule that survives both a log-spaced sweep and a flattened sawtooth —
see the rejected alternatives in `0018`.

Three call sites needed patching alongside, each an advice path that only became
*reachable*: the repeat warning said *"1 different values"* and offered nest advice for a
row that never moved; the accessor refusal did the same; and `_nearest`'s distance
warning reported a value as absent from the only axis holding it, because on a held row
the typical gap it compares against is scatter-sized. Its threshold is now floored at
`_axis_atol`, which on a driven axis is far the smaller of the two and changes nothing.

Verified not to disturb anything else, by measurement rather than reading: `_axis_atol`
has exactly two callers; every tracked spectral export and the TRPL directory was
reloaded and every parameter row reclassified, flipping exactly one — `V_A` in
`PL_Vbot_power_sweep_26_08_10_…csv`, held at −5 V with 18 µV of scatter, which *should*
be one setting and is not that file's sweep axis; and every module-level parameter dict
in `tests/` was reclassified with zero flips, cross-checked by grep (tests add scatter
only to counts, images and spectra, never to a parameter row). `sweep_grid()` counts with
exact `np.unique` and was untouched. 676 tests pass.

Covered by `tests/test_loaders_axis_tolerance.py`: ten axes parametrised against the
settings they hold, a companion test pinning that each held row *would* defeat a
span-only tolerance, the two restored safeguards, the absence of the spurious distance
warning, the fine-sweep-on-an-offset regression, and
`test_the_reach_of_the_detection_is_deliberate` fixing the boundary above. Fixtures draw
from per-row seeded generators after an order-dependent draw made one assertion flaky.

**A21. A sweep both coarse and narrow for its offset collapsed to a single setting.**
**[FIXED — 2026-08-18]** **[verified against real data]** Found 2026-08-18 reviewing the
A20 fix before merge, so it never shipped. `_axis_driven` called a row held only when both
its signs failed, and A20 presents those two blind spots as complementary. They are two
descriptions of one region: a sweep that is **both** narrow for its offset and coarse
defeats them together. Sign 2 fires when `median|Δ| < 0.1 × travel`, which for an even
sweep of *n* points needs `1/(n-1) < 0.1`, i.e. more than eleven readings — while the case
sign 2 exists to catch is exactly the one sign 1 cannot see.

Measured with the shipped helpers:

```
np.linspace(300.0, 300.2, n)  n = 3..10 -> held, _axis_atol 0.2 (the full travel), 1 level
                              n = 11    -> driven, but only by float noise in the diffs
np.linspace(5.000, 5.004, 5)            -> held, _axis_atol 0.004, 1 level
```

So a real five-point gate step of 1 mV at 5 V, or a six-point 300.00 to 300.20 K sweep,
warned *"This row was held at one setting for the whole file"* and had every coordinate
lookup on it refused — behaviour `main` did not have. A20's claim that "a false collapse
needs both signs wrong at once" is true and insufficient: the conjunction is reachable.
The slip is visible in `0018`'s own load-bearing note, which says `_AXIS_STEP_FRAC` has
least margin at small point counts "where sign 1 is carrying the decision anyway" — at
small point counts on a large offset, sign 1 is the sign that is blind.

*Fixed* by a third sign, decision `0019`: a row whose non-zero steps in acquisition order
all share a sign is being driven, however few readings it has. Exact repeats are dropped
first, so a slow axis's plateaus do not read as reversals. It abstains below
`_AXIS_MIN_MOVES = 4` moves, where direction is a coin toss — which leaves a four-point
narrow sweep still collapsing, the known residue.

The sign is **added**, not substituted, and that is the whole safety argument: OR-ing can
only move rows held → driven, so none of the counterexamples that killed A20's rejected
alternatives (the flattened sawtooth, the 201-step sweep at 5 V, the log-spaced power
series) can be lost to it. Verified rather than assumed — sign 1 alone still recognises
each of them, so sign 3 is never what rescues a nest.

Verified to A20's own standard: every loadable spectral export and the TRPL directory
reloaded with **zero** rows reclassified, and the false-driven rate on held 5 V rows
unchanged except at five readings (0.5% → 2.2%) and six (1.1% → 1.2%). 679 tests pass.

Covered by two new rows in `tests/test_loaders_axis_tolerance.py`'s `AXES` table at the
point counts that failed, and
`test_direction_is_only_evidence_once_a_row_has_moved_enough`, which pins the
`_AXIS_MIN_MOVES` boundary, both travel directions, a row that turns round, and the slow
axis's plateaus.

**A22. A held gate whose read-back drifts is reported as varying, and flips `gate_mode`.**
**[OPEN]** **[verified by running]** `_axis_driven`'s second and third signs read the
*shape* of a row's motion and carry no magnitude, so a monotone drift is seen at any
amplitude. `0018` §4 measured the rule's reach against i.i.d. scatter only, and concluded
a source-meter holding a gate is recognised; drift was never in scope. Measured: a gate
held at 5 V drifting monotonically by 1 mV over 21 points is driven at `rtol=1e-3`, `1e-2`
**and** `1.0`, where the pre-`0018` `span > rtol × RMS` test excluded it at all three.
Pure jitter of the same size is unaffected, which is why the existing test does not catch
it.

Not cosmetic, because `varying_parameters()` is not only a report. `gate_mode` and
`_gate_candidates` both key off membership in it, so a grounded second gate with a slow
thermal drift turns `"top-gate only"` into a dual-gate verdict — the Pearson branch then
correlates a real sweep against a drift and answers confidently.

*Not fixed here.* Giving `gate_mode` a private one-sign test re-splits the two definitions
`0018` §3 deliberately joined, and would let the report and the loader's grouping
contradict each other again. The honest fix is a declared instrument resolution — the
`sweep_atol=` argument `0018` left unbuilt for want of a file that needs it, and which
CLAUDE.md says not to add until a committed file does. This entry is the evidence for
when one appears. Until then the exposure is documented in `_axis_driven`'s docstring: a
held row is recognised only while its scatter is directionless.

**A23. A cosmic ray wider than half the median window survived its own repair.**
**[FIXED — 2026-08-18]** **[verified against real data]** Found 2026-08-18 from
`examples/example_1L_WSe2.ipynb`, where a spike at 748 nm stayed visible on the spectral
map with `cosmic_rays=` declared. The loader, the ladder and the plotting call were all
correct; `remove_cosmic_rays` was not.

Both the per-iteration fill in `_detect_cosmic_rays_1d` and the final replacement took
their local median with `median_filter` over the whole spectrum, so a flagged pixel sat
inside its own median window. Once a flagged run reached half the window the median was
drawn from the spike itself, and the value written was self-consistent: every later pass
returned it again. The boundary was a run of `(median_window - 1) / 2` pixels — 3 at the
default window of 7, which is exactly the "1–3 pixels wide" the docstring gave as the
definition of a cosmic ray, so nothing wider had ever worked. `test_processing_cosmic_rays`
planted spikes 1 and 3 pixels wide and nothing else, which is why it went unseen.

Measured on `PL_power_sweep_26_08_05_15_59_21_iter_0.csv`, sweep index 2 — a spike four
pixels wide on a 590-count baseline:

```
 px    raw   repaired   flagged        px    raw   repaired   flagged
643    606      606        0          643    606      590        1      <- after the fix
644    920      920        1          644    920      590        1
645   1980      920        1          645   1980      590        1
646   3412      920        1          646   3412      589        1
647   2503      920        1          647   2503      588        1
648    614      614        0          648    614      588        1
```

All four were detected and three were replaced with 920 — a value taken from the fourth.
The surviving 330-count plateau was still 149 714 in the plotted array after the Jacobian
and the `bg_region_eV` subtraction, against a whole-map maximum of 551 135.

*Fixed* by decision `0023`: a flagged pixel takes the median of the pixels in its window
that are **not** flagged. `_fill_flagged` is the single implementation and both final
replacement paths now call it, so the detection fill and the two replacements are one
piece of code instead of three. A window with nothing unflagged in it has no median; those
pixels keep their raw values and a `UserWarning` names them and points at `median_window`.

*Residue, stated in the function's `Notes`.* A **flat-topped** spike is bounded more
tightly than `median_window` implies, and silently: its interior is reached only by
replacing the edges and recomputing, one pixel in per pass, and the medians stop coming
from the baseline once the still-unflagged interior outnumbers the baseline in the window.
Measured on a flat top, repair is complete to `median_window // 2 + 3` pixels (6 at the
default, against 3 before) and beyond that the interior keeps its full height with no
warning, because every pixel that *was* flagged had a clean neighbour. Catching it needs a
local noise estimate — the same missing piece as the over-flagging finding below — and a
diagnostic built without one was measured firing 459 times on `PL_20uW_Vbot_sweep_*.csv`.

*Also found, not fixed.* `sigma_lap` is one number per spectrum, taken over the whole
dispersion axis, which is mostly baseline. At high excitation the shot noise on the X0 line
stands well above it, so peak pixels are flagged: 3704 flags on `PL_20uW_Vbot_sweep_*.csv`,
and on the power sweep 48 in the brightest sweep against 0 in the dimmest, clustered at
717–726 nm — on X0. The value changes are small (1683 → 1679 counts), so nothing visible is
destroyed, but the top of the power range is being lightly median-smoothed. A local noise
estimate is the fix.

*Test:* `tests/test_processing_cosmic_rays.py`, 8 new cases (18 → 26 in that file) — a
parametrised width sweep 1–6 at the default window, each planted pixel flagged and returned
to the baseline; a nine-pixel curved spike at `median_window=7` warning and keeping its
three centre pixels raw; and the same spike fully repaired at `median_window=15`, with the
warning asserted absent. 729 tests pass.


**A24. `build_irf_kernel` and `_build_lifetime_dictionary` disagree about where zero
delay is.** **[verified by running]**
`build_irf_kernel` places the instrument-response peak at index
`ceil(window_before/dt)` of the kernel it returns. `scipy.ndimage.convolve1d` always
treats index `len(weights)//2` as zero delay. The two coincide only when *window_before*
and *window_after* are equal, and the defaults are 0.3 ns and 2.0 ns.

Measured with a spike kernel at the notebook's `dt` = 4 ps: kernel length 576, IRF peak
at index 75, scipy's zero-delay index 288. Every dictionary column is therefore shifted
`(288 - 75) x 4 ps` = **0.852 ns early**, inside a `fit_scan_lifetime` window only 1.5 ns
wide (`t_range=(-0.2, 1.3)`). The model's rise happens before the fit window opens, so
every `tau_rise` and `tau_decay` from this path is fitted against a curve not lined up
with the data, and `examples/example-trpl.ipynb`'s committed outputs carry those numbers.

Two fixes were measured to give correct alignment: zero-pad the short side inside
`build_irf_kernel` so the peak lands on `len//2`, or pass
`origin = argmax(kernel) - len(kernel)//2` to `convolve1d` (always inside scipy's
permitted range of `-(L//2)` to `(L-1)//2`).

`_build_lifetime_dictionary`'s docstring asserts the opposite — that this "introduces no
spurious time shift, which matters because that shift would otherwise be degenerate with
(and bias) the fitted lifetimes themselves" — so it has to be corrected with the code. No
test reaches `build_irf_kernel`, `fit_sparse_lifetime` or `fit_scan_lifetime`, which is
why this survived.

**A25. `RamanMap` can plot uninitialised memory as data.**
**[FIXED — 2026-08-25]** **[verified by running]**
`RamanMap.__init__` allocates `self.counts` with `np.empty` and scatters rows into it by
`searchsorted` index. Its guard compares `n_x * n_y` against the row count, which a file
with a duplicated `(X, Y)` position can still satisfy — the duplicate overwrites one cell
and leaves another never written, holding whatever was in memory. That cell then reaches
`plot_image` and a mode fit as though it were a measurement. `np.full(..., np.nan)` plus a
check that every cell was written would fail loudly instead.

*Fixed as sketched*, with the check asked in terms of the cause rather than the symptom:
`np.bincount(iy * n_x + ix, minlength=n_y * n_x)` counts the rows landing in each grid
cell, and more than one in any cell is refused. "Every cell was written" and "no position
repeats" are the same condition once the row count matches, so one array answers both, and
the count is what lets the message name the offending coordinate.

**It refuses rather than warns**, which is the opposite of what **A12** did for a duplicate
`_iter_N` index, and the difference is what is left in the caller's hands. There, every
file was present and unmodified, so a warning let the researcher narrow the prefix and
nothing was decided on their behalf. Here, continuing means handing back an array with a
position in it that was never measured — there is no version of that the caller can act on
downstream, and the guard immediately above in the same function already refuses an
incomplete grid for the same reason. No winner is picked among the duplicate rows; that
would be the `_iter_N` decision CLAUDE.md forbids.

The message **names the repeated `(X, Y)` in µm** rather than counting the collisions,
following `_order_by_iter`'s wording under A12: the coordinate is what identifies the cause,
which is two acquisitions copied into one file, or a partly copied scan.

**`np.full(..., np.nan)` stays even though the check now raises.** The check covers the one
known route to an unwritten cell; the fill means any later change to the scatter indices
shows an unwritten cell as absent rather than as memory contents. One line, and it is the
half of the sketch that does not depend on having predicted the cause correctly.

*Test:* `tests/test_raman_map.py`, 3 new cases (10 → 13). The duplicate fixture repeats the
first position in place of the last, so 3 unique X × 2 unique Y still equals 6 rows and the
complete-grid guard passes — the defect exactly. It fails against the pre-fix code with
`DID NOT RAISE`.

The two "no cell unwritten" cases pass either way, and that is the finding worth keeping:
under `np.empty` the never-written cell held **finite** garbage, so `np.isfinite(...).all()`
was true against the broken code. A NaN-only assertion would therefore have tested nothing
— the same hazard as **A3**, where every case rendered because `set_path_effects` accepts
anything. They are the did-not-regress half, pinning that the fill has not started leaving
real holes; the refusal is what pins the bug.

Verified on the real map: `examples/data/Raman/map2.txt` still loads 10 × 8 × 1024 with both
hand-parsed spot values unchanged, so the new check does not refuse a good file.

**A26. `locate_residual_peak` can seed an amplitude below its own bound, and the crash is
not caught.**
When the discovery fit over-predicts across the whole `shoulder_range`,
`locate_residual_peak` returns a negative height. `fit_multi_voigt` uses it as the
shoulder's amplitude seed while `_bounds` sets that amplitude's lower limit to `0.0`, so
`curve_fit` raises `ValueError: Initial guess is outside of provided bounds`.
`fit_multi_voigt` catches only `RuntimeError`, so this propagates rather than warning and
skipping — and it happens in a pixel-by-pixel map loop, where one weak pixel ends the
whole run.

**A27. `NormalizedSpectrumPanel` smooths by default, and `animate_wl_pl_spectra_grid`
cannot turn it off.**
The panel defaults `smooth_window=11`, so Savitzky-Golay smoothing runs unless a caller
sets it to `None`. `animate_wl_pl_spectra_grid` constructs the panel itself and exposes no
`spectrum_style` door, so through that entry point every spectrum in the animation is
smoothed with no way to opt out. Same shape as the `plot_spectral_map` `median_kernel=3`
default, and against *corrections are opt-in*: the least-assuming default is no smoothing.
The missing door also departs from
`dev/decisions/0022-the-animation-wrapper-returns-its-panels.md`, which forwards one dict
to the spectrum panel.

**A28. `normalize=True` became a background subtraction.**
`plot_spectrum` and `plot_single_spectrum` changed from `y / y.max()` to
`processing.normalise_minmax(y)`. The two differ: `normalise_minmax` also subtracts each
spectrum's own floor. Whether that is wanted is a real choice — right when a dark pedestal
should not survive normalisation, wrong when the floor is signal — but it is a different
quantity under an unchanged argument name, so a figure made with `normalize=True` before
and after does not show the same numbers.

**A29. `RamanSpectrum` dies before it can refuse a map export.**
**[FIXED — 2026-08-25]** **[verified by running]**
`RamanSpectrum` carries a message telling a caller to use `RamanMap` for a map-shaped
file, and cannot reach it: `np.loadtxt` raises first. On the committed
`examples/data/Raman/map2.txt`:

```
ValueError: the number of columns changed from 1024 to 1026 at row 2
```

The cause is recorded in `dev/instruments/labram.md`: every row of a map export holds 1026
tab-separated fields, but row 0's first two are *empty*, and `loadtxt`'s default
whitespace delimiter collapses them away, so row 0 reads as 1024 columns against row 1's
1026. Classifying the file before parsing it, or reading with an explicit tab delimiter,
would let the written message do its job.

*Fixed by classifying before parsing.* `_labram_field_count` — a bounded peek beside
`RamanSpectrum`, in the shape of `_n_rows_upto` and `_classify_csv` — returns the
tab-separated field count of the file's first data line, and more than 2 is refused naming
`RamanMap`. Measured on the committed files: a single spectrum reads 2, `map2.txt` reads
1026.

**The explicit tab delimiter the sketch offered as an alternative does not work.** It
fixes the ragged read, and then `np.loadtxt(delimiter="\t")` fails converting row 0's two
*empty* leading fields to float — a different error that still does not name `RamanMap`,
and one that would change how every single-spectrum file is parsed on the way. The file
has to be identified before it is read, not read more carefully.

**Both refusals are kept, and both are reachable.** The pre-parse count catches a
tab-separated map; the post-parse shape check catches a map-shaped body written with
*spaces*, where tab-splitting the first line returns 1 and the new guard correctly stands
aside. That asymmetry is also why the threshold is `> 2` rather than `!= 2`: a
space-separated single spectrum counts 1 and must pass through untouched. The shared tail
of the two messages is one constant, `_RAMAN_MAP_ADVICE`, so the two routes cannot drift
apart in what they tell the caller to do.

*Test:* `tests/test_raman.py`, 2 new cases and one tightened (11 → 13).

`test_map_export_is_rejected_not_misread` **already existed and passed against the
defect** — it asserted a bare `pytest.raises(ValueError)`, which numpy's own *"the number
of columns changed from 1024 to 1026 at row 2"* satisfies just as well as the message the
class was trying to give. Now matched on `use RamanMap`, and it fails against the pre-fix
code. Worth keeping as the lesson: an unqualified `raises(ValueError)` on a file that has
two ways to fail tests only that it failed.

The synthetic ragged-first-row case matches on `fields on its first data line`, wording
only the pre-parse guard uses, so the two routes are told apart rather than both being
accepted as "it raised". The space-separated case is the did-not-regress half, and exists
so the post-parse check cannot quietly become unreachable.

Adjacent, found while writing them: `test_extra_columns_raise_a_clear_error` is
tab-separated, so it now takes the **pre-parse** route rather than the shape check it was
written for. Left as it is — it asserts the caller-visible outcome, which is unchanged —
with the new pair carrying the route-level claims.

**A30. `plot_spectral_map(rescale_img=True)` blanks the whole panel on one `NaN`.**
**[FIXED — 2026-08-21]** **[verified by running]**
`rescale_intensity(data, in_range="image", out_range=(0, 1))` takes its limits from the
array's own minimum and maximum, and a single `NaN` makes both of them `NaN`, so every
cell comes back `NaN` and nothing is drawn. `plot_image` already carries the fix —
`np.ma.masked_invalid` before the rescale, because a masked array's min and max skip the
masked entries and the mask survives the arithmetic — with a test at
`tests/test_plot_image.py:76`. `plot_spectral_map` has no such line: it goes straight
into `rescale_intensity`. Reachable, because `spectra_source="contrast"` is a ratio
against a reference frame, so a zero reference pixel gives a non-finite column. Found
while closing E3 and deliberately left out of that change; the fix is the same one line
`plot_image` already has, plus a test.

*Fixed as sketched*, with two things the sketch did not say. **The line's position is
load-bearing in a way it is not in `plot_image`:** `plot_spectral_map` runs
`processing.smooth_median` first, and that returns a plain array, so a mask applied before
the filter is discarded. It goes between the two. And the mask is applied
**unconditionally**, as `plot_image` does, rather than inside the `rescale_img` branch —
`pcolormesh` already leaves a bare non-finite cell undrawn, so the default path is
unchanged, and one spelling cannot drift from the other.

*Test:* `test_rescale_img_survives_a_guarded_contrast_pixel` in
`tests/test_plotting_spectral_map.py`, reproducing the reachable route rather than
planting a `NaN` — a reference with one zero-count pixel, which `spectral_contrast` guards
to `NaN`, giving a non-finite **column**. It reads the map on the **wavelength** axis
because the energy axis argsorts the pixels, so the guarded pixel is not column 2 there;
the first draft asserted the wrong column and failed for that reason rather than for the
defect. Assertions are on the values, not only the mask: every cell is masked in the
broken case, so a mask-only test would pass against it. Confirmed by reverting the one
line — all three rows of column 0 come back masked, and skimage emits its own
*"Rescaling will broadcast NaN to the full image"* warning, which nothing in the package
was listening for.

Documented on the `Returns` block, which now says non-finite cells are masked and take no
part in the colour limits or in `rescale_img`'s range — the caller-facing half of the
same fact `plot_image` states on its `img` parameter.

*Not addressed, and adjacent:* `smooth_median` still spreads a non-finite cell across its
footprint, so `median_kernel > 1` on a block holding a guarded column widens that column
by the kernel. Whether the median filter should skip non-finite pixels is its own
question, in `processing`, not in the plotting call.


## B. Dead parameters — accepted, documented, silently ignored

**B1. `AttoCubePLScanRealSpace(bg_region=, bg_stat=)` are stored and never read.**
**[FIXED — 2026-08-10, 4fe8f24]** **[verified against real data]** They were assigned
in `__init__` and read nowhere, so `load_frame()` returned `np.loadtxt(...)` untouched
and constructing with a background region gave frames identical to constructing
without one. No viewer could show a subtracted frame at all: `plotting.py` had no
`bg_region` anywhere outside `DiffusionCloudPanel`. Both parameters were also absent
from the class docstring, so the signature was their only documentation — and there
they read as supported.

**The sketch above ("apply `_apply_bg_region` in `load_frame`") would have introduced a
double subtraction, and was not taken.** `analyse_diffusion_sequence` calls
`scan.load_frame(i)` and forwards `bg_region` to `analyse_diffusion_cloud`, which
subtracts again; `diffusion.py:403-411` is an explicit comment relying on `load_frame`
being raw, and `DiffusionCloudPanel`'s `init_artists` and `update` would have been a
second instance. That is the mechanism of **A5**, and `diffusion.py` has no tests at
all to have caught it.

**Fixed as a sibling array instead.** `load_frame` is unchanged — the file's own counts
— and `load_frame_bg` carries the correction, mirroring `spectra` / `spectra_bg` and
the *raw arrays are never mutated after load* rule. `diffusion.py`,
`plot_diffusion_cloud` and `DiffusionCloudPanel` therefore needed no change, and the
double subtraction cannot arise.

Three decisions worth recording:

- **The viewer toggle is `frame_source={"best"|"raw"|"bg"}`, not `raw=True/False`.**
  Brandon proposed the boolean; it was argued down to a string because
  `spectra_source=` already answers this exact question with a named vocabulary, so a
  boolean would give the package two spellings for "which version of the data is
  this", read as a double negative when asking for the corrected frame, and have
  nowhere to grow if image cosmic-ray repair ever lands. `_resolve_frame` is built
  after `_resolve_spectra` and reuses its two error wordings.
- **One dict, no labels table.** Brandon asked whether `_FRAME_SOURCES` and a
  `_FRAME_SOURCE_LABELS` should be a single nested dict to avoid maintaining two —
  a fair instinct, and checking it found that `_SPECTRA_SOURCE_LABELS` is *dead*
  (imported at `plotting.py:32`, read nowhere). So no labels table was added at all;
  `_resolve_spectra` builds its message from `list(_SPECTRA_SOURCES)` and the frame
  resolver does the same. The dead spectral dict went with A17, which rewrote that
  vocabulary; its last import went when this branch was rebased onto that work.
- **`"best"` defaults to subtracted, and that is not a silent correction.** The opt-in
  is `bg_region=None` at the loader; once the researcher has asked there, a plot
  honouring it is the `best_energy_spectra` rule rather than a second decision.
  Defaulting the plot to raw would silently discard a correction they requested.
  `bg_stat` is now validated at construction, since `_apply_bg_region` falls through
  to the mean for anything but `"median"`.

*Test:* `tests/test_loaders_real_space.py`, 8 new cases (18 → 26) — `load_frame` still
raw with a `bg_region` set, the median/mean pedestals distinguished by a corner whose
two statistics differ by construction, `load_frame_bg` refusing without a region, an
invalid `bg_stat`, and the four `_resolve_frame` behaviours including an object
exposing only `load_frame` degrading to raw. Plus `tests/test_plotting_frame_source.py`
(7 cases), which reads the array back **off the axes** rather than checking the call
succeeded — forwarding a parameter and then ignoring it is precisely what B1 was.

Verified on the committed stark-shift sequence with a 40 × 40 corner: pedestal 209.0,
`load_frame` bit-identical to a scan loaded without `bg_region`, median and mean
estimators giving different frames (209.0 vs 208.9512), and
`analyse_diffusion_cloud(load_frame(0), bg_region=R)` equal to
`analyse_diffusion_cloud(load_frame_bg(0))` to the last digit — the single-subtraction
invariant, measured rather than assumed.

Not done here: **A5**, and `animate_wl_pl_spectra` constructs its
scans internally with no `bg_region` passthrough, so it is the one public path that
cannot reach this.

**B2.** **[FIXED — 2026-08-20]** `analyse_diffusion_cloud(scale_units=)` / `analyse_diffusion_sequence(scale_units=)`
are forwarded but never stored; `DiffusionResult` has no such field, though its
docstring says the unit is "used only in `__repr__`".
*Fix:* add the field and use it in `__repr__`, or drop the parameter.

*The field was added, and the deciding question was what the conversion is.*
`pixel_scale` is **µm per pixel on a CCD frame** — pixels to a length, not volts to a
length. The calibration in use is derived per session from the laser reference image;
`examples/example-trpl.ipynb` computes
`pixel_scale = laser_diameter_um / (2 * laser_ref.radius)`, a known spot diameter in µm
over the fitted 1/e² radius in pixels. Because the scale is the caller's to calibrate,
the unit is the caller's to name, so it cannot be hardcoded and something has to carry
it. (Volts to µm is a different conversion the package deliberately does not do — the
piezo rows are drive voltage and no µm/V calibration is in the file, settled under
**E7**. A third case needs no conversion at all: `RamanMap`'s X/Y are already µm, from
the file's own `#AxisUnit` fields.)

Dropping the parameter was the other option in the sketch and was rejected, because
**two readers were already waiting for the label**:

- `DiffusionResult.__repr__` printed `Centroid (real)` and `Area (real)` — bare numbers
  whose scale the reader cannot know, and an area that is in the unit *squared*.
- `plotting.plot_centroid_trajectory` labelled its y-axis `Centroid (real)` and its lines
  `x (real)` / `y (real)`: the literal word *real* standing where a unit belongs, because
  the unit could not be reached from the result.

The shape is not new here. `DiffusionSequenceResult` already carries `var_label` /
`var_units` for the external axis, and `loaders._CURATED` stores `(row, scale, unit)` as
one entry — pairing a scale with its unit is the established pattern.

Three details worth recording:

- **The field is appended, not placed beside `pixel_scale`.** A dataclass can be
  constructed positionally, so inserting a field silently changes what such a call means
  — the hazard CLAUDE.md names for `<Thing>Plot` returns. One site constructs
  `DiffusionResult` and it uses keywords, so this is cheap insurance rather than a fix.
- **`__repr__` now aligns its colons on the widest label** instead of a hardcoded width,
  because a caller-supplied unit has no fixed length.
- **The unit prints only beside a number that was converted.** With no `pixel_scale`
  neither real-space row appears, so a unit can never be shown against a pixel value.

`DiffusionSequenceResult.scale_units` reads off `frames[0]`, matching its sibling
accessors: one call analyses the whole sequence, so every frame carries the same unit.

Not touched: `plot_diffusion_cloud` and `DiffusionCloudPanel` forward `pixel_scale` but
not `scale_units`, and neither displays a real-space unit — the first draws the image in
pixels. Adding a parameter to the standing counter-example to *parameters earn their
place* (**E11**) would buy nothing.

*Test:* `tests/test_diffusion.py`, 6 new cases (5 → 11). Four fail against the pre-fix
code — the unit on both repr lines, the µm default, the unit reaching every frame of a
sequence, and the axis label read back off the axes — and the two that pass either way
are the negative ones, which pin that no unit is printed when nothing was converted.

**B3.** **[NOT A DEFECT — checked 2026-08-20]** `AttoCubeSampleImage.__init__`
(`AttoCubeSampleImage.__init__`) doesn't forward
`bg_region`/`bg_stat` to `_AttoCubeImage`, so sample images can't be
background-corrected even though the base class supports it.

*The mechanism is as described; the conclusion that it should be fixed is wrong.* The
override replaces the base's four-parameter `__init__` with a two-parameter one and calls
`super().__init__(path, laser_ref)`, so `bg_region` and `bg_stat` always take their
defaults. Python does not merge the two signatures — the subclass's is the whole public
signature — so `AttoCubeSampleImage(path, bg_region=R)` is a `TypeError`. The **attribute**
still exists, because the base assigns it; it is simply always `None`.

**A white-light sample image should not take a scalar background**, for two reasons that
are about the measurement rather than the class:

- **The corner of a reflection image is not dark.** `_apply_bg_region` subtracts one
  number, the median or mean of a patch. On a real-space PL frame that patch has no
  emitter in it, so the statistic estimates detector offset and stray counts. On a
  white-light frame the same patch is substrate, and substrate reflects — the statistic
  estimates the substrate's reflectance. Subtracting it yields counts measured from the
  substrate, which reads like a contrast without being one: a reflectance contrast is a
  dimensionless ratio.
- **A reflectance image wants a ratio, not a difference.** ΔR/R = (I_sample − I_ref)/I_ref,
  with a dark frame taken off both where one exists. A corner median is neither the dark
  frame nor the reference, so `bg_region` is the wrong instrument even for a caller who
  wants the sample image to be quantitative.

Two supporting observations. **Nothing computes a number from a sample image** — its only
consumers are `show_image` and `plotting.plot_image` — and both are invariant to an
additive constant under autoscaling (`rescale_intensity(in_range="image")` maps min–max
onto 0–1 either way). The knob would move `plot_image`'s colour-bar numbers and the
meaning of an explicit `clim=`, and change nothing else. And where a white-light
background genuinely has to go, the package already removes it with an estimator shaped
like it: `AttoCubeLaserReferenceImage._preprocess` runs a white top-hat, because that
background is flake contrast and reflectivity gradients rather than a constant.

So the rule is not *images never get a background subtracted*. It is: subtract when a
number is computed from the image, and use an estimator that matches the background's
shape. A PL frame passes both tests — `analyse_diffusion_cloud` thresholds the image and
takes areas and second moments, where a pedestal biases a fractional threshold and so the
cloud area — which is why `bg_region` exists at all (**B1**). A sample image fails both.

*What this entry points at that is real:* **there is no image-space reference or
flat-field path anywhere in the package.** The reflectance / differential-reflectance and
cavity work will need one, and it is a ratio against a second frame, not a region of the
same one. That is a feature to design when a committed file needs it, not a parameter to
forward.

*Recorded* as a one-line **don't** in `.claude/CLAUDE.md`, and pinned by
`tests/test_image_background_scope.py` (5 cases): a PL frame takes a region and applies
the named statistic, `AttoCubeSampleImage` and `SingleImage` both refuse one at the
signature, and both still carry `bg_region is None` so a viewer reading the attribute off
a duck-typed image finds no region rather than an `AttributeError`. The `__init__`
override is load-bearing because of that refusal — deleting it would inherit the base's
parameters — and now says so in a comment.

*Loose end, not fixed here:* `show_image(show_bg_region=True)` on an image with no region
draws no box and still sets `legend=True`. Harmless, and its own small change. — Filed as
**B5** and fixed there the same day. "Harmless" understated it: the `legend=True` was
inert, and the box's own label had never been reachable in any case.

**B4.** `fitting.voigt_approx` is implemented but not reachable from any `fit_*`
entry point; `constants.py` imports `hbar` unused.

**[HALF WRONG AS FILED; the other half FIXED — 2026-08-20]** `voigt_approx` **is**
reachable, and was made so by the PR #19 fitting work: `fit_voigt` passes it to
`_fit_single_peak`, and `multi_voigt` sums one call per peak, which is what
`fit_multi_voigt`, `fit_raman_modes` and `fit_pl_peaks` all fit with. `plotting`
reconstructs each component with it too. Nothing to do.

The `hbar` half stood, and is fixed here: `scipy.constants.hbar` was imported by
`constants.py` and read by nothing in the package. `h`, `c`, `e` and `epsilon_0` on the
same line are all used — `h`/`c`/`e` build `HC_EV_NM`, and `e`/`epsilon_0` are
re-exported as `E_CHARGE`/`EPS_0` — so only the one name went.

**B5. `show_image`'s background-region box could never be labelled, and an absent
region was a silent no-op.** **[FIXED — 2026-08-20]** **[verified by running]** Found
2026-08-20 while settling **B3**. Three faults in four lines of
`_AttoCubeImage.show_image`:

```python
if show_bg_region:
    processing._draw_region_box(ax, self.bg_region, bg_region_color, label="bg region")
    legend = True
if laser_annotation and self.laser_ref is not None:
    self._add_laser_circle(ax, self.laser_ref, legend=legend)
```

- **The box's label was dead in every case.** The only legend anywhere in this path was
  the one `_add_laser_circle` built as `ax.legend(handles=[circle], loc="upper right")` —
  an explicit handle list, so it omitted the box even when the box was drawn.
- **`legend = True` did not mean what it reads as.** It looks like *drawing the box
  implies a legend*. In fact it could only turn on the **circle's** legend, and only when
  a circle was being drawn anyway.
- **An absent region drew nothing and said nothing.** After **B3** that is the permanent
  state on `AttoCubeSampleImage` and `SingleImage`, which take no `bg_region` at all.

Measured on an `AttoCubePLImage`, counting artists on the axes afterwards:

| call | boxes | circles | legend |
|---|---|---|---|
| no region, `show_bg_region=True` | 0 | 0 | none |
| no region, `show_bg_region=True`, laser | 0 | 1 | circle only |
| **has region, `show_bg_region=True`** | **1** | 0 | **none** |
| has region, `show_bg_region=True`, laser | 1 | 1 | circle only |

Row 3 is the one that shows `legend = True` doing nothing at all.

*Fixed* by building **one legend in `show_image`, from the artists it actually drew.*
Neither drawer can own it — `ax.legend(handles=[...])` takes an explicit list, so a
legend raised inside either one omits the other's artist, which is exactly the bug. An
absent region now warns and names the class. `_add_laser_circle` **loses its `legend`
parameter**: its one caller no longer uses it, and dropping it moves that helper toward
`plotting._draw_laser_circle`, which draws and returns and builds no legend — the shape
**D2**'s remaining merge needs.

`show_bg_region` and `bg_region_color` were also absent from the `Parameters` block,
so the signature was their only documentation. Added, along with the `Warns` section.

**One behaviour change, chosen deliberately:** `legend=` now decides, and
`show_bg_region=True` no longer forces a legend on. The forcing was kept in view and
rejected — an argument that silently overrides another argument is harder to predict, and
the legend it forced did not contain the box it was forced on behalf of. The visible
difference is narrow: with a laser circle, `show_bg_region=True` and `legend=False`, a
circle-only legend used to appear and now does not.

`stacklevel=2` is **measured**, not one of **A11**'s fifteen. Every reachable path is the
researcher calling `show_image` directly: the one override,
`AttoCubeLaserReferenceImage.show_image`, never passes `show_bg_region`, so the warning
cannot arrive through it.

*Test:* `tests/test_show_image_annotations.py`, 6 cases, reading the artists back off the
axes because `show_image` returns only `(fig, ax)`. Four fail against the pre-fix code —
the box in the legend, box and circle labelled together, `legend=False` honoured, and the
warning with `caught[0].filename == __file__` — and the two that pass either way are the
did-not-regress ones.

*Adjacent, not fixed:* `AttoCubeLaserReferenceImage.show_image` sets `self.laser_ref =
self`, calls the base, then resets it to `None`. If the base call raises, the object is
left pointing at itself permanently.

**B6. A curated row's scale can be overridden and its unit cannot.**
**[FIXED — 2026-08-20]** **[verified by running]**
Found 2026-08-20 while fixing **B2**; the same fault one module over, and worse.
`loaders._CURATED` stores each entry as `(row label, scale, unit)`, and the override loop
in `_decode_and_describe` says so in its own comment: *"Only label (index 0) and scale
(index 1) can be overwritten."* There is no `curated_units`.

So a caller who follows `scanner_x`'s own docstring — *"Converting to µm needs a
per-stage µm/V calibration that is not in the file; supply one through
`curated_scales`"* — converts the numbers to µm and gets every axis label, `__repr__`
line and legend entry still reading **V**. B2 dropped a label that was offered; this
keeps a label that has become wrong, which is the more dangerous of the two, and it
happens on the one path the documentation actively recommends.

*This entry's diagnosis was half wrong, and the fix turned on the correction.* The
symptom above is real, but **`_CURATED[...][2]` was never what labelled anything.**
Traced by reading every reader of `self._curated`: the unit slot was read by exactly one
thing, the `curated_parameters` property — a documentation view. `_curated_value`
discards it. Every user-visible unit came from a *different copy*:

| copy | reaches |
|---|---|
| `_SWEEP_TYPES[key][2]` | `_resolve_sweep` → `sweep_unit` → `sweep_axis_label`, the `__repr__` sweep line, the held-axis warning, `_coordinate_text` legends, nest labels |
| a hardcoded `"µW"` in `__repr__` | the Power line |
| `sweep_unit="V"` panel defaults in `plotting` | `SpectrumLinePanel`, `NormalizedSpectrumPanel` — already owed to **E12** |

So the fault was not one missing override: **the unit for one quantity lived in three
places, only the scale was overridable, and any override desynchronised them.** Adding
`curated_units=` alone would have fixed `curated_parameters` and left every label reading
V — the entry's own headline symptom untouched.

*Fixed* as two coupled halves, recorded in
`dev/decisions/0029-a-curated-rows-unit-is-declared-with-its-scale.md`:

1. **`curated_units=`** on both live loaders, third store in the same override loop,
   round-tripped through HDF5 (`FORMAT_VERSION` 2.2 → 2.3). It reaches the gate and
   current entries, as `curated_scales` does and `curated_labels` does not — a unit
   claims nothing about wiring.
2. **The duplicate in `_SWEEP_TYPES` collapsed.** The five curated-backed rows now hold
   `None` for their unit, meaning *read it from the registry*; only `index`,
   `electric_field` and `carrier_density` still spell one, having no curated row behind
   them. Behaviour-preserving, because all five pairs already agreed
   (`V`/`V`, `V`/`V`, `µW`/`µW`, `V`/`V`, `V`/`V`) — pinned by a regression test rather
   than asserted.

**A changed scale with no unit warns**, naming the entry, both scales and the unit left
standing. Not raised: the numbers are right either way and the researcher may be
mid-calibration. Not silent: nothing downstream can detect it. Restating the unit
silences it, which is what decision 0005's polarity flip does — `-1e9` is still nA. Its
`stacklevel=3` is **measured** on both live doors, not counted off `def` lines; it is a
new site, not one of **A11**'s fifteen. `AttoCubePLVabScan` adds a frame and so points
at its own `super().__init__`; it is deprecated and gets a comment, not an argument.

Two tests were pinning the defect as correct behaviour:

- `test_constructor_overrides_flow_through_registry` asserted
  `curated_parameters["power"] == ("Galvo_X", 1.0, "µW")` — a scale dropped to 1.0, so
  raw counts, still claiming microwatts.
- `test_curated_overrides_restored` checked the label and the scale across a round trip
  and never the unit. That is why this went unnoticed.

Measured on `examples/data/position-scan/PL`, `curated_scales={"scanner_y": 12.5}`:

| | `curated_parameters` | `sweep_axis_label` | `__repr__` |
|---|---|---|---|
| scale only | `('Scanner Y', 12.5, 'V')` | `Piezo $y$ (V)` | `0 → 1125 V` |
| scale + unit | `('Scanner Y', 12.5, 'µm')` | `Piezo $y$ (µm)` | `0 → 1125 µm` |
| + HDF5 round trip | `('Scanner Y', 12.5, 'µm')` | `Piezo $y$ (µm)` | `0 → 1125 µm` |

The legend path with it: `_coordinate_text` goes from `'Piezo $y$ (V) = 125'` to
`'Piezo $y$ (µm) = 125'`.

*Test:* 11 new cases in `tests/test_loaders.py` and 2 in `tests/test_hdf5_roundtrip.py`
(895 → 908). 15 fail against the pre-fix code, including the two corrected above; the
negative ones — a restated scale is not a rescale, a stored scale does not warn on read
— are the did-not-regress cases.

*Adjacent, not fixed:* decision 0005 §79-82 recorded in passing that `curated_scales`
*replaces* rather than multiplies, so `{"i_bot": -1.0}` silently returns amps. That is
this same fault seen earlier and left; the warning now catches the unit half of it, not
the magnitude half. `curated_scales`'s docstring also claimed "Same keys as
*curated_labels*", which was never true — the role-backed keys are refused there and
accepted here; corrected while in the block.


## C. Documentation that contradicts the code

Worse than missing docs, because they will be acted on.

**C1.** **[FIXED — before 2026-08-13; confirmed while repairing this file's references]**
`__init__.py`'s quick start used `AttoCubePLScan`, `plotting.plot_pl_map`, and
`DeviceGeometry(t_hbn=..., b_hbn=..., tmdc=...)` — **none of which existed**.

*Fixed:* the quick start now reads `AttoCubeSpectralSweep`, `plot_spectral_map` and
`DeviceGeometry.from_single(...)`, and declares `gates=` as the loader requires. It was
carried along by the 2026-07-30 rewrite rather than by a change aimed at this entry, which
is why it was never marked. Note that this entry's own suggested replacements
(`AttoCubePLVabScan`, `plot_pl_map_Vab_scan`) had themselves been superseded by the time
the fix landed — the first is deprecated, the second is a shim.

**C2.** **[FIXED — 2026-08-20]** README §5 and §6 pass `bg_region=` to `fit_scan_peak` and
`extract_dipole_length`; neither accepts it (background is a load-time concern).
Both examples raise `TypeError` as written. Neither mentions the new `baseline`
argument.

*Fixed:* `bg_region=` is gone from both snippets and `baseline="constant"` is shown in
both, with a sentence in §5 saying where a background is actually declared
(`bg_region_nm=` / `bg_region_eV=` on the loader) and what `baseline` does that a
subtraction does not — it is a model term fitted alongside the peak, which is why it
defaults on. §6 points at §5 rather than repeating it.

Every keyword the two snippets now name was checked against
`inspect.signature`: `scan`, `x_axis`, `x_range`, `model`, `baseline` on
`fit_scan_peak`; `scan`, `x_range`, `model`, `Efield_range`, `baseline` on
`extract_dipole_length`. That is the check this entry is about — a keyword the function
does not accept is a `TypeError` on the first line the reader runs.

Not addressed here, and still open in **C4**: §2 loads with the deprecated
`AttoCubePLVabScan` and its example output says `AttoCubePLScan`, a class that does not
exist.

**C3.** `AttoCubePLVabScan` docstring says the Jacobian is applied "if `True`
(default)"; the signature default is `False`, and `False` is intended. README §2
inherits the same claim. Fix the docs, not the code.

**C4.** README's package-structure tree omits `diffusion.py` and the whole
`reference/` sub-package; its module-reference tables omit `SingleSpectrum`,
`SingleImage`, `AttoCubePLImage`, `plot_power_series`, `animate_panels`, and the
`AnimationPanel` family.

**C5.** `docs/api/constants.md` is a hand-maintained mirror of `constants.py` while
every other API page uses `::: mkdocstrings`. It will drift.
*Fix:* keep the prose intro, replace the tables with `::: tmdc_optics_tools.constants`.

**C6. Existing docstrings predate the docstring convention.** Added 2026-07-31, when
*a docstring is a contract, not a changelog* was written into CLAUDE.md. The rule is
recorded; the code does not yet follow it. Deferred by decision — Brandon is doing the
pass himself — and noted here so a convention with known violations does not sit
unapplied and unnoticed.

Mostly self-inflicted, by the 2026-07-30 rewrite. The known sites:

| Site | What offends |
|---|---|
| `AttoCubeTRPLSweep` class | A paragraph arguing why it is a separate class rather than a mode |
| `AttoCubeTRPLSweep.spectra_type` param | "Deliberately unlike `AttoCubeSpectralSweep`, which requires…" |
| `best_energy_spectra` | Defends the decision at length — this is the worked before/after in `dev/design-principles.md` §4 |
| `_cross_check_companion` | Why there is no value check, ending "which is how warnings get ignored" |
| `loaders.py` module docstring, `AttoCubePLVabScan` | "Deprecated **pre-rename** name", "reproduces the **pre-rename** behaviour" |
| `AttoCubeSpectralSweep` note **[FIXED — 2026-08-04]** | "no reflectance export has been characterised yet (see **E9** in `dev/defects.md`)" — stale as well as misplaced. Note deleted; the layout fact restated positively on the `.csv` bullet, which now says PL/R/RC share the block layout and that it is identified from the header rather than from `spectra_type`. |
| `DeviceGeometry.electric_field` | Two dates: "before 2026-07-30", "~0.6 % higher than pre-2026-07-30 results" |
| 6 sites in `loaders.py` | `:func:`_drop_unwritten_blocks``, `:attr:`_CURATED``, `:meth:`_cross_check_companion`` cited from *public* docstrings |
| `hdf5.py` module docstring | "What is deliberately *not* stored" — content is good, framing is a defence |

Two judgement calls to make during the pass rather than mechanically:

- **`electric_field`'s Notes is the strongest docstring in the package** — it stops
  someone "simplifying" an exact formula into a 0.6 %-low one. But it argues in terms
  of what the function did before 2026-07-30 and what the group's MATLAB computes.
  Trimming to "two forms that look tempting, and their magnitudes" keeps the warning
  and moves the history to A2, which already carries it. The one place where the rule
  costs something.
- **`hdf5.py`'s heading survives a rename**: "Not stored (all derivable from what
  is)" tells a user what they will not find in the file, which is documentation. Only
  the word *deliberately* is addressed to a maintainer.

No behaviour changes, so the suite is a straight regression check (154 tests as of
2026-08-04); `mkdocs build --strict` confirms cross-references still resolve once the
private ones are inlined.

**C7. Extracting the shared base silently emptied the rendered API pages.**
**[FIXED — 2026-07-31]** **[verified against the built site]** mkdocstrings-python
defaults `inherited_members` to **false**, so when the sweep classes' shared surface
moved onto the private `_AttoCubeSweep` base on 2026-07-30, every inherited member
stopped being rendered. `AttoCubeSpectralSweep` went to **6** documented members and
`AttoCubeTRPLSweep` to **3** — `sweep_axis`, `v_top`, `power`, `ef`, `to_hdf5`,
`get_parameter`, `varying_parameters`, `gate_mode`, `sweep_grid` and the rest were all
absent from the site. Only members redefined on the subclass survived.

*Fixed:* `inherited_members: true` in `mkdocs.yml`. 6 → **32** and 3 → **29**.

*Worth keeping as a lesson about the check, not just the setting:* `mkdocs build
--strict` stayed green throughout. It catches broken references and missing nav
entries — it cannot notice that a page lost two thirds of its content, because
nothing is *wrong*, there is just less of it. A refactor that moves members between
classes therefore needs the built output counted, not merely built. The count is
three lines of `re.findall` over `site/api/<page>/index.html` for
`id="<module>.<Class>.<member>"`.

**C8. `animate_wl_pl_spectra` documents a `suptitle_fmt` parameter that has never
existed.** **[FIXED — 2026-08-12]** The `**engine_kwargs` passage in `animate_wl_pl_spectra`
offered
```suptitle_fmt``, ``n_frames``, ``writer``` as examples of what is forwarded to
`animate_panels`. `animate_panels` has no `suptitle_fmt` — the shared title is assembled
from `frame_count_fmt` (`:1522`) and `suptitle_sep` (`:1523`). Passing the documented name
reaches `**engine_kwargs` and dies as an unexpected keyword argument, so the docstring
sends the reader to a `TypeError`.

*Fixed:* the passage now names `frame_count_fmt`, `suptitle_sep`, `frames` and `writer`,
all of which exist. Rewritten rather than patched, because `n_frames` was replaced by
`frames=` in the same change (decision `0014`).

**C9. A stale "circular import" comment in `plot_power_series`.** **[FIXED — 2026-08-13]**
`plot_power_series` opened with `from .constants import HC_EV_NM  # local import to
avoid circular at module level`. There is no circularity to avoid: `constants.py` imports only
`scipy.constants`, and `plotting` already did `from .constants import _x_axis_name_unit`
at module level (`:28`). The comment asserted a constraint that does not exist, which is
the kind of thing a later reader defers to.

*Fixed:* `HC_EV_NM` joined the module-level import, which made the local one provably
redundant, and both it and the comment went. Landed with the conjugate-axis helper rather
than as a drive-by — that helper needs `HC_EV_NM` at module scope, so leaving the local
import would have meant a function-level import shadowing a module-level name for no
reason.

**C10. `normalise_minmax`'s docstring is wrong about flat sweeps.** **[verified by running]**
The Returns block says "Sweeps with zero range (max == min) are left as-is". They are not.
The body computes `lo = min`, `span = max - lo`, replaces a zero span with `1.0`, and
returns `(spectra - lo) / span`, so a flat spectrum of constant 200 counts comes back as
all zeros. The `span[span == 0] = 1.0` guard prevents the division by zero; it does not
preserve the input. Either the sentence goes, or the function returns those columns
unchanged — and which is right depends on whether a flat sweep should plot at 0 or at its
own value, so it is a choice, not a typo.

**C11. `as_image_grid`'s docstring misdescribes `as_grid`.** **[NOT A DEFECT — checked
2026-08-20]** **[verified by running]**
It says `as_grid` "only accepts a 1-D or 2-D array", and justifies a
flatten-then-reshape-back step by it. `as_grid` reshapes any array whose trailing
dimension is `n_sweeps`, so the claim is false and the step it justifies is unnecessary.

*Wrong as filed — the docstring is accurate.* `as_grid` refuses anything outside
`ndim in (1, 2)` before it looks at the trailing axis, and has since `558c2e7` on
2026-08-06, which is nearly two weeks before this entry was written on 2026-08-19. A 3-D
stack with the right trailing axis raises. So the flatten-and-reshape-back step in
`as_image_grid` is doing real work, not standing on a false premise.

Kept rather than deleted, because the entry names a question the code does not answer:
**should `as_grid` accept an N-D array?** Its reshape is `array.shape[:-1] + nest.shape`,
which is already dimension-agnostic — only the guard is not — so widening it is one line
and would let `as_image_grid` hand its stack straight over. Two reasons not to do it
unasked. The guard's message enumerates the two accepted shapes, and a wider rule needs a
wider message. And an image stack is `(height, width, n_frames)` while a signal array is
`(n_points, n_sweeps)`, so "the trailing axis is the sweep" is a claim about the *caller's*
memory layout that only `as_image_grid` currently makes on the caller's behalf. Related:
**E24**, which wants `as_image_grid` to stop building the whole stack at all — a decision
that would change what, if anything, is handed to `as_grid` here.

*Pinned* by `test_as_grid_refuses_a_3d_array_even_with_the_right_trailing_axis` in
`tests/test_loaders_nesting.py`, so the docstring's premise cannot quietly stop being
true.


## D. Duplication

**D1.** `_draw_region_box` existed **verbatim** in both `processing.py` and
`diffusion.py` — 12 lines, identical signature and defaults, confirmed by diff on
2026-07-30 and again on 2026-08-10. Only the `processing` copy was ever reached:
every call site qualifies it as `processing._draw_region_box`, in both `loaders` and
`plotting`, and nothing named the `diffusion` one.

**[FIXED — 2026-08-10]** **[verified by running]** Deleted the `diffusion` copy, and
with it `import matplotlib.patches as patches` at the top of that module — the dead
function was its only user, so the import went stale in the same edit. 14 lines, no
new import needed: `diffusion.py` already does `from . import processing`.

Nothing to test, and that is the point of taking it first: the deleted name was
unreachable, so a regression check is the only check available. 420 tests green,
`mkdocs build --strict` green (the name is private, so it was never rendered).

*Found while verifying this:* `loaders` imports the bare name
(`from .processing import _draw_region_box, …`) but every call site spells it
`processing._draw_region_box`, so **that import is unused**. Left alone — a separate
change, and it wants the whole line checked rather than one name pulled out of it.

**[FIXED — 2026-08-20]** The whole line was checked, as this asked. `_draw_region_box`
is imported and never used under the bare name: the one call site in `loaders` is
`_AttoCubeImage.show_image`, which spells it `processing._draw_region_box`. The other two
names on the line are used unqualified and stay — `jacobian_correction_wvl2E` at two
sites, `subtract_background` at three. `loaders` also does `from . import processing`, so
the qualified call site needs nothing added.

**D2.** Two laser-circle drawers with different styling defaults:
`loaders._AttoCubeImage._add_laser_circle` (dashed, no halo, `loaders.py:5088`) and
`plotting._draw_laser_circle` (solid + halo), plus a third inline copy in
`ImageSequencePanel.init_artists`.
*Fix:* one helper in `plotting`; `loaders` calls it.

*Reference corrected 2026-08-12:* this entry used to cite the inline copy in
`animate_real_space_PL_map` — **A3** replaced that one with
`_draw_laser_circle`. The surviving inline copy is `ImageSequencePanel`'s, and it is the
one an `AnimationPanel` author would copy from.

*Half done — 2026-08-18.* `ImageSequencePanel.init_artists` now calls
`_draw_laser_circle` and keeps the artist as a public `laser_circle`, so the last inline
copy in `plotting` is gone and two drawers remain. Two defaults changed with it, both
already accepted as incidentals under **A3**: `zorder` 3 → 4, invisible because nothing
in the package sits between them and that axes holds only the image and the circle, and
the circle gains the helper's `label`, inert because no legend is built anywhere in the
animation-panel path.

**Still open:** `loaders._AttoCubeImage._add_laser_circle`. Closing it has to settle a
question this half did not raise — the two drawers spell the label differently
(`"Laser 1/e² (5.0 px)"` against `"$1/e^2$ Radius (5.0 px)"`), and the `loaders` one is
visible, because `show_image(legend=True)` puts it in a legend.

**D3.** `DeviceGeometry` precomputes `self.slabs` in `__init__` "for efficiency",
then `eps_stack` calls `self._slabs()` again while `d_stack` uses the cached one.
Still open after the A2 pass — deliberately left, to keep that change to one thing.
Two sources for one list: they agree today, but the cache goes stale if anyone
assigns `geom.d_hbn_top = 60` post-construction, at which point `eps_stack` would
update and `d_stack` would not.

Note this **blocks a reuse cleanup**: `eps_2d` now reads `self.d_2d` for its
numerator, but `eps_stack` cannot symmetrically read `self.d_stack`, because that
would mix the cached list (numerator) with a freshly built one (denominator) inside a
single expression — worse than the present duplication. Resolve the cache first, then
the reuse follows. Dropping the cache is the obvious call: `_slabs()` sums three or
four tuples, so "for efficiency" is not buying anything measurable.

**D4.** ~~`fit_multi_lorentzian` re-implements `_make_result` inline~~ — fixed
alongside the baseline work.

**D5.** `extract_dipole_length` accepts both `ef_range` and `Efield_range` for the
same thing, with a precedence rule. Pick one and deprecate the other. (Pylance
already flags `ef_range` as unused.)

**D6.** `reference/processors/__init__.py` does a dynamic glob-import over the
directory **and** then `from .X import *` for every module — each processor is
imported twice, and `import_module` is imported both at module level and inside the
loop.
*Fix:* keep only the explicit imports plus an `__all__`.

**D7.** **[FIXED — 2026-08-20]** `plotting` imports `from . import diffusion as _diffusion` at module top and
then re-imports the same module inside `plot_diffusion_cloud` and
`DiffusionCloudPanel._get_seq_result`. Similarly `plotting` has a block
commented "Lazy imports" that is in fact executed at module import.

*Fixed by deleting all three.* Both function-local `diffusion` imports go: there is no
circularity to defend against — `diffusion` imports `processing` and `loaders`, and
`loaders` imports no `plotting` — and the module-level import at the top of `plotting` is
itself the proof, since it would fail at import time otherwise. The two functions now
read `_diffusion` off the module namespace like every other call site.

The "Lazy imports" block turned out to be **a duplicate as well as a mislabel**. It
imports `Normalize`, `LogNorm`, `BoundaryNorm` and `ScalarMappable`, and the top of the
file imports the same four names — the mid-file block is the older of the two
(`ef1c3ed`, 2026-07-22) and the top-of-file imports arrived later (`35be4ab`,
2026-08-10), which is what made it redundant. Deleted rather than re-commented; the names
it claims to keep out of the module namespace have been in it since August either way.

Nothing was left function-local by this change. The one deliberate function-local import
in the package is `sklearn` inside `fit_sparse_lifetime`, whose reason is import cost and
whose comment says so — see **E22**.

Not touched: `plot_diffusion_cloud`'s remaining shape, which is **E11**.

**D8.** Plotting bypasses `processing`: `plot_spectrum` / `plot_single_spectrum` do
`y / y.max()` with no zero guard, while `processing.normalise_peak` exists and guards it.

**D9.** Two background conventions for the same idea —
`processing.subtract_background` (spectral, `mean`, x-range) vs
`processing._apply_bg_region` (image, `median|mean`, slice-pair). Fine to keep both,
but they should be named as a pair and documented together.

## E. Design, packaging, and project questions

**E1. `_CURATED` fail-fast makes some real files unloadable.** — **fixed 2026-07-30**
Raised `KeyError` if *any* curated row was missing — including `Scanner X`/`Scanner
Y`, which the code itself marks provisional. A scan file from a different instrument
config couldn't be loaded at all.

*Fixed as part of the `AttoCubeSpectralSweep` rewrite, as this item asked.* No
curated row is mandatory now: the file loads, and each curated property raises (with
the available labels listed) only if accessed. The one fail-fast that remains is for
the rows the **declared** `sweep=` needs, so the requirement follows what the caller
said they measured rather than a fixed list. `varying_parameters()` reports what
actually changed, which is the check a missing-row error was standing in for.

**E2. `save_figure(prompt=True)` calls `input()` by default** — blocks in notebooks,
scripts, and CI.
*Fix:* default `prompt=False`; require an explicit filename. (Also: `import os` sits
inside the function body for no reason.)

**E3. `plot_pl_map_Vab_scan` has three questionable defaults in ~20 lines.**
- `np.tile` builds two full `(n_pixels, n_sweeps)` coordinate meshes; `pcolormesh`
  accepts 1-D `x`/`y`. **[FIXED — 2026-08-21, `fcd8b41`]** **[verified against real
  data]** The 1-D call needs the block transposed, because `pcolormesh` reads `C` as
  `(rows=y, cols=x)` only when the coordinates are 1-D, so `mesh.get_array()` and
  `mesh._coordinates` now run `(n_sweeps, n_pixels)`. Checked on
  `examples/data/1L-WSe2-PL` at 1340 × 161: identical quads, colour values, colour
  limits and axis limits. Reasoning and the rejected `np.broadcast_to` alternative:
  `dev/decisions/0031-the-spectral-maps-mesh-runs-one-row-per-sweep-point.md`.
- **`median_kernel=3` runs a 2-D median filter**, i.e. it smooths *across gate
  voltage*, mixing physically independent sweeps. **Decided: default should be `1`
  (off), with 2-D kept available.** **[FIXED — 2026-08-21]**
  **[verified by running]** `plot_spectral_map` defaults `median_kernel=1`; a value
  above 1 still applies the square footprint, and the docstring now says that the
  footprint spans the sweep axis as well as the pixel axis. Two tests in
  `tests/test_plotting_spectral_map.py` pin both halves —
  `test_no_median_filter_runs_unless_a_kernel_is_named` names no kernel, so it is the
  default it sees, and `test_a_kernel_above_one_still_filters` names 3, so deleting the
  filter would be caught. `plot_pl_map_Vab_scan` forwards `*args, **kwargs` and needed
  no change.

  Both of those tests compare the drawn array against `scan.spectra`, so both took a
  `.T` when the bullet above transposed the mesh. That is why the three bullets landed
  as one branch rather than in any order: a shape mismatch inside `np.allclose` raises
  instead of reporting a difference, so whichever change went second would have broken
  the other's tests on arrival.
- `rescale_img=True` rescales the whole map to [0,1], silently changing what the
  colour bar means; the `colorbar_label` parameter is accepted but then overwritten
  (line 197, with the old line left commented out). **[FIXED — in two parts]** The
  `colorbar_label` overwrite went with the E12 label work (2026-08-06): the call now
  reads `colorbar_label if colorbar_label is not None else _signal_label(scan,
  normalized=rescale_img, ...)`, so a caller's string is verbatim and a derived one
  reads `norm.`. The colour bar therefore no longer misstates what was drawn.
  What remained was that `clim` is read in the data's own units while `rescale_img`
  replaces them, so the two together drew one flat colour — measured spread across
  the colour scale, `clim=(500, 1500)` on counts of 503–1497: 0.994 raw against
  0.001 rescaled. The pair is now refused, in `plot_spectral_map` and `plot_image`
  alike. **[FIXED — 2026-08-21, `129258f`]** Reasoning and the rejected
  warn-and-draw and reinterpret-the-limits alternatives:
  `dev/decisions/0030-clim-and-rescale-img-are-refused-together.md`.

**E4. Test coverage.** Only `tests/test_loaders.py` and `tests/test_laser_spot.py`
exist — nothing covers `fitting`, `processing`, `diffusion`, or `plotting`, which is
where A1 and A4 live. `pytest` is not in `pyproject` (no `dev`/`test` extra) and is
not installed in the `viz-sci-plot` env. Tests are local-only by choice; no CI test
job wanted.

*Update.* `pytest` is installed in `viz-sci-plot` and the whole suite runs. It is still
**not declared** in `pyproject.toml`, in any extra, so the dependency exists only in that
one environment and `pip install -e ".[docs]"` does not provision it. Declaring
`[project.optional-dependencies] test = ["pytest"]` fixes the *signal* only: it does not
commit the project to a CI test job, and it leaves tests local-only by choice.

That signal is the whole point. This item drifted twice in two days while the audit was
open — present, then absent, then present again after a manual install — so an undeclared
test dependency has no way to announce itself when an env is rebuilt, and any docs
describing it go stale silently, as they did in both directions.

*Update, 2026-08-20 (`a65b6fa`).* `pytest` is declared:
`[project.optional-dependencies] test = ["pytest"]`. The "no CI test job wanted"
sentence above is **reversed** — `.github/workflows/tests.yml` runs the suite on every
pull request and every push to `main`, on Linux, Windows and macOS. Reasoning and
rejected alternatives: `dev/decisions/0027-the-suite-runs-in-ci-on-three-systems.md`.
E4's coverage half stands unchanged: `diffusion` still has no tests.


**E5. Packaging.** `pyproject` lists `ffmpeg` as a dependency: that PyPI package is
an unmaintained wrapper, not the ffmpeg *binary* matplotlib needs. Use
`imageio-ffmpeg` or document it as a system dependency. `requests` and `h5py` are
only used by `reference/` and would fit better as an extra.

*Also:* the installed editable metadata advertises a console script that does not
exist on `main`. `src/tmdc_optics_tools.egg-info/entry_points.txt` declares
`tmdc-convert = tmdc_optics_tools.converters:main`, but `converters.py` lives only on
`dev/hdf5` (commit `ecfb87d`) and `pyproject.toml` has no `[project.scripts]` table.
It is leftover metadata from a `pip install -e .` made while that branch was checked
out; `egg-info/` is gitignored, so this is local-only and clears on reinstall.
Harmless until someone runs `tmdc-convert` and gets an `ImportError` — but it makes
the *installed* package look like it has a CLI that the source tree does not have.
Whether the converters belong on `main` is a merge decision, not a packaging one.

**E6. Small sharp edges.**
- `AttoCubeLaserReferenceImage.show_image` temporarily sets `self.laser_ref = self`
  and resets it afterwards — leaks the mutation if the plot call raises.
- Default threshold is `"1/e"` in `plot_diffusion_cloud` but `"otsu"` in
  `DiffusionCloudPanel`, so the same image analysed statically and in an animation
  gives different contours.
- `analyse_diffusion_cloud` rebinds its own `threshold` parameter to the computed
  value (line 316).
- `diffusion` uses an absolute `from tmdc_optics_tools.loaders import ...`
  where the rest of the package uses relative imports.
- `DiffusionSequenceResult.x_real` indexes `vals[0]` — `IndexError` on an empty
  sequence.

**E7. Constants provenance — unresolved.** Not defects, but undocumented:
`power_scale = 0.303e6`; `EPS_TMDC["HS"] = 7.5`; `T_MONOLAYER` = 0.65 nm for all four
materials. Each is recorded with what is missing in `dev/physics-conventions.md` §8, and
the citations they would need are listed in §9 of that file.

*`Scanner X` / `Scanner Y` units struck 2026-08-04 — **answered: volts.** The scanners
are piezos and the rows carry drive voltage, so scale 1.0 was right and the unit string
was not; the `position_x` / `position_y` sweep keys became `piezo_x` / `piezo_y`.
Recorded in `dev/instruments/attocube.md`.*

*(The displacement-field formula was checked at audit time and found self-consistent,
then **superseded 2026-07-30**: it had been the thin-TMDC approximation, 0.59% low, and
is now exact. See **A2** and `dev/physics-conventions.md` §2 before touching it.)*

**E7a. `EPS_HBN = 3.9` may not match its own citation.** Added 2026-07-30, still open.
The four TMDC values match the cited paper's bulk out-of-plane figures, but that paper's
hBN out-of-plane value is usually quoted as **3.76**, and 3.9 is also the canonical SiO₂
value. Now that `electric_field` is exact, `eps_hbn` no longer enters it directly — it
still sets `eps_stack`, so it moves the field through the ~98% of the stack that is hBN.
The most tractable of the three provenance gaps, because it is the only one a reachable
source could settle. Worth one look at the paper.


**E7b. Gate polarity is not recorded per scan.** **[FIXED — 2026-08-05]**
**[verified by running, against `examples/data/stark-shift/`]** Added 2026-07-30.
`_CURATED` hardwired `v_top → "V_A"` / `v_bot → "V_B"` and applied it silently, so a
voltage applied to a device's **bottom** gate was read as the top gate. Three
consequences, in increasing severity: `sweep="bottom_voltage"` passed validation and
returned the *top* channel, with a plot that looked entirely normal; `ef` was mirrored,
so any dipole extracted from it had the wrong sign; and `gate_mode` returned
`"top-gate only"` — the repr did not merely assume wrong, it **asserted** wrong.

A partial fix on 2026-07-30 recorded the mapping and surfaced it in `__repr__` and in
exported HDF5, which made a transposition visible on sight but still defaulted when
nothing was said. Closed 2026-08-05 by requiring a keyword-only `gates=` and refusing to
produce `v_top`, `v_bot`, `ef` or the gate sweep types without it.

**Nothing in the code is still open.** The wiring itself has to come from the lab
notebook per session — that was always true, and is now enforced rather than assumed.

Decision, rejected alternatives and consequences:
`dev/decisions/0002-gate-wiring-must-be-declared.md`.


**E7c. `gates` could not describe a single-gated device.** **[FIXED — 2026-08-05]**
**[verified by running, against `examples/data/stark-shift/`]** Raised immediately by
E7b's fix, on a real device: one electrode drives the bottom gate, the other contacts the
TMDC to ground it. With no top gate, no `{"top", "bottom"}` assignment was correct -
including the transposition the caller had been worried about. Passing the grounded row
as `v_top` would have returned a field with the wrong denominator by ≈2× for a 53/46 nm
stack, in a slab that is now the terminating electrode.

Fixed by making the role vocabulary describe device **topology**, adding a `"channel"`
role for a contact to the TMDC itself, and deriving what is computable from which roles
are present. A geometric carrier-density path came with it.

Decision, rejected alternatives and consequences:
`dev/decisions/0003-gates-declares-device-topology.md`.


**E8. Repo weight.** ~160 example CSVs live in the working tree. They are there
deliberately, to reproduce the example notebooks. The root `data/` directory is
empty and unused.

**E9. The instrument export format was nowhere described.** — **largely closed
2026-07-30**, by sample files rather than by code, exactly as this item asked.
`examples/data/reflectance-contrast/` (an R sweep + its substrate reference) and
`examples/data/TRPL/` (three decays + a metadata companion) were committed, and reading
them settled five of the six unknowns.

The format record is `dev/instruments/attocube.md`. Two things the sample files exposed
that inference had missed are filed as their own findings: the zero-filled over-allocated
blocks (**A6**), and that column-count arithmetic cannot recover the block count at all -
the layout must be read from the header names.

**Still open, and no file can answer it:** which acquisition software and version emits
this format, and whether the layout is version-stable.

The synthetic-fixture caveat stands and is worth keeping in mind: `make_spectral_csv` is
written from the same understanding as the parser, so it pins the decoding *contract* but
cannot catch a shared misunderstanding. The real files are the check on that, which is
why a handful of tests load them directly.


**E10. Working context that is not written down anywhere.** Not defects; questions
whose answers change what an agent (or a new group member) does, and which currently
have to be guessed per task. Grouped here so they can be answered in one pass.

- **Measurement data on the lab share.** Real scans live under
  `//lanesnas.epfl.ch/lanes/Brandon/01_Projects/...` (visible in
  `.claude/settings.local.json`). The guardrails in CLAUDE.md cover in-repo paths
  only. May the share be read? Written to, ever? Is it the canonical store, or a
  copy?
- **Notebook policy.** `examples/*.ipynb` are load-bearing — the *Verification*
  section wants them smoke-run — but nothing says whether a signature change updates
  them in the same commit or whether Brandon re-runs them himself. "Changes are made
  one at a time" pulls the other way.
- **Branch policy.** Feature work goes on a dedicated branch rather than `main`. That
  is current practice but appears in no committed file; for a package aimed at ~15
  people it belongs in CLAUDE.md. Separately, the branch list needs a prune: of eight
  remote branches, `dev/animation_plots`, `dev/complete_loader`, `dev/laser_spot_fix`
  and `dev/hdf5` were last touched 2026-06-26, and `dev/hdf5` alone carries an
  unmerged module (`converters.py`, plus `tests/test_converters.py` — see E5). Which
  are alive?
- **Current research focus.** CLAUDE.md says what the group measures, not what is
  being measured *now*. One line — sample, physics question, target figure — would
  settle a lot of small judgment calls (does this fit need error bars, should this
  plot default to the linear Stark regime) that are otherwise guessed. Arguably
  belongs in task prompts rather than a committed file, but the absence is currently
  filled by inference either way.

**E11. `plot_diffusion_cloud` is over-parameterised because its return contract is
broken.** `plot_diffusion_cloud` takes ~30 keyword arguments, of which ~15 are pure
matplotlib styling: `contour_color`/`contour_lw`/`contour_ls`,
`centroid_color`/`centroid_marker`/`centroid_ms`, `roi_color`, `bg_region_color`,
`laser_color`/`laser_linewidth`/`laser_linestyle`/`laser_halo_color`,
`xlabel`/`ylabel`, `colorbar_label`.

The cause is the return, not the signature. The function returns `fig, ax, result`
— a `DiffusionResult`, not an artist — so it breaks the
`(fig, ax, <artist>)` convention and hands back no handle for the image, contour,
centroid marker, or laser circle. Callers therefore have **no** route to restyle
after the fact, and the style parameters are the workaround. Every one of them is a
permanent public commitment for something the caller could otherwise do in one line.
A related symptom sits in the body: `ax.legend(fontsize=5, …)`
hardcodes what `set_style`'s `legend.fontsize` already governs, overriding the user's
own style with no opt-out.

`laser_halo` is the one style argument to keep — the white halo is what makes a thin
red circle legible over a dark colormap, so it is correctness-of-reading rather than
decoration. See *parameters earn their place* in `dev/design-principles.md` §2.

*Fix:* return the artists alongside the result, then delete the style parameters.
The return shape is the open decision — either a named tuple carrying `result` plus
`im`, `contour`, `centroid`, `laser` (keeps a 3-tuple, and `None` for artists that
were not drawn reads naturally), or a 4-tuple `(fig, ax, result, artists)`. Prefer
the former: it stays shape-compatible with the rest of `plotting` and lets fields be
added later without breaking unpacking. Also drop the hardcoded legend `fontsize`.

This is a breaking change to a function used by the example notebooks, so
`examples/` needs a pass in the same commit.

Two existing findings are instances of the same principle and are worth fixing
together with this one: **D2** (two laser-circle drawers with different styling
defaults) and **E3** (`plot_pl_map_Vab_scan`'s questionable defaults). **E6**'s
`threshold` mismatch between `plot_diffusion_cloud` and `DiffusionCloudPanel` is
adjacent but separate — that one is a physics default, not styling.

**E12. `plotting` still speaks gate-sweep PL.** Added 2026-07-30, created by the loader
rewrite. Nothing is broken — `gate_axis` / `gate_axis_label` survive as aliases and every
call site still runs — but the module lags the loader in two ways, one now closed.

- **Hardcoded "PL intensity" in ~6 places.** **[FIXED — 2026-08-06]** A reflectance sweep
  came out labelled "PL intensity (counts)". Replaced by one module-wide label contract:
  `dev/decisions/0011-label-contract-derive-or-verbatim.md`. Two defects were fixed in
  passing, neither previously recorded — `plot_image` discarded `colorbar_label` whenever
  `rescale_img=True`, always overwriting it with a hardcoded string; and a second,
  **dead** `signal_label` property sat ~300 lines above the real one with a bare `return`
  as its body, so any reordering of the class body would have turned every signal label
  into `None`.
- **Names and arguments still carry the gate-sweep assumption** — *open*.
  `plot_current(ef_axis=)`, `plot_spectrum`'s hand-rolled `E_F` legend default, and
  `SpectrumLinePanel(sweep_attr="scanner_y", sweep_unit="V")` — a position default
  carrying a voltage unit, now redundant since the panel can read the scan's own sweep
  axis and unit.

**Do the `plot_current` rename first.** **E15** already changed that function's signature
and its return, and the remaining step changes it again; landing them separately breaks
every caller twice for one function's worth of churn. The planned tests for it also need
`gates=` in their fixtures now, since a current row cannot be attributed to an electrode
without a declared mapping.

TRPL plotting and lifetime fitting are **unwritten** rather than broken: the x-axis
resolver knows only energy and wavelength, and `fitting` has no exponential model. The
baseline machinery is already generic over the model function, so a decay fit is a small
generalisation there rather than a parallel implementation.

Full remaining plan: `dev/plan-E12.md`.


**E13. `remove_cosmic_rays` was reachable but not wired to a scan.**
**[FIXED — 2026-08-05]** Added 2026-08-05. The function existed with tests and nothing
called it, so every caller spelled out `processing.remove_cosmic_rays(scan.spectra)` and
carried the cleaned array by hand. Two faults followed: the repair arrived *after*
`__init__` had already built the background-corrected, energy-axis and contrast arrays
from the unrepaired counts, so it agreed with none of them; and nothing recorded that a
repair had happened, so `__repr__` was silent and `to_hdf5` could not write it down.

Fixed by a load-time `cosmic_rays=` dict at the head of the wavelength-space chain,
adding `spectra_cr` and `cosmic_ray_mask` and reassigning nothing. Amended by **A17**,
which mirrored the correction ladder across both spectral axes.

Decision, rejected alternatives and consequences:
`dev/decisions/0001-cosmic-ray-repair-at-load-time.md`.

*Test:* `tests/test_loaders_cosmic_rays.py`, 14 cases. The pedestal-bias test is the
ordering one: it plants a spike inside the background window and pins the ≈129-count
over-subtraction it causes when the repair is not run first. `make_spectral_csv` grew
`roi1=` / `wavelength=` overrides so a test can write spectra with real scatter — the MAD
noise estimate has nothing to work against on the default index ramp.


**E14. Nested sweeps were not expressible.** **[FIXED — 2026-08-06]** Added 2026-08-06.
Closes **A8** and implements the grid API deferred since 2026-07-31, with a different
spelling from the one settled then.

A 2-D raster arrives as one flattened file and nothing in the rows states the nest, so
the sweep axis was a silent sawtooth and any map built from it was wrong. The harder
motivating case cannot be expressed at row level at all: both gates moved together to
sweep the displacement field, so every gate row takes a different value at every point
while the field they encode takes exactly `n_fast`.

Fixed by `fast_sweep=` / `slow_sweep=`, both resolving through the existing sweep
vocabulary so a *derived* quantity can be an axis, with `as_grid()` as a view over the
still-flat arrays and value/position accessors on top. The verification mechanism was
later amended by **A13** — compare level indices, not raw readings.

Decision, rejected alternatives and consequences:
`dev/decisions/0004-nested-sweeps-fast-and-slow.md`.

*Test:* `tests/test_loaders_nesting.py`, 52 cases. The raster fixtures are **synthetic**:
the committed reflectance export turns out to be the first 50 points of a 41 × 51 scan -
one complete X row plus 9 — so the real file exercises the aborted-raster refusal and
nothing else. It has a test of its own for exactly that.

---


**E15. Electrode currents were channel-named in a registry of role names.**
**[FIXED — 2026-08-07]** `_CURATED` held `"Ich1": ("I_A", …)` and
`"Ich2": ("I_B", …)` — three faults in one entry: the only camel-case keys in a
registry of snake_case attribute names; a `1`/`2` indexing that the file (`I_A` / `I_B`)
does not use; and a *channel* named in the one place everything else names a *role*.

It mattered because `V_A` and `I_A` are one source-meter terminal, so `I_A` belongs to
whichever electrode channel A reached — which only `gates=` records. Reaching the bottom
gate's leakage therefore meant knowing both that `1↔A` and that `A↔bottom`, and only
the second was written down anywhere. The two currents also got one blanket description
("leakage currents") when on a contacted device they are two different quantities -
leakage across a dielectric on a gate, transport into the flake on the channel contact.
`examples/old_example_stark_shift.ipynb` shows the cost directly: `ich2_label="I_A"`,
`ich1_label="I_B"` — label overrides used to repair a mapping the index names got wrong.

Fixed by `i_top` / `i_bot` / `i_channel`, resolved from `gates=`. `Ich1` / `Ich2` deleted,
along with the `ich1_label` / `ich2_label` shims that could no longer reach anything.

Decision, rejected alternatives and consequences:
`dev/decisions/0005-electrode-currents-are-role-named.md`.

*Found in passing, not fixed:* `curated_scales` **replaces** a scale rather than
multiplying it, so flipping a current's polarity is `{"i_bot": -1e9}` and
`{"i_bot": -1.0}` silently returns amps. Harmless for the voltages, whose scale is `1.0`.
Caught by a test written the wrong way round first.

*Test:* `tests/test_loaders.py` — currents transpose with the wiring, the channel
contact's own row, the grounded-electrode raise, the non-source-meter-row raise, and the
scale override.

---

**E16. `plot_power_series` draws a twin axis and does not return it.**
**[FIXED — 2026-08-18]** (The function is `plot_spectral_series` since PR #36; the entry
keeps the name it was filed under.) With
`twin_axis=True` the function creates `ax_twin = ax.twiny()`, labels it, and returns
`fig, ax, cb, lines`. The axes object is unreachable, so restyling its ticks or label —
the thing the return contract exists to make possible — is impossible without walking
`fig.axes`.

This is the same failure as **E11**, at a smaller scale: CLAUDE.md's rule is that a
function drawing several artists returns all of them, and the return *is* the styling API.
`plot_current` already gets this right, returning `ax_right` for its power trace.

*Fix:* append `ax_twin` to the return, `None` when `twin_axis=False` — mirroring how `cb`
is already handled. It is a breaking change to a 4-tuple, so it should land with something
else that touches this function; the natural partner is migrating it onto a shared
energy↔wavelength helper (see **E17**) rather than on its own.

*`plot_spectrum` — fixed, 2026-08-18.* It had acquired the same unreturned `ax_twin`; it
now returns `fig, ax, line, ax_twin`, `None` when `twin_axis=False`. That return was a
3-tuple and the parameter was unreleased, so the widening cost nothing. The entry above
stays scoped to `plot_power_series`, whose 4-tuple is the one with callers.

*Fixed 2026-08-18.* `plot_spectral_series` returns `fig, ax, cb, lines, ax_twin`, `None`
when `twin_axis=False`, landing with **E17**'s migration as the single break both entries
asked for. Six call sites: five in `tests/test_plotting_labels.py` and one notebook cell.
The rename that arrived in between did not make this cheaper — it had already been merged,
so the callers paid for the two edits separately after all.

**E17. Two implementations of an energy↔wavelength second axis, and the newer mechanism is
the better one.** **[FIXED — 2026-08-18]**
`plot_power_series(twin_axis=)` builds it with `ax.twiny()` and manually
relabelled ticks: tick *positions* stay in the primary unit and only their text is
rewritten, so the nm labels fall wherever the eV ticks happened to land (2.048, 1.937 …)
and the axis freezes if anyone later changes `set_xlim`.

`ax.secondary_xaxis("top", functions=(f, f))` does the same job properly — matplotlib
chooses ticks in the *target* unit, so the labels come out round, and the axis tracks later
limit changes. `HC_EV_NM / x` is self-inverse, so the function pair is one function twice.

*Half done — 2026-08-13.* `_conjugate_x_axis(ax, x_axis)` now exists in `plotting`, covering
both directions, and `SpectrumLinePanel(twin_axis=True)` uses it. Measured on an energy
axis, it places ticks at 500, 550 … 800 nm; the `twiny` version relabels the eV ticks, so
the same axis would read 495.9, 550.4, …

*A third site nearly appeared — 2026-08-18.* `plot_spectrum` gained a `twin_axis`
carrying a byte-for-byte copy of the `plot_power_series` block, comments included, while an
unrelated change was being written. That copy was removed from the change before it merged,
so it is not in the history, and `plot_spectrum`'s `twin_axis` is on `_conjugate_x_axis`
from its first commit.  Measured the same way: 500, 550 … 800 nm on a 500-800 nm sweep, and a later
`ax.set_xlim(1.60, 1.80)` re-ticks the top axis to 690 … 770 nm instead of freezing.
Pinned by `tests/test_x_axis_vocabulary.py`. The lesson is that a duplicated block invites
a third copy faster than it invites a fix — see
[0021](decisions/0021-the-conjugate-axis-has-one-implementation.md).

*The last site — 2026-08-18.* `plot_spectral_series` (this function, renamed by PR #36)
is on `_conjugate_x_axis`, with **E16**'s return change in the same commit. One
implementation of the conjugate axis now, in `plotting`, reached from three call sites.
Measured on a 500-800 nm sweep plotted in energy: 500, 550 … 800 nm, and
`ax.set_xlim(1.60, 1.80)` afterwards re-ticks to 690 … 770 nm. That zoom-after-plotting is
what the committed example notebook for this function does on the line after the call, so
it had been rendering a top axis that described the unzoomed view. Pinned by
`tests/test_x_axis_vocabulary.py`.

**E18. The animation engine has no tests at all.** **[FIXED — 2026-08-12]** `animate_panels`,
`animate_wl_pl_spectra`, the shared-title composition, `frame_label`, and the frame-count
minimum are entirely unpinned. Nothing anywhere calls `animate_panels`; the only panel
method any test touches is `SpectrumLinePanel.init_artists`, at three call sites
(`tests/test_plotting_labels.py`, `tests/test_cosmic_rays_downstream.py`).
`ImageSequencePanel` is reached only indirectly, through
`tests/test_plotting_laser_circle.py`'s end-to-end render, and `DiffusionCloudPanel` not at
all.

*Correction 2026-08-18:* the indirect reach claimed for `ImageSequencePanel` did not
exist. That end-to-end render goes through `animate_real_space_PL_map`, which builds its
own figure and never constructs a panel. The two tests that do construct one
(`tests/test_plotting_frame_source.py`) use a scan with no `laser_ref`, so its laser
branch had no coverage of any kind until **D2**'s first half added some.

That makes it the least-defended surface in `plotting`, and it is the one an outside
contributor is most likely to extend — the frame-window work in PR #16 landed two
crash-on-render bugs there that a single render test would have caught.

*Fixed:* `tests/test_plotting_animation.py`, 13 cases plus one `xfail`, following the
`tests/test_plotting_laser_circle.py` idiom — `matplotlib.use("Agg")`, duck-typed panel
stand-ins, an autouse figure-closing fixture, and a real `PillowWriter` save for anything
the constructor cannot catch. Pins the frame-count minimum, an explicit `n_frames=`, lock-step
advance, the suptitle composition (counter, panel labels, dropped `None`s, the
nothing-to-say case), that the title tracks the frame *through a render* rather than
freezing, and that a label resolved inside `init_artists` still reaches the title — the
ordering the load-bearing comment in `animate_panels` describes but nothing enforced.

*Two things measuring it settled.* Building an animation and priming the writer both draw
frame 0, so `update` sees `[0, 0, 0, 0, 1, 2, 3]` for a four-frame animation; the tests
compare the sequence with consecutive repeats collapsed, so they pin our frame order rather
than matplotlib's priming. And the layout assertion turned up **E20**.

**E20. The shared title is placed from a hardcoded axes fraction and collides with anything
the panels draw on top.** **[FIXED — 2026-08-12]** **[verified by running]**
`animate_panels` puts its shared title at
`transAxes` y=1.04, or y=1.12 with each panel's axes title nudged `pad=-4`. Both numbers are guesses about how tall the panels' decorations
are, and the title is an axes artist so nothing lays it out.

A panel that adds a secondary top axis is enough to break it. Matplotlib's
`_update_title_position` lifts the axes title above the top spine's tick labels and axis
label — roughly 30 pt — while `pad=-4` pulls it back only 4 pt, and the shared title's y
knows nothing about either. Measured at `figsize=(10, 4)`: with an axes title and no top
axis the shared title clears it by **20.6 px**; add a secondary top axis and the axes title's
top rises to 395.8 px while the shared title starts at 375.7 px — **20.1 px of overlap**, the
two strings drawn through each other.

No shipped panel draws a top axis yet, which is why this has never been seen. It stops being
hypothetical the moment one does.

*The sketched fix above was wrong, and measuring it is what showed why.* It proposed
drawing once and placing the title from the measured top of the panels' decorations. There
is nowhere to place it: `ax.get_tightbbox().y1` is **395.8 in every configuration** tried,
because `constrained_layout` does not grow the figure to fit decorations — it shrinks the
axes. The panels already reach the top of the usable area, so no measurement finds free
space above them. The mechanism is better stated the other way round: the axes box top
falls from 375.5 to 335.2 when a secondary axis appears, so a position given as a *fraction
of the axes* slides down while the axes title stays pinned at the layout's top.

*Fixed:* use a real `fig.suptitle`, which is the only shared title `constrained_layout`
reserves vertical space for, and set `blit=False`. Clears the panel titles by **8.3 px**,
identically across every combination tried — 1, 2, 3 and 4 panels × five figure sizes ×
secondary axis present or absent. The fake header, both magic fractions, and the engine's
`pad=-4` restyling of the caller's panel titles are all deleted.

*Why `blit=False` is not a regression.* Blitting repaints only axes artists, which is why
the fake header existed. Checked across all three output paths, for both header styles and
both blit settings:

| header | blit | notebook slider | GIF | MP4 |
|---|---|---|---|---|
| fake (axes text) | either | updates | updates | updates |
| real `suptitle` | `True` | **frozen** | updates | **frozen** |
| real `suptitle` | `False` | updates | updates | updates |

So a GIF-only check would have passed a title that is frozen in the notebook player *and*
in MP4 — the two paths this group actually uses. Nothing is given up: both save paths draw
full frames regardless of the flag (measured slightly **faster** without blit, 49.1 s vs
54.0 s for 40 frames), and `to_jshtml`'s slider steps through frames rendered in advance,
which blitting cannot speed up. Only live playback in a desktop window or `%matplotlib
widget` now redraws more per frame.

*One visible side effect.* Matplotlib warns from `Animation.__del__` when an animation is
collected having never drawn. Setting up blitting used to force an init draw, which marked
the animation as started as a side effect, so the warning was suppressed by accident.
Building an animation and never rendering it now warns — which is arguably the correct
signal, since nothing happened. The tests filter that one message because many of them
assert on the built figure deliberately.

*Consequence for the panel protocol:* `AnimationPanel.update`'s returned artists no longer
drive anything. The return is kept — all three panels already do it, it documents which
artists a panel owns, and it lets a caller drive the panels itself — but the base class
docstring no longer claims it exists for blitting.

*Test:* `tests/test_plotting_animation.py` — the clearance assertion is parametrised over
4 panel counts × 4 figure sizes, since the bug was scale-dependent;
`test_the_engine_does_not_blit` pins the flag, and
`test_the_header_updates_in_the_notebook_player` pins the *consequence* through
`HTMLWriter`, so flipping blit back on fails on behaviour rather than on a flag.

**E19. `DiffusionCloudPanel` analyses every frame even when the animation shows fewer.**
`_get_seq_result` (`DiffusionCloudPanel._get_seq_result`) calls `analyse_diffusion_sequence(self.scan, …)` with
no frame limit, so the segmentation, smoothing and contour extraction run over the whole
sequence. `animate_panels`' frame count only limits what is *drawn*: `init_artists`
receives `n_frames` and `_resolve_var` truncates the label array to it, but the analysis has
already happened by then.

So `animate_panels(panels, frames=range(5))` on a 500-frame scan does 100× the analysis it
needs — and cloud analysis is the expensive part, not the rendering. The frame-window work
(decision `0014`) makes this sharper rather than better: a selection exists precisely so a
long scan becomes tractable, and for this panel it saves rendering while saving nothing on
analysis.

*Fix:* pass the frames actually being shown into `analyse_diffusion_sequence`. It needs a
frame-selection argument of its own to accept them, so this is a `diffusion` change with a
`plotting` caller, not a `plotting` fix — which is why it is filed rather than folded into
the window work.


**E21. Three new plotting functions return no artists.**
`plot_spectra_overlay` returns `(fig, (ax_raw, ax_norm))`; `plot_multi_voigt_overlay` and
`plot_fit_param_comparison` return `(fig, axes)`. None hands back the lines, the point
markers or the `ErrorbarContainer`s it created, so restyling means reaching into
`ax.lines` and re-deriving which artist is which — the pressure that grows style
parameters back. `dev/decisions/0024-long-plotting-returns-are-named.md` sets the shape:
`(fig, ax, artist)`, or a `NamedTuple` named `<Thing>Plot` once there is more to return.

Not breaking today — nothing outside `examples/example_position_xy_scan.ipynb` calls them.
That is the reason to do it soon rather than later: a return-shape change is silent for a
caller unpacking positionally, so the cost rises with every caller added. Fixing
`plot_spectra_overlay` also means editing that notebook, which unpacks
`fig, (ax_raw, ax_norm) = plotting.plot_spectra_overlay(...)`.

**E22. Importing the package imports scikit-learn.** **[FIXED — 2026-08-20]**
`plotting` imports `fitting` at module level, and `fitting` imports
`sklearn.linear_model.Lasso` at module level, so `import tmdc_optics_tools` pulls in
scikit-learn for anyone who only wants to plot. It is declared in `pyproject.toml`, so
this is not a missing dependency — but it arrived with `fit_sparse_lifetime`, and every
existing environment failed to import the package until scikit-learn was installed. A
function-local import inside `fit_sparse_lifetime` would confine the cost to the one
function that needs it.

*Fixed as sketched* — the import moved into `fit_sparse_lifetime`, the only user. This is
the one function-local import in the package that is *not* the stale-comment kind
described in **D7**: there is no circularity here, the reason is cost, and the comment at
the import says so.

Removed in the same edit: `fitting` imported `from . import processing` **twice**, three
lines apart. Only the line carrying `constants` alongside it survives.

*Test:* `tests/test_import_cost.py`, 2 cases. The first runs `import tmdc_optics_tools`
in a **fresh interpreter** and asserts no `sklearn` module is present afterwards — it has
to be a subprocess, because by the time the suite reaches this file the session has
imported scikit-learn for something else and `sys.modules` here would say nothing about
the package. Confirmed by reverting: it fails against the pre-fix code.

The second is the other half, and is the reason this is not simply a deletion: a lazy
import that is never exercised is a `NameError` waiting to happen, and nothing else in the
suite calls `fit_sparse_lifetime`. It is a smoke test only — that the call completes and
returns a `SparseLifetimeResult` — and makes no claim about the lifetimes, which are
fitted against a model misaligned with the data until **A24** is fixed.

**E23. `fit_scan_lifetime`'s two defaults contradict each other.**
`t_range=(-0.2, 1.3)` gives a 1.5 ns fit window while `tau_range=(1e-3, 5.0)` puts
candidate lifetimes up to 5 ns in the dictionary. The function's own docstring warns
against exactly this: keep the largest candidate within a few times the window width,
because a candidate much longer than the window is nearly flat across it and therefore
degenerate with an offset, and one tiny coefficient at a very long lifetime can then
dominate the amplitude-weighted average. One of the two defaults should move; which one
depends on the decay being measured, so it is a physics call.

**E24. `as_image_grid` loads every frame at once.**
It stacks the whole sequence: for `examples/data/position-xy-scan` that is 58 frames of
512x512, roughly 120 MB, duplicated again by `np.stack` and again by
`processing.reorder_grid`. `ImageSequencePanel` pulls one frame at a time through
`load_frame` precisely so an animation does not hold the sequence in memory, and routing a
grid animation through `as_image_grid` defeats that.

**E25. Every `:func:`/`:class:` cross-reference renders literally on the docs site.**
The docstrings use reStructuredText roles — ``:func:`write_sweep` ``,
``:class:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep` `` — **504 of them** across
eight modules:

| module | roles | | role | count |
|---|---|---|---|---|
| `loaders.py` | 210 | | `:attr:` | 168 |
| `plotting.py` | 129 | | `:func:` | 150 |
| `fitting.py` | 68 | | `:class:` | 97 |
| `converters.py` | 19 | | `:meth:` | 63 |
| `diffusion.py` | 15 | | `:data:` | 26 |
| `hdf5.py` | 9 | | `:mod:` | 1 |
| `processing.py` | 7 | | | |
| `constants.py` | 2 | | | |

None of them render. mkdocstrings parses the *sections* of a NumPy docstring with griffe,
then renders each section's **text as Markdown**. An RST role is not Markdown, so nothing
interprets it and it passes through verbatim. What a reader gets on the site, from
`hdf5.read_sweep`:

```html
Read an HDF5 file written by :func:<code>write_sweep</code>.
```

— the `:func:` shown as prose, the name in code style, and **no link**. Every one of the
504 is a cross-reference that silently is not one, which is the whole point of putting
them there.

*This is not a configuration problem.* Cross-references work on these pages when something
else generates them: `signature_crossrefs: true` produces four real
`<a class="autorefs">` links on the `hdf5` page alone, pointing at `write_sweep`,
`read_sweep`, `AttoCubeSpectralSweep` and `AttoCubeTRPLSweep`. The linking machinery is
present and working; the docstrings are simply written in a syntax it does not read.

*Fix, measured rather than assumed.* mkdocstrings' own cross-reference syntax is
`[text][fully.qualified.path]`. Changing one line and rebuilding:

```
before   Read an HDF5 file written by :func:`write_sweep`.
after    Read an HDF5 file written by [`write_sweep`][tmdc_optics_tools.hdf5.write_sweep]
```

```html
<a class="autorefs autorefs-internal" href="#tmdc_optics_tools.hdf5.write_sweep">
  <code>write_sweep</code></a>
```

A working link. The experiment was reverted; nothing in the tree carries it.

*Why this is not a quick sweep.* Three reasons to do it deliberately rather than with one
regular expression:

- **The target has to be fully qualified.** `:func:`write_sweep`` resolves by context in
  Sphinx; `[...][...]` does not. Every one of the 504 needs its module path worked out.
  A wrong path is *caught*, though — see the note below — so this is tedious rather than
  dangerous.
- **`~` means something.** `:class:`~tmdc_optics_tools.loaders.AttoCubeSpectralSweep``
  displays only the last component. The Markdown form needs that split by hand into link
  text and target.
- **Some are private.** A role pointing at `_classify_csv` or `_order_by_iter` has no
  rendered page to link to, and `dev/design-principles.md` says never to cite a private
  helper from a public docstring — so those are not translations but deletions, decided
  one at a time.

*A mistyped target cannot reach `main`.* Measured, after an earlier draft of this entry
claimed the opposite: `mkdocs_autorefs` logs `Could not find cross-reference target
'...'` naming the **source file and line**, and `mkdocs build --strict` turns that
warning into a failed build.

```
WARNING - mkdocs_autorefs: api\hdf5.md: from src\tmdc_optics_tools\hdf5.py:442:
          (tmdc_optics_tools.hdf5.read_sweep) Could not find cross-reference target
          'tmdc_optics_tools.hdf5.no_such_function'
Aborted with 1 warnings in strict mode!
```

`docs.yml`'s `build` job runs that command on every pull request, so the check is
automatic. This removes the main hazard of the conversion: an unresolvable target fails
loudly. What the build cannot catch is a target that resolves to the **wrong existing
object**, which is why the pass still wants reading rather than a regular expression.

*Not urgent.* The docstrings read correctly in the source, which is where most of this
package is read today, and the text around each role still says what it says. The cost is
navigability on the site and a `:func:` prefix that looks like a typo to anyone who does
not know RST. **Its own pass, per module** — the same standing rule as the `stacklevel`
audit, though for a weaker reason than that one: here the build is a real safety net, and
what remains is the `~` handling and the judgement calls about private helpers.

*Worth deciding first: MkDocs or Sphinx.* These 504 roles are Sphinx syntax, so the entry
assumes the answer is "convert them". Sphinx would make them work as written. That trade
belongs in a decision record before the conversion starts, not after it.


## Settled design decisions

Every decision that closed a finding in this file now has its own append-only record in
`dev/decisions/`, carrying the alternatives that were rejected and why. The 2026-07-30
rewrite is `0006`-`0009`; the gate work is `0002`, `0003` and `0005`; nesting is `0004`;
the cosmic-ray wiring is `0001`. Index and conventions: `dev/decisions/README.md`.

None of those records carry a finding number of their own: they are work that *closed*
findings (**E1**, **A6**, most of **E9**, part of **E7b**) rather than findings
themselves. The standing one-line **don't** for each lives in `.claude/CLAUDE.md` under
*Do not re-propose*.


## Suggested order

1. **A18** — the live bug left in section A once the Raman and TRPL entries are set
   aside. It is cheaper than this list has assumed: the frame-window work removed
   `_resolve_var`'s truncation, so the frames being drawn are known where they are
   needed. (This item used to read **A5**, "the last live bug in section A", to be
   taken with **E11**; A5 was fixed on 2026-08-07 and `tests/test_diffusion.py` exists
   now, so E11 no longer has a bug riding on it.)
   (A1 fixed 2026-07-28; A2, A3 and A6 fixed 2026-07-30; A5 fixed 2026-08-07; A8 fixed
   2026-08-06; A10 and A7 fixed 2026-08-07; A9 and B1 fixed 2026-08-10; A4 deferred.)
   **A11** (warning locations) and **A12** (duplicate iteration indices) were both
   opened by the A7 work; A12 is fixed, A11 is still owed and is its own deliberate
   pass over 15 sites.
2. **A22** — a held gate that drifts reads as varying, which `gate_mode` then answers
   confidently on. Blocked on a file that needs a declared instrument resolution
   (`sweep_atol=`); until one appears the entry is the evidence, not the work.
3. **C1–C3** — the doc corrections, since they actively mislead. **C8** and **C9** join
   them, and both are cheapest taken by whatever next edits their function.
4. **D2–D6, D8, D9** — dead parameters and duplication; mechanical, low risk. (B1 fixed
   2026-08-10, and was not mechanical: the sketched fix would have introduced a second
   instance of A5. **B3** was checked on 2026-08-20 and is not a defect. **B4** was half
   wrong and half fixed the same day, as were **D1** and **D7** — none of the four was as
   mechanical as this line assumed, because three of them were entries the code had
   already overtaken. **B5** was opened and fixed the same day too, found while settling
   B3: the same class of fault as B1, a parameter accepted and then not acted on.
   **B2** was fixed on 2026-08-20 by adding the field, and opened **B6** — the same
   fault in `loaders._CURATED`, where a scale can be overridden and its unit cannot.
   **B6** was fixed the same day and was *not* mechanical either: its filed diagnosis
   was half wrong, and the unit that actually reached the labels lived in
   `_SWEEP_TYPES`, so the fix was a deduplication as well as a new argument. That makes
   five entries on this line the code or the analysis had already overtaken.)
5. **E2, E3, E5** — the remaining design calls, one at a time.
6. **E11 (with D2)** — the `plot_diffusion_cloud` signature and return shape. Last
   because it is the only breaking change on the list and touches `examples/`.

**Opened 2026-08-12 by the review of PR #16** (`animate_panels` frame windowing):
**A18**, **C8**, **C9**, **E16**, **E17**, **E18**, **E19**, and **E20** which pinning E18
turned up. **E18 and E20 are both fixed** — E18 gated the rest, because the engine could not
be safely changed while it was untested, and E20 had to precede any panel that draws a top
axis. The frame-window work is now unblocked. **A18** and **E19** are both
`DiffusionCloudPanel` and both wanted `diffusion`'s first tests, which **A5** brought on
2026-08-07 — so they no longer wait on **E11**, and go together. **E16** and **E17**
landed as one change, as planned. **C8** and
**C9** ride along with whatever touches their function.

**Opened 2026-08-19 by the review of PR #19** (Raman loaders, PL peak fitting, TRPL
lifetime fitting, merged onto `main` on 2026-08-19): **A24–A29**, **C10**, **C11**,
**E21–E24**. Every one is in code that has never been released, so none carries a
deprecation cost — which is the argument for taking them now rather than later. Order:

1. **A24** first, and ahead of everything else on this list. It is the only entry here
   that makes numbers wrong rather than raising, `examples/example-trpl.ipynb`'s committed
   outputs already carry them, and it needs the first test to reach `fit_sparse_lifetime`
   at all. The docstring's contrary claim is corrected in the same change.
2. **A25** — same class as A24 (wrong data, no error raised) and much cheaper:
   `np.full(..., np.nan)` plus a check that every grid cell was written. **Fixed
   2026-08-25**, and the check went in as a per-cell row count rather than a
   written-flag grid, so the message can name the repeated coordinate.
3. **A26** and **A29** — two crashes, each with an obvious fix, neither needing a
   judgment call. A26 matters most in a map loop, where one weak pixel currently ends the
   run. **A29 fixed 2026-08-25**, and the sketch's second option (an explicit tab
   delimiter) turned out not to fix it; **A26** is still owed.
4. **A27** with **E21** — a default that forces smoothing on, and three returns that hand
   back no artists. Different faults, but the same new public surface, and both are
   cheapest to change before anything depends on the current shapes. E21 also touches
   `examples/example_position_xy_scan.ipynb`.
5. **A28** — needs a decision on what `normalize=` should mean before it can be a change.
6. **C10**, **E23**, **E24** — ride along with whatever next touches their
   function. **C10** and **A28** are the same question seen from two sides and should be
   settled together. (**C11** was checked on 2026-08-20 and is not a defect; the entry
   now carries the open question it hides instead.)

**E22 was worked around before it was fixed.** scikit-learn was installed into
`viz-sci-plot` on 2026-08-19 so the suite could run; the module-level import went on
2026-08-20. *That install went missing again, and the cause is not the one E4
describes.* On 2026-08-21
`tests/test_import_cost.py::test_the_function_that_needs_sklearn_can_still_reach_it`
failed in `viz-sci-plot` with `ModuleNotFoundError: No module named 'sklearn'`, the suite
otherwise green. **scikit-learn is correctly declared** — `pyproject.toml` lists it in
`[project] dependencies`, not in an extra, so it is required rather than optional, and CI
installs `".[test,colormaps]"` which pulls the base list. The environment was simply
never re-installed after the line was added, so the package was present while one of its
required dependencies was not.

The lesson is the mirror of E4's, and worth keeping separate from it: a *declared*
dependency still does not reach an environment built before the declaration, and the
only signal is a test failing on a machine where the file says it should pass. Cleared by
`pip install -e ".[docs]"` on 2026-08-21, which resolved the base list and brought
scikit-learn 1.9.0 with it; the suite is 917 passed, 0 failed, and
`mkdocs build --strict` is green.

**Opened 2026-08-21 while closing E3:** **A30**. Same class as A25 — a figure drawn from
values that are not measurements — and it rides along with whatever next touches
`plot_spectral_map`'s rescale path. **Fixed the same day**, on its own rather than as a
passenger: it was the cheapest open entry in the file, the fix already existed in
`plot_image`, and the test already existed for `plot_image` to be modelled on. **A25** was
the cheapest one left after it, and was fixed on 2026-08-25 along with **A29** — the two
Raman-loader entries, taken as two changes. **A26** is the cheapest one left now.

Outside this order: **E9 is largely closed** — sample files arrived, and R/RC and
TRPL support landed on 2026-07-30 along with the rename and arbitrary-sweep rewrite
(folding in E1, part of E7b, and turning up A6). **E12** is the plotting half of that
work and should ride along with E3/E11. **E10** is questions only Brandon can answer;
nothing above depends on them, but every task touching data paths or notebooks does.

## Possible future work

**F1. A test job in CI — parked, not rejected.** **[FIXED — 2026-08-20, `a65b6fa`]**
Today CI (`.github/workflows/docs.yml`)
builds and deploys docs only, and CLAUDE.md records tests as local-only *by decision*;
that decision stands until deliberately revisited. Noted here because the groundwork
is already done rather than because it is owed.

The usual obstacle does not apply: every test is hermetic. `test_loaders.py` writes
synthetic CSVs into pytest's `tmp_path` and `test_processing_cosmic_rays.py` is pure
NumPy with fixed seeds, so nothing reads `data/`, `examples/data/`, or the gitignored
`.h5` files. What is missing is a `[project.optional-dependencies] test = ["pytest"]`
extra and roughly twelve lines of YAML — `docs.yml` with two lines swapped.

Two things such a job would settle as a side effect: whether the package really
supports the `requires-python = ">=3.9"` it advertises (a 3.9/3.12 matrix would say;
`constants.py` and `__init__.py` lack `from __future__ import annotations`, so it is
plausible but unverified), and whether the `ffmpeg` hard dependency is the package
that was intended, which a clean-room install would expose.

The case for doing it strengthens when other group members start opening PRs — a
solo maintainer running a 6-second suite locally gains little, but "works on my
machine" stops being a private problem once it is the group's standard workflow.

**Branch protection is part of the same decision.** `origin` already carries a rule
that changes to `main` must go through a pull request. The push on 2026-07-28 went
straight to `main` and the remote reported `Bypassed rule violations` — the
maintainer has permission to override it, so nothing failed. Nothing is broken and
nothing needs undoing, but the two settings only mean something together: tests that
run on pull requests gate nothing if changes reach `main` by direct push, and a PR
rule that is routinely bypassed is a rule in name only. So this is one decision, not
two — either drop the protection and keep pushing directly as sole maintainer, or
keep it, route work through PRs, and let CI be what makes the PR worth opening.
Settle it at the same time as the workflow file. It only starts to bite once someone
else commits.

> **When picking this up:** Brandon is new to CI. Explain what each part does and why
> before adding it, go a step at a time, and check understanding rather than landing a
> working workflow file in one move. The goal is that he can debug a red run himself
> afterwards, not just that CI is green.

*Done, 2026-08-20* — `a65b6fa` landed the test job, `37b9280` brought the `docs.yml`
actions up with it. See `dev/decisions/0027-the-suite-runs-in-ci-on-three-systems.md`.
Two claims in this entry were **wrong**, corrected here rather than edited above:

- **Not every test is hermetic.** Eight files read `examples/data/` directly:
  `test_raman.py`, `test_raman_map.py`, `test_loaders_trpl.py`, `test_hdf5_roundtrip.py`,
  `test_loaders_real_space.py`, `test_contrast.py`, `test_fitting_peaks.py`,
  `test_loaders_nesting.py`. CI works because that data is committed. The paths are
  repo-relative, so pytest must start in the checkout root — every workflow step does by
  default, and a `working-directory:` would break them.
- **The suite is 862 tests and ~175 s locally, not 41 and 6 s.** In CI it is 862 on each
  of three machines: about 1 min on Linux and macOS, 2 min on Windows.

Two things this entry expected the job to settle, and what actually happened:

- **E5 is not settled and CI will never flag it.** `ffmpeg` 1.4 — the unmaintained PyPI
  wrapper — resolves and installs cleanly on all three systems. A clean-room install
  cannot tell that it is the wrong package. E5 stands exactly as written.
- **`requires-python` is untouched.** Both workflows pin 3.12 and no 3.9 run has ever
  happened, so whether `>=3.9` is honest remains unverified.

Branch protection was settled at the same time, as this entry asked: the three checks are
required on `main`, `strict` (branches up to date) off, the admin bypass kept.

**F2. `cmocean` breaks when matplotlib removes `N` from `ListedColormap`.** `cmocean`'s
`cm.py` passes `N` to `ListedColormap`, which matplotlib deprecated in 3.11 and states it
will remove in **3.13**. Until then it is a warning: CI reports 89 warnings per run
against 1 locally, because CI installs matplotlib 3.11.1 while `viz-sci-plot` is on
3.10.9, where the deprecation does not fire. 88 of the 89 are this.

Nothing here is ours to fix, and there is no upgrade to take: `cmocean` 4.0.3 is the
newest release on PyPI and is the version CI installs. `cmocean.__version__` reports
`v3.0.3` regardless, so trust the distribution metadata rather than the attribute.

**What breaks when 3.13 lands, if upstream has not moved:** `import cmocean` raises
rather than warns. That takes out the `colormaps` extra, the CI install
(`pip install -e ".[test,colormaps]"` — see
`dev/decisions/0027-the-suite-runs-in-ci-on-three-systems.md`), and the three
`cmocean`-gated tests in `tests/test_plotting_cmap.py`. The `cmcrameri` half is
unaffected.

Three ways out when it bites, none of them urgent now: drop `cmocean` from the
`colormaps` extra, pin `matplotlib<3.13` for that extra, or wait for upstream. Recorded
here so the trigger is known in advance rather than discovered by a group member whose
install stopped working — which is precisely the class of problem CI was added to
surface, and did on its first run with the extra installed.

## Verification

- Add `pytest` to a `[project.optional-dependencies] dev` extra and install it —
  the suite is currently unrunnable in the active env.
  *(Update, 2026-07-28: pytest installed in `viz-sci-plot`; `python -m pytest -q`
  runs 41 tests green. Still undeclared in `pyproject.toml`.)*
  *(Re-checked 2026-07-29: 41 passed, pytest 9.1.1; still undeclared. See the E4
  update for the file-by-file breakdown and why declaring it matters.)*
  *(Done 2026-08-20, `a65b6fa`: declared as
  `[project.optional-dependencies] test = ["pytest"]`. 862 passed locally, and 862 on
  each of Linux, Windows and macOS in CI.)*
- Per bug, a focused unit test: `remove_cosmic_rays` on a synthetic spike
  (done — `tests/test_processing_cosmic_rays.py`);
  `repr(DeviceGeometry.from_single("WS2", 53, 46))` not raising.
- Smoke-run `examples/example_stark_shift.ipynb` and
  `examples/example_exciton_diffusion_power_scan.ipynb` against `examples/data/`
  after the diffusion and geometry fixes — these exercise A2 and A5 end to end.
- `python -m mkdocs build --strict` must still pass after the doc edits.

> Note: this file lives outside `docs/` on purpose. Anything under `docs/` is built
> and published to the public GitHub Pages site even when absent from `nav`.
