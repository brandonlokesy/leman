# E12 — `plotting` still speaks gate-sweep PL

Written 2026-08-04. This is a **plan**, not a record of landed work: the code
described here was implemented, verified, and then reverted at the author's
request so it can land deliberately rather than as one large pass. Finding E12
itself is in `defects.md`; this file is the fix, in enough detail to
re-apply mechanically.

Status of the tree when this was written: the `plot_pl_map_Vab_scan` →
`plot_spectral_map` rename **has** landed (with its `FutureWarning` shim and
`tests/test_plotting_spectral_map.py`). Nothing below is applied.

**Update 2026-08-06.** Steps 1 and 2 are now **landed**, with one design
reversal: `colorbar_label` was kept rather than deleted, and generalised into a
module-wide *`None` derives / a string is verbatim* contract. See *Why
`colorbar_label` stays* below, which replaces the section that argued the
opposite. Steps 3 and 4 remain unapplied.

---

## The problem, restated

`plotting` lags the loader in two ways. Neither breaks anything, which is why it
survived the 2026-07-30 rewrite.

1. **Hardcoded "PL intensity"** in five places. A reflectance sweep is labelled
   as PL. `scan.signal_label` exists for exactly this.
2. **Names and arguments carry the gate-sweep assumption** —
   `plot_current(ef_axis=)`, `plot_spectrum`'s hand-rolled `E_F`/`V_top` legend
   label, `SpectrumLinePanel(sweep_attr="scanner_y", sweep_unit="V")`.

Plus a gap that grew after the finding was written: `contrast_label` exists and
nothing uses it, so `spectra_source="contrast"` reaches the array and then
labels the axis as an intensity.

---

## Step 1 — the vocabulary is wrong before any plotting change

**[LANDED 2026-08-06.]** The `SIGNAL_LABELS` half had already gone in ahead of
this file's other steps, but *without* its test update — leaving
`test_contrast.py::test_spectra_type_not_mutated_by_supplying_a_reference` red
on `main`, pinning the old `"Reflectance (counts)"`. Corrected with the rest of
this step. `signal_unit` is added; a dead second `signal_label` property
(a bare `return`, shadowed by the real one later in the same class body) was
deleted from the curated-parameter section on the way past.

`constants.SIGNAL_LABELS` has a physics error worth fixing on its own, and
fixing it first means steps 2–3 have something correct to read.

```python
"R":  ("Reflectance",   "counts"),      # a ratio, given a unit
"T":  ("Transmission",  "counts"),      # same
```

Reflectance is *defined* as R = I_r/I_i ∈ [0, 1] — dimensionless. The loader
reads R through the identical spectral layout as PL, i.e. raw CCD counts. So a
counts-valued array labelled "Reflectance" asserts a normalisation that has not
happened. Transmittance likewise.

**The rule:** a CCD integrates photons, so everything it records directly is an
*intensity* in counts, whichever beam produced it. The ratio names belong to the
derived arrays, whose labels come from `contrast_label`.

```python
SIGNAL_LABELS = {
    "PL":   ("PL intensity",          "counts"),
    "R":    ("Reflected intensity",   "counts"),   # renamed
    "RC":   (r"$\Delta R/R_0$",       ""),
    "T":    ("Transmitted intensity", "counts"),   # renamed
    "A":    ("Absorbance",            ""),
    "TRPL": ("PL intensity",          "counts"),
}
```

`"A"` is already right: absorbance = −log₁₀T is dimensionless, and
"Absorption" in `SPECTROSCOPY_TYPES` (the technique) versus "Absorbance" here
(the quantity) is correct usage, not a mismatch — it is exactly the distinction
the `R`/`T` rows were getting wrong.

Carry the reasoning as a comment above the dict, since the next person to add a
row needs the rule, not just the result.

### `signal_unit` on `_Sweep`

`signal_name` (bare) and `signal_label` (composed) exist; the unit alone is not
reachable, which is *why* `plotting` hardcodes a composed string instead of
recomposing one. Add the missing third member, on the base class so both the
spectral and TRPL classes get it:

```python
@property
def signal_unit(self) -> str:
    """
    Native unit of :attr:`signal_name`; empty for dimensionless ratios.

    Exposed separately from :attr:`signal_label` so a caller that rescales
    the data can substitute its own unit rather than parse one out of a
    composed string.
    """
    return SIGNAL_LABELS[self.spectra_type][1]
```

This is not a new pattern — it is the trio the sweep axis already has
(`sweep_label` / `sweep_unit` / `sweep_axis_label`).

Also update: the `ACSpectralSweep` class docstring `Attributes` block, and
`contrast_label`'s docstring, which quotes the old `"Reflectance (counts)"`.

### Naming asymmetry, noted and deliberately not fixed

The sweep trio uses `_label` for the bare string and `_axis_label` for the
composed one; the signal trio uses `_name` for bare and `_label` for composed.
Same shape, opposite convention. Pre-adoption, so it is cheap to align —
preferred direction is `signal_name`/`signal_unit`/`signal_label` kept as-is and
the sweep's bare `sweep_label` renamed to `sweep_name`, since "label" reading as
"the composed thing you put on an axis" is the more useful convention. Left out
of this plan because it is a rename of a public loader property with its own
blast radius (HDF5 metadata keys, `SpectrumLinePanel` attribute of the same
name) and bundling it makes the pass harder to review.

---

## Step 2 — two helpers, then the label sites

**[LANDED 2026-08-06.]** The helpers as implemented differ from the sketch
below: they split into `_signal_name_unit` (returns the `(name, unit)` pair) and
`_signal_label` (composes it). `plot_power_series`'s `spectrum_offset` branch
needs the pair, not a composed string, so it can swap the unit for
`"a.u., offset"` — splitting the primitive out avoids a `normalized` × `offset`
combinatorial parameter. Six sites, not five: `plot_image` came along because
its `rescale_img` branch was silently discarding the caller's label.

Both go next to `_resolve_x_axis` in `plotting.py`.

```python
def _signal_label(scan, normalized: bool = False, source: str = None) -> str:
    """
    Return the y-axis (or colour-bar) label for a scan's measured signal.

    The quantity and its native unit come from the scan, so a reflectance sweep
    is not labelled as PL.  Only the calling plot function knows whether it
    rescaled the values, and a rescaled array has no unit left — hence
    *normalized*, which substitutes "(norm.)" for whatever the native unit was.

    *source* is a ``_SPECTRA_SOURCES`` key; a contrast source is a different
    physical quantity from the scan's raw signal and takes ``contrast_label``.

    Objects that declare no measurement type fall back to a neutral
    "Intensity" — a :class:`~leman.loaders.SingleSpectrum` is a
    2-row CSV as likely to be a bare-substrate reflectance reference as PL.
    """
    if source is not None and source.startswith("contrast"):
        # contrast_label is already complete: the ratio is dimensionless, so
        # there is no native unit for "(norm.)" to replace.
        name = getattr(scan, "contrast_label", r"$\Delta R/R_0$")
        return f"{name} (norm.)" if normalized else name

    if normalized:
        return f"{getattr(scan, 'signal_name', 'Intensity')} (norm.)"
    return getattr(scan, "signal_label", "Intensity (counts)")


def _sweep_value_label(
    scan,
    index : int,
    fmt   : str = "{label} = {value:.3g} {unit}",
) -> str:
    """
    Describe one sweep point, e.g. ``"$E_F$ = -3.2 mV/nm"``.

    Reads the declared sweep axis, so the string names whatever was swept
    rather than assuming a gate voltage.  A sweep with no unit (an undeclared
    sweep, or a raw CSV row) leaves the ``{unit}`` field empty; the result is
    stripped so it does not end in a stray space.
    """
    return fmt.format(
        label = getattr(scan, "sweep_label", "Sweep index"),
        value = scan.sweep_axis[index],
        unit  = getattr(scan, "sweep_unit", ""),
    ).strip()
```

Two things that only became clear while writing them:

- **`SingleSpectrum` has no `spectra_type`**, so `plot_single_spectrum` cannot
  ask. The `getattr` fallback to `"Intensity (counts)"` is not a compromise but
  an improvement on the hardcoded PL: a 2-row CSV in this package is most often
  the bare-substrate reflectance reference.
- **`normalized` and `source` are separate axes.** A contrast is already
  unitless, so `(norm.)` on it replaces nothing; a rescaled intensity loses a
  real unit. Collapsing them into one argument gets one of the two wrong.

### The call sites

As landed — every one takes the shape
`label if label is not None else _signal_label(...)`:

| Function | Change |
|---|---|
| `plot_spectral_map` | `colorbar_label=None`; the `" (counts)"` / `" (norm.)"` append is gone |
| `plot_spectrum` | **new** `ylabel=None` → `_signal_label(scan, normalized=normalize)`. Legend default left alone — that is Step 3 |
| `plot_single_spectrum` | **new** `ylabel=None` → `_signal_label(spectrum, normalized=normalize)`; hits the `getattr` fallback |
| `plot_power_series` | `ylabel=None` rebuilt on `_signal_name_unit(scan, spectra_source)`, so a contrast source finally gets `contrast_label` |
| `SpectrumLinePanel` | `ylabel="Counts"` → `None` → `_signal_label(scan)` |
| `plot_image` | `colorbar_label=None`; **bug fix**, the argument was discarded whenever `rescale_img=True`. Untyped array, so it defaults to "Intensity (counts)" rather than calling `_signal_label` |

`normalized` **substitutes** a unit and never adds one, so a dimensionless
quantity is left alone: a ratio such as ΔR/R₀ already reads as normalised, and
`$\Delta R/R_0$ (norm.)` says the same thing twice. Asked and settled
2026-08-06. The residual case — `rescale_img=True` on a scan whose
`spectra_type` is itself `"RC"`, where the values shown are a [0, 1] remap
rather than the ratio — is what the verbatim override is for.

`plot_power_series` passes no `normalized`: `bg_region` subtracts a pedestal but
leaves the unit intact, so only the source decides the label there.

### Why `colorbar_label` stays, defaulting to `None`

**Reversed 2026-08-06, and landed.** This section previously argued for deleting
`colorbar_label` and routing callers to `mesh.colorbar.set_label(...)`. Three
things in the tree decided against it:

- **The convention already shipped.** `plot_power_series(ylabel=None)` already
  meant *`None` → derive from the scan; a string → verbatim*. Deleting the
  equivalent on `plot_spectral_map` would not have left the module
  parameter-free, only inconsistent.
- **`colorbar_label` had three incompatible contracts**, which is the actual
  defect: `plot_spectral_map` appended a unit, `plot_image` *discarded* the
  argument whenever `rescale_img=True`, `plot_diffusion_cloud` used it verbatim.
- **The `"PL intensity (norm.) (norm.)"` bug came from the append**, not from
  the parameter existing. A verbatim contract makes it unrepresentable.

*Parameters earn their place* also states the boundary that applies here: *"Ask
whether the plot could be misread without the argument; if so, it is not
trivial."* A colour bar reading "PL intensity (counts)" over a
reflectance-contrast map is that misread — a label is semantics, not decoration.
And the escape route relied on, `mesh.colorbar`, is an informal matplotlib
back-reference that is `None` whenever `colorbar=False`.

**The contract, now uniform across every signal-label parameter in the module:**
`None` derives from the scan, a string is used exactly as given, and nothing is
ever appended to a caller's string.

```python
plot_spectral_map(scan)                                   # "PL intensity (counts)"
plot_spectral_map(scan, rescale_img=True)                 # "PL intensity (norm.)"
plot_spectral_map(scan, colorbar_label=r"$\Delta R/R_0$") # exactly that
```

---

## Step 3 — the renames

**Do the `plot_current` bullet first.** E15 (2026-08-07) already changed that
function once — `color_ich1` / `color_ich2` deleted, `lines` appended to the return —
and this step changes it again. Both are breaking; landing them apart breaks callers
twice over one function.

- **`plot_current(ef_axis=True)` → `sweep_axis=True`.** Body becomes
  `x, xlabel = scan.sweep_axis, scan.sweep_axis_label`, with
  `np.arange(scan.n_sweeps)` / `"Sweep index"` when `False`. The `False` branch
  is not dead weight: the index is the useful reading when the sweep is
  non-monotonic, e.g. a raster flattened into one file (cf. A8).

  Current signature, after E15 — the `ef_axis` line is the only one this bullet
  touches, and the `scan.ef` branch it names is the block the new body replaces:

  ```python
  def plot_current(scan, ax=None, figsize=(6, 3.5), dpi=None,
                   ef_axis=True, color_power="C2") -> tuple:
      ...
      return fig, ax_left, ax_right, lines
  ```

  Two things E15 changed that this bullet must not undo. The current traces are
  built by looping the declared roles (`i_top`, `i_bot`, `i_channel`) and skipping
  those the scan refuses, then raising if none survived — that loop stays. And
  **the function now requires `gates=`**, so any test fixture for it needs a
  declared mapping; without one a current row cannot be attributed to an electrode
  and every role raises. `color_power` survives here and is still owed to E11.
- **`plot_spectrum`'s legend default** — drop the hand-rolled `E_F`/`V_top`
  branch for `_sweep_value_label`.
- **`SpectrumLinePanel`** — `sweep_attr`, `sweep_label`, `sweep_unit`, `ylabel`
  all default to `None` = ask the scan. `init_artists` reads
  `scan.sweep_axis` when `sweep_attr is None`. `_frame_title` gains `.strip()`
  so a unitless sweep does not leave a trailing space.
- **`animate_wl_pl_spectra`** — `sweep_attr=None, sweep_unit=None`, passed
  through unchanged.

**A named `sweep_attr` must not inherit the scan's sweep label or unit** — it is
a different array. So naming one falls back to the attribute name and no unit;
only `sweep_attr=None` reads `sweep_label`/`sweep_unit`.

### Behaviour change this causes, and why it is correct

`animate_wl_pl_spectra` used to default `sweep_attr="scanner_y", sweep_unit="V"`,
so a bare-path call captioned frames `"scanner_y = 7 V"`. Reading the declared
sweep instead means a bare path — which `_spectrum_scan` builds with no
`sweep=` — captions by index: `"Sweep index = 0"`.

Auto-detecting `piezo_y` would violate G1 (*an undeclared sweep means the index,
never an auto-detected parameter*). So the fix belongs at the call site: pass a
pre-built sweep with `sweep="piezo_y"`. Say so in the `spectra` docstring entry.

### Correction owed to the audit

The E12 entry in `defects.md` calls the old `sweep_unit="V"` default "a position
default carrying a voltage unit, **wrong** before this rewrite and now
redundant". The 2026-08-04 answer that the scanner rows carry piezo *drive
voltage* makes it not wrong — only redundant. Fix that line when this lands.

### Not renamed, deliberately

`plot_real_space_PL_map` / `animate_real_space_PL_map` stay. Real-space imaging
genuinely is a PL measurement here; renaming for symmetry would make them less
accurate, not more.

`plot_image` and `plot_diffusion_cloud` keep `colorbar_label="Intensity
(counts)"`. They plot real-space images rather than typed scans, and
`plot_diffusion_cloud`'s style parameters belong to E11.

---

## Proposed improvement — delete the panel's sweep parameters entirely

Raised after step 3 landed, and it goes further than step 3 did. **Not
implemented; needs a decision first.**

Step 3 only *reduced* the duplication — three parameters defaulting to the scan
instead of to `"scanner_y"`/`"V"`. But the loader signature is:

```python
def __init__(self, path, *, spectra_type=None, sweep=None,
             sweep_label=None, sweep_unit=None, ...)
```

**The loader already takes `sweep_label=` and `sweep_unit=`**, and
`_bind_sweep_axis` falls back to `meta.get("sweep_label")`, so a declared label
round-trips through HDF5. On the normal path the panel's two label parameters
are therefore not merely inferable — they are a worse-placed copy of a
capability that already exists one layer down, where it is declared once and
reaches every plot and every export.

The same argument covers `sweep_attr` itself: `sweep=` accepts any raw CSV row
label, so "caption by Galvo_X" is already

```python
ACSpectralSweep(path, spectra_type="PL", sweep="Galvo_X",
                      sweep_label=r"Galvo $x$", sweep_unit="V")
```

which is `sweep_attr` + `sweep_label` + `sweep_unit` collapsed into the
declaration that also fixes the x-axis of every other plot of that scan.

**The change:** delete all three from `SpectrumLinePanel`, have it read
`scan.sweep_axis` / `sweep_label` / `sweep_unit` unconditionally, and drop the
`animate_wl_pl_spectra` passthrough. ~15 lines out, three docstring entries
gone, two tests rewritten. Makes `sweep=` on the loader the single place a sweep
axis is ever described.

**Open question that blocks it:** this forecloses captioning frames by an array
that is *not* the axis being animated — animate along piezo y, caption with gate
voltage. The panel shows only one value, so it looks marginal, but whether the
group does this is not answerable from the code.

### Why `ylabel=` is a different case and stays

Worth stating, because it looks like an inconsistency and is not:

- **Sweep labels are per-file facts the file does not state.** A raw row's unit
  is unknowable from the export, so the caller must be able to declare it —
  hence `sweep_label=`/`sweep_unit=` on the loader.
- **Signal labels are fixed by a closed declared vocabulary.** `spectra_type` is
  required, validated against `SPECTROSCOPY_TYPES`, written into HDF5 and
  trusted thereafter (G1). A per-scan `signal_label=` override would reopen
  exactly what that decision closed — a guess outliving the session.

So the signal side gets no loader override, and `plot_power_series(ylabel=)`
stays as the plot-local escape hatch for a derived array the package does not
model.

---

## Step 4 — TRPL plotting, out of scope for this pass

`_resolve_x_axis` knows only `"energy"` and `"wavelength"`; a TRPL sweep exposes
neither, so it raises `AttributeError`. The shape to aim for is to let the scan
answer rather than adding a third hardcoded branch:

```python
if x_axis == "native":
    return scan.axis, scan.axis_label     # TRPL: time (ns)
```

which needs a matching `axis` / `axis_label` pair on both classes
(`ACTRPLSweep.axis_label` exists; the spectral class has no `axis`).

Deferred because it is new public surface rather than a correction, and because
it is not useful without a lifetime fit — `fitting.py` has no exponential model.
The baseline machinery (`_with_baseline`, `_complete_p0`, `_complete_bounds`,
`_make_result`) is already generic over the model function; only
`_fit_single_peak` hardcodes the 3-parameter `_PEAK_PARAM_NAMES`.

---

## Tests

`tests/test_plotting_labels.py`, 28 tests. Three pin mechanisms rather than
values, and are the ones worth keeping if the file is trimmed:

- **`test_ratio_quantities_are_dimensionless`** asserts the invariant over
  `SIGNAL_LABELS` — a ratio name iff an empty unit — so re-adding
  `("Reflectance", "counts")` fails regardless of which key it goes under.
- **`test_no_unit_is_appended_twice`** reproduces the concrete defect the
  compose-then-append design invited.
- **`test_dimensionless_type_gets_no_unit`** covers what a blanket
  `f"{name} (counts)"` gets wrong, in both the plain and normalised cases.

The rest: colour-bar and ylabel parametrised over `spectra_type`; the contrast
source in both `contrast` and `ratio` modes (previously untested — asserting the
contrast gets `contrast_label` while the raw source keeps its own); the
`SingleSpectrum` fallback; sweep-value legend labels declared and undeclared;
`plot_current` on both branches; `SpectrumLinePanel` defaults and the
named-attribute override.

Fixtures reuse `make_spectral_csv` from `test_loaders` and `_write_reference`
from `test_contrast`.

**One existing test needs updating:** `test_contrast.py:251` pins
`"Reflectance (counts)"`. Its subject is that `spectra_type` survives a
`reference=`, not the wording, so the expected string changes to
`"Reflected intensity (counts)"` with a comment on why.

Two gotchas found the hard way, recorded so they are not rediscovered:

- `plot_spectral_series` (named `plot_power_series` when this was written) returns
  `(fig, ax, cb, lines, ax_twin)`, not the module's usual 3-tuple.
- `pcolormesh` with `shading="auto"` resolves to `"nearest"` for equal-shaped
  X/Y/C, so `mesh._coordinates` holds cell *edges* — outer edges extrapolated
  half a cell beyond the data. Averaging adjacent edges recovers the centres.

---

## Verification when this was implemented

- `python -m pytest -q` → **203 passed** (from 175 before the pass).
- `python -m mkdocs build --strict` → green, no broken anchors.

Sequence to land it in, each step independently testable:

1. `SIGNAL_LABELS` + `signal_unit`. Pure vocabulary and one property; the
   existing suite should stay green apart from `test_contrast.py:251`.
2. `_signal_label` / `_sweep_value_label` + the five call sites, with the label
   tests.
3. The renames. Touches `examples/`.
4. TRPL plotting, with the fitting side.

The proposed panel-parameter deletion slots in after 3, once the
caption-by-a-different-array question is answered.

Not bundled: E3's `median_kernel` default and E11's `plot_diffusion_cloud`
signature work. The audit wanted them landed together with the rename to spare
`examples/` repeated passes; the rename has already gone in alone, so forcing
them together now only makes one commit harder to review.
