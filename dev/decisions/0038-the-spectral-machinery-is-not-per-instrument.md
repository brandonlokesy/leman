# 0038 — A sweep of spectra gets its own base, and it has no `__init__`

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-26 |

## Context

A second acquisition system — the BigTable — writes spectral sweeps of the same
physical quantities in a different file layout. Everything a loader for it needs
already exists: the correction ladder, the wavelength↔energy conversion, the
gate vocabulary, the sweep axis, the nest machinery. All of it sat on classes
named for the AttoCube.

Two things were mixed up in that arrangement.

**`_AttoCubeSweep` was not about the AttoCube.** It held the parameter store,
the curated registry, gate resolution, sweep-axis resolution, the nest and HDF5
export — what any sweep from any instrument needs.

**`AttoCubeSpectralSweep` was two classes wearing one name.** Three of the four
class attributes `dev/architecture.md` calls "the *entire* difference the base
class sees" describe the measured **axis**, not the instrument: `"wavelength"`,
`"spectra"`, `"pixels"`. Only `_LAYOUT_KIND` is about the exporter. So "this is
a sweep of spectra" was a real concept in the code with no name, and everything
belonging to it — `energy`, the correction ladder, `pixel_slice`,
`get_spectrum_at` — was reachable only through one instrument's class.

`_CURATED` mixed the two levels inside a single table: the *keys* are the
package's contract, named by `_SWEEP_TYPES`, `_SWEEP_REQUIRES`,
`_GATE_ROLE_CURATED` and `_ROLE_CURRENT_CURATED`, while the *rows* are one
exporter's. `power`'s `0.303e6` exists because the AttoCube writes a raw
photodiode voltage; the BigTable writes microwatts.

## Decision

1. **Three levels, and one question decides which.** `_Sweep` holds what is
   independent of *what the measured axis is*. `_SpectralSweep` holds what is
   independent of *which instrument wrote the file*, for a sweep of spectra. A
   loader holds its decoder, its rows and its `__init__`, and nothing else.
   `AttoCubeTRPLSweep` is a sibling of `_SpectralSweep`, not a child: no energy
   axis, no correction ladder.

2. **`_SpectralSweep` defines no `__init__`.** Each loader writes its own
   explicit signature and calls the shared steps by name at the point they
   happen:

   ```python
   self._check_spectral_arguments(...)     # 0. before the read
   payload = self._decode_and_describe(...) # 1. axis-independent
   self.wavelength = payload["wavelength"]  # 2. the arrays
   self._validate_payload()
   self._bind_aux_spectra(...)
   self._bind_sweep_axis(...)               # 3. needs n_sweeps
   self._bind_nesting(...)
   self._build_correction_ladder(..., stacklevel=4)
   ```

   This extends the rule already stated in `dev/architecture.md` §5. The order
   is load-bearing and a reader should be able to see it.

3. **The curated key set is shared; the rows are the instrument's.**
   `_Sweep._CURATED` and `_Sweep._SIBLING_CURRENT` are empty. Each instrument
   names its own tables at module level, and both AttoCube classes point at the
   same pair.

4. **`_LAYOUT_KIND` is not a test for which class an object is.** Where the
   HDF5 writer used `_LAYOUT_KIND == "spectral"` to mean "carries two regions of
   interest", each class now declares `_HDF5_SIGNALS`. Where it meant "has the
   spectral correction ladder", the writer asks
   `isinstance(scan, _SpectralSweep)`.

## Rejected

**A template-method `__init__` on `_SpectralSweep`, with a `_select_signal`
hook each loader overrides.** Fewer lines, and it was the first design tried. It
hides an order that has to be got right — the sweep axis cannot be resolved
before `n_sweeps` exists — behind a call a reader has to go looking for, which
`dev/architecture.md` §5 had already ruled out for the same reason. It also
breaks the rendered API page: `AttoCubeSpectralSweep.__init__` carries no
docstring of its own, so mkdocstrings prints its *signature* above the class
docstring's parameter table. An inherited `__init__` would print a signature
with no `roi=` above a table documenting `roi`, and `mkdocs build --strict`
cannot see that.

**`**kwargs` forwarding from each loader up to a shared `__init__`.** Barred by
*parameters earn their place*, and it moves a mistyped-argument error from load
time to first use — the opposite of why those two checks precede the read.

**A `_SpectralCorrections` mixin, with both loaders as direct children of
`_Sweep`.** Its one advantage is that it forces each loader to write its own
`__init__`, which decision 2 above already does. It buys nothing further and
costs a two-parent MRO in a module that has none.

**Keeping the instrument's rows on the shared base and overriding only the
label.** `power`'s scale and unit differ between the two systems as well, and
0029 settled that a scale and its unit are one fact that move together. The
label alone is not the instrument-specific part.

**Subclassing `AttoCubeSpectralSweep` directly for the new instrument.** One
change instead of five, and no risk to existing behaviour. But BigTable data
would be an `AttoCube…` object underneath for good, and that name reaches error
messages, `__repr__` and saved archives.

**Renaming `hdf5.FORMAT_NAME`** (`"tmdc_optics_tools.attocube_sweep"`) to match.
It is a string already written into every existing archive.

**Updating the old class name where `dev/defects.md` records what it was called
on a date.** Records are append-only; the result would be a sentence whose name
and date disagree.

## Consequences

- A new spectral instrument writes five things: a decoder, a `_CURATED`, a
  `_SIBLING_CURRENT`, a `_validate_payload` and an `__init__`. It inherits the
  whole correction ladder, both axes and every accessor.
- `_LAYOUT_KIND` now means only what it says: which export layout a CSV decoder
  accepts, plus the axis-kind lookup in `hdf5.write_sweep`.
- Two failures that leave both the suite and `mkdocs build --strict` green are
  now covered by `tests/test_sweep_class_surface.py`: a member that moves to the
  wrong class and stops being reachable (the failure `dev/defects.md` records as
  **C7**, where a page went from 32 members to 6), and a warning whose frame
  count starts blaming a line inside the package.
- `_build_correction_ladder` takes `stacklevel` as a parameter, the way
  `_order_by_iter` already does, because it sits one frame further from the
  caller than `__init__` did. Measuring that is how the standing `stacklevel`
  audit found that the Jacobian-without-background warning already overshoots
  the construction line by one; that off-by-one is unchanged here and still
  belongs to the audit.

## Load-bearing choices

- **`_bind_sweep_axis` and `_bind_nesting` are called straight from each
  loader's `__init__`, never from a shared helper.**
  `tests/test_loaders_nesting.py` pins a warning depth to exactly that call
  distance, and says so in its docstring.
- **Every rung of the ladder is assigned unconditionally, `None` included.**
  `_resolve_spectra` tells *this class offers no such correction* from *it was
  not requested* by whether the attribute exists, so a conditional assignment
  would turn the second message into the first.
