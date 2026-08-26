# Physics conventions — what each symbol means, and when it is valid

This file answers **"is this number right?"** for every physical quantity the package
computes: what the symbol stands for, its defining expression, its units, and the
conditions under which it stops being true.

It does **not** describe how to obtain the value from a scan object — array shapes,
which attribute holds what, how a quantity is declared at load time. That is
`dev/architecture.md` §2, and it is not repeated here. Where a symbol appears in both,
the ownership is:

| Question | File |
|---|---|
| How do I get this value, what shape is it, how is it declared | `dev/architecture.md` §2 |
| What does it mean physically, what is its unit, when is it invalid | **this file** |
| What the instrument wrote and in what units | `dev/instruments/attocube.md` |
| Why the code responds to it the way it does | `dev/decisions/` |

---

## 1. Symbol table — code name to physical quantity

The physical reading of each symbol, and the thing about it that is most often got
wrong.

| Code symbol | Physical quantity | Unit | Most common misreading |
|---|---|---|---|
| `V_A`, `V_B` | raw acquisition-channel potentials, as the instrument wrote them | V | that they say *which electrode* — they do not; that is per-session wiring and must be declared |
| `v_top`, `v_bot` | potential on the top / bottom gate electrode | V | naming them by channel instead of role, which moves an undeclared wiring fact out of a record and into someone's head |
| `v_channel` | potential on a contact to the TMDC itself | V | treating it as a third gate — it sits *inside* the stack, carries no thickness, and enters no field |
| `ef` / `electric_field` | out-of-plane field in the 2-D layer, `E_2D` | mV/nm | that it is defined for a single-gated device — it is not; see §2 |
| `eps_stack` | ε of the homogeneous slab of thickness `d_tot` with the same capacitance as the real stack | — | conflating it with the TMDC-only value, or putting it in the denominator |
| `eps_2d` | TMDC-only series-capacitor permittivity | — | — |
| `EPS_TMDC["HS"]` | the **TMDC-only** permittivity of a heterostructure | — | reading "HS" as stack-wide; this key has the opposite scope to `eps_stack` |
| `gate_capacitance(gate)` | `ε₀ε_hBN / d_hBN` per unit area | F/m² | including the TMDC thickness — the TMDC is the *counter-electrode*, not a slab inside the capacitor |
| `carrier_density` | sheet carrier density **difference**, electrons positive | cm⁻² | reading it as absolute `n`; it is referenced to `v_ref`, a gate voltage, not to a threshold |
| `T_MONOLAYER` | monolayer thickness, 0.65 nm for all four materials | nm | that it is per-material |
| `power` | excitation power at the sample | µW | — |
| `scanner_x`, `scanner_y` | piezo **drive voltage**, not position | V | reading them as a distance; that needs a µm/V calibration no file contains |
| time axis (TRPL) | delay time | ns | that the file's `Wavelength` column name is meaningful here |
| `cosmic_ray_mask` | which pixels were replaced by the repair | bool | see §6 — a pixel flagged in *most* sweeps is a hot detector pixel, not a cosmic ray |
| `contrast`, `energy_contrast` | reflectance contrast `ΔR/R₀` | — | feeding it to a peak fit or an intensity colour bar; it is negative-going |
| `spectra`, `spectra_cr`, `spectra_bg` | detector counts at successive correction stages | counts | that a suffix names the *only* correction applied — it names the **last** one |

---

## 2. Displacement field

`DeviceGeometry.electric_field` is **exact as written**; it is not an approximation of
anything and does not simplify.

```
ε_2D · E_2D = ε_stack · E_stack,   E_stack = (V_BG − V_TG) / d_tot
⇒ E_2D = (V_BG − V_TG) / d_tot · (ε_stack / ε_2D)
d_tot = d_2D + d_hBN,top + d_hBN,bottom
```

Result in mV/nm.

**Why it is exact.** **D** is continuous with no free charge between the gates, so
`ε_i·E_i` is the same in every slab. `ε_stack` is by construction the ε of the
homogeneous slab of thickness `d_tot` with the same capacitance as the real stack, and
`ε_2D` is the TMDC-only series-capacitor value.

### Two wrong forms, both already shipped somewhere in the group

- **`ε_hBN` in place of `ε_stack`** in the numerator is the thin-TMDC approximation —
  *"pretend the whole stack is hBN"*. Low by `(d_2D/d_hBN)(1 − ε_hBN/ε_2D)`: 0.59% for
  53/46 nm hBN around a MoSe₂/WSe₂ bilayer, growing for thicker TMDC stacks or thinner
  hBN. This is what the group's MATLAB scripts compute and what this function did
  before 2026-07-30, so current fields are ~0.6% higher than older results. Not wrong,
  just not exact.
- **`ε_stack` in the denominator** instead of the numerator is wrong by ~1.8× and is
  *not* an approximation of anything. The old MATLAB reaches the correct answer only
  because its `eps_hs` line, `2*t*e_hbn*e_tmdc/(2*t*e_hbn)`, cancels algebraically to
  `eps_tmdc` for any input. Repairing that cancellation alone takes you from 0.6% low
  to 82% high.

A third form circulates in inherited group material: it keeps the `+ d_2D` term but
substitutes `d_tot` for `d_hBN`, which counts the TMDC twice. It is 1.29% low — worse
than the thin-TMDC approximation. Keeping `d_hBN` there instead makes it exact and
equal to the form above.

### Sign

Inherited group material is **internally inconsistent about the sign**: some
expressions for this field carry a leading minus and others do not. Do not inherit a
sign from it, and do not add a sign to the physics. Which electrode a channel drove is
not inferable from any file and must come from the lab notebook per session.

### Why there is no single-gate case

The derivation fails twice for a single-gated device: there is no second equipotential
to define `V_BG − V_TG`, and a grounded TMDC *is* the free charge that the
no-free-charge assumption excludes. One gate is also one degree of freedom, so field
and density are locked together — independent control of the two is exactly what a
dual-gate anti-symmetric sweep buys.

---

## 3. Carrier density

`carrier_density` sums `C_i(V_i − V_ref)/e` over the declared gates, with
`C_i = ε₀ε_hBN/d_hBN` per unit area, signed with electrons positive, in cm⁻².

- **The TMDC is the counter-electrode**, so neither its thickness nor `eps_stack`
  enters. Reaching for `eps_stack` here is the sign you have picked up the wrong tool.
- **The result is a density difference.** `v_ref` is a gate voltage, not a threshold.
  Absolute `n` needs the voltage at which the channel populates — a transfer curve or
  the PL charging step — which no file records.
- **Geometric only, so it is an upper bound.** Quantum and interface-trap capacitance
  are in series and make the effective value smaller.
- The density is referenced to the declared channel contact, so a *driven* channel
  moves the reference under the axis. Legitimate for a source-drain bias, wrong for a
  doping sweep, and no file distinguishes them.

---

## 4. Reflectance contrast

`ΔR/R₀ = (S − R)/R` against a bare-substrate reference.

- **The Jacobian cancels in a ratio.** `(S·λ²/hc)/(R·λ²/hc) = S/R` exactly, so an
  energy-axis contrast is built with the Jacobian off regardless of what was requested,
  and applying it to the numerator alone is an error.
- **Sample and reference must share an exposure.** For a reference scaled by `k`,
  `(S − kR)/(kR)` is a *biased* contrast, not a rescaled one — so a reference taken at
  a different integration time or excitation power gives a wrong answer that no later
  normalisation repairs. A 2-row reference CSV carries no parameter rows, so this
  cannot be checked or corrected automatically: matching the acquisition, or supplying
  the ratio explicitly, is the caller's responsibility. Same shape of problem as gate
  polarity.
- Background comes off **both** arrays before the ratio — a pedestal in either biases a
  contrast non-linearly.
- A grid mismatch between sample and reference is refused rather than interpolated:
  resampling changes the numbers and smooths the data.

---

## 5. Fit baselines

Peak models decay to zero in their wings, so an un-subtracted dark-count pedestal is
otherwise absorbed by inflating amplitude and FWHM. A Lorentzian's 1/x² wings are
partly degenerate with a flat offset, so FWHM is **more** window-sensitive when a
baseline term is fitted than when it is not; the centre is set by symmetry and stays
robust either way.

---

## 6. Cosmic rays and hot pixels

The repair identifies outliers per pixel along the sweep axis. The physical
distinction it cannot make on its own:

- A **cosmic ray** hits one pixel in one exposure. It appears in a single sweep.
- A **hot or damaged detector pixel** reads high in *every* exposure. It appears in
  most or all sweeps.

So a pixel flagged across most sweeps is a detector defect, and repairing it silently
removes a systematic that the researcher needs to know about — which is why that case
warns and names what was affected rather than being quietly cleaned or quietly kept.

---

## 7. Lifetimes inherit the time-axis assumption

The TRPL bin width and range are an instrument fact recorded in
`dev/instruments/attocube.md`, and they are **not independently confirmed**. Any fitted
lifetime inherits that assumption directly, and `_TRPL_TIME_UNIT` is the single place
to change it.

---

## 8. Open questions — provenance not yet established

Ask before documenting or changing any of these.

- `power_scale = 0.303e6`, attributed to "calibrated by CdG" — when, which
  objective/filter set, and does it vary per session?
- `EPS_TMDC["HS"] = 7.5` is unsourced and sits among per-material values. It is *not*
  the harmonic mean of a MoSe₂/WSe₂ bilayer (that is 7.299), and it exceeds both
  constituents, so it is not an average of the values above it either.
- `T_MONOLAYER = 0.65` nm for all four materials — a deliberate approximation, or an
  unfinished table?
- `EPS_HBN = 3.9` — the four TMDC values do match the cited paper's bulk out-of-plane
  figures, but that paper's hBN out-of-plane value is usually quoted as **3.76**, and
  3.9 is also the canonical SiO₂ value. Worth checking against the table directly; it
  propagates into `eps_stack`, though only weakly into `electric_field` now that the
  exact form is used.

## 9. Citations owed

Under the project's citation rule, a reference must let a reader find the source.
Five references in this package currently do not, and none can be completed without
information only the group has:

| Reference as it stands | What is missing |
|---|---|
| "the senior's thesis" — source of the thin-TMDC form and of the sign inconsistency in §2 | author, year, title, institution. Equation numbers were removed from this file because without the thesis itself they point nowhere. |
| "Laturia et al. 2018" — source for `EPS_HBN` and the four TMDC permittivities | full reference and DOI, plus which table the values were read from |
| "calibrated by CdG" — source of `power_scale` | who, when, and against what setup |
| the BigTable parameter row map — what each positional row of that export is | a document from the acquisition program, or the program itself. The names in `_BIGTABLE_ROWS` come from a MATLAB snippet written by a colleague who uses the setup; it cannot be cited, and it maps row 9 to two different quantities and calls row 13 a gate current in amperes at a magnitude of 19 mA. Rows 10, 12, 13, 17 and 30 are left unnamed rather than guessed. Detail in `dev/instruments/big-table.md`. |
| `PL_PEAKS["WSe2"]` seed positions — X0, XT and IX for WSe₂ | a source for each seed, or a statement of which spectra they were tuned against and by whom. Note that `EXCITON_ENERGY["WSe2"]["XA0"]` gives 1.75 eV for the same neutral intralayer exciton that `PL_PEAKS` seeds at 1.70 eV; the two have not been reconciled. |

Until these are filled in, the claims that rest on them are recorded here as
*inherited group practice*, not as literature values.

## 10. Raman modes — WSe₂, by layer count

`examples/data/Raman/*.txt` is WSe₂, bilayer or monolayer per the filename
(`*bilayer*` / `*monolayer*`) and per session identification. The LabRAM header carries
no material or layer-count field, so neither is independently checkable from the data —
see `dev/instruments/labram.md`.

Fitting the six example spectra with `fitting.fit_raman_modes(..., material="WSe2",
n_layers=2)` and `n_layers=1` finds modes consistent with Pan et al., *"Signature of
lattice dynamics in twisted 2D homo/hetero-bilayers"*, 2D Materials **9**, 045018
(2022), doi:10.1088/2053-1583/ac83d4.

| Fitted, bilayer | Fitted, monolayer | Literature | Assignment |
|---|---|---|---|
| ≈250.5–250.6 cm⁻¹ | ≈250.1 cm⁻¹ | ≈250 cm⁻¹ | E₂g/A₁g, nearly degenerate |
| ≈258.6–258.8 cm⁻¹ | ≈260.3–260.5 cm⁻¹ | ≈260 cm⁻¹ | 2LA(M), second-order double resonance |
| ≈309–309.3 cm⁻¹ | **absent** | ≈309 cm⁻¹ | B₂g |

### Why ≈250 cm⁻¹ is one peak and not a doublet

E₂g and A₁g are *nearly* degenerate, which is why they do not split into two resolvable
peaks at this resolution. Treating ≈250 cm⁻¹ as a splittable doublet is the wrong model:
it is not what the cited paper reports, and a two-peak fit there does not converge on
these spectra.

### Why 2LA(M) is not doublet-adjacent

2LA(M) is second-order (double resonance), a different scattering mechanism from the
first-order E₂g/A₁g and B₂g modes. That is why it is roughly 10× weaker, and why seeding
it near ≈250–253 cm⁻¹ — as though it were the other half of a doublet — either pins at a
fit bound or fails to converge. Only a seed near its actual position converges cleanly.

Its position is found from the data rather than assumed: the residual of a
main-peak-only fit locates it, and the result matches the paper's ≈260 cm⁻¹ after the
fact. It is **not** taken as equal between bilayer and monolayer merely because the mode
carries the same name in both — ≈258.7 vs ≈260.4 cm⁻¹ is a real difference, not fit
noise.

### Why B₂g is absent in the monolayer

Every monolayer spectrum checked has a flat baseline at ≈309 cm⁻¹ — not a small or
unresolved peak. This is consistent with B₂g requiring interlayer coupling, which a
single layer does not have. `constants.RAMAN_MODES["WSe2"][1]["modes"]` therefore lists
two modes and not three; adding B₂g there does not cause the fit to drop it.
