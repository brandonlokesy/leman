# Constants

Physical constants, material parameters, and literature values used throughout
the package.  All values are module-level and can be imported directly:

```python
from leman.constants import EPS_HBN, EXCITON_ENERGY
```

---

## Physical constants

| Name | Value | Description |
|---|---|---|
| `HC_EV_NM` | 1239.84 eV·nm | Product *hc* in eV·nm — used for λ ↔ E conversion |

---

## Dielectric constants

Out-of-plane relative permittivities used in the displacement field model.

| Name | Value | Material | Reference |
|---|---|---|---|
| `EPS_HBN` | 3.9 | hBN | Laturia et al. 2018 |
| `EPS_WS2` | 6.1 | WS₂ | npj 2D Mater. Appl. 2, 6 (2018) |
| `EPS_WSE2` | 7.4 | WSe₂ | — |
| `EPS_MOSE2` | 7.2 | MoSe₂ | — |
| `EPS_MOS2` | 6.2 | MoS₂ | — |

---

## Lookup dictionaries

### `T_MONOLAYER`

Monolayer thickness in nm, keyed by material name string.

```python
T_MONOLAYER = {"WS2": 0.65, "WSe2": 0.65, "MoSe2": 0.65, "MoS2": 0.65}
```

### `EPS_TMDC`

Dielectric constant lookup, keyed by material name string.
`"HS"` is a generic heterostructure approximation.

### `EXCITON_ENERGY`

Approximate intralayer exciton energies (eV) at low temperature for
hBN-encapsulated monolayers. Useful as starting guesses for peak fits.

```python
EXCITON_ENERGY["WS2"]["XA0"]   # → 2.02 eV
EXCITON_ENERGY["MoSe2"]["XA0"] # → 1.66 eV
```

| Material | Exciton | Energy (eV) | Reference |
|---|---|---|---|
| WS₂ | XA⁰ | 2.02 | Sci. Rep. 5, 9218 (2015) |
| WS₂ | XB⁰ | 2.41 | PRL 113, 076802 (2014) |
| WSe₂ | XA⁰ | 1.75 | Nature Nanotech. 8, 634 (2013) |
| MoSe₂ | XA⁰ | 1.66 | Nat. Commun. 4, 1474 (2013) |
| MoS₂ | XA⁰ | 1.86 | PRB 94, 075440 (2016) |
| MoS₂ | XB⁰ | 2.00 | PRB 94, 075440 (2016) |

### `INTERLAYER_EXCITON_ENERGY`

Approximate interlayer exciton energies (eV). Both orderings of the
heterostructure string are populated automatically at import time, so
`INTERLAYER_EXCITON_ENERGY["MoS2/WSe2"]` and
`INTERLAYER_EXCITON_ENERGY["WSe2/MoS2"]` are equivalent.

### `BINDING_ENERGY`

Exciton binding energies (eV).

| Material | Binding energy (eV) | Reference |
|---|---|---|
| MoS₂ | 0.310 | PRB 94, 075440 (2016) |
| WSe₂ | 0.500 | Nano Lett. 15, 6494 (2015) |
| WS₂ | 0.320 | PRL 113, 076802 (2014) |
| MoSe₂ | 0.550 | Nat. Mat. 13, 1091 (2014) |

### `BANDGAP_ENERGY_BL` / `BANDGAP_ENERGY_BULK`

Bilayer and bulk bandgap energies (eV) where available.

| Name | Material | Value (eV) | Reference |
|---|---|---|---|
| `BANDGAP_ENERGY_BL` | MoS₂ | 1.60 | PRL 105, 136805 (2010) |
| `BANDGAP_ENERGY_BULK` | MoS₂ | 1.29 | PRL 105, 136805 (2010) |
| `BANDGAP_ENERGY_BULK` | WS₂ | 1.40 | J. Phys. Chem. 86, 463 (1982) |
| `BANDGAP_ENERGY_BULK` | MoSe₂ | 1.10 | J. Phys. Chem. 86, 463 (1982) |
| `BANDGAP_ENERGY_BULK` | WSe₂ | 1.20 | J. Phys. Chem. 86, 463 (1982) |