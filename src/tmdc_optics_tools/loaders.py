# tmdc_optics_tools/loaders.py
"""
Data loaders and device geometry for TMD heterostructure measurements.

Classes
-------
DeviceGeometry
    Encodes the physical geometry and dielectric constants of a vdW stack.
AttoCubeSpectralSweep
    A sweep of spectra taken on the AttoCube cryogenic confocal — any
    measurement type, over any scanned parameter.  Reads the raw CSV export or
    an HDF5 file written by its own :meth:`AttoCubeSpectralSweep.to_hdf5`.
AttoCubePLVabScan
    Deprecated pre-rename name for the above, fixed to PL gate sweeps.
SingleSpectrum
    Single spectrum from a 2-row CSV.
RamanSpectrum
    Single Raman spectrum from a LabRAM-style ``.txt`` export.
RamanMap
    2-D spatial Raman map from a LabRAM-style ``.txt`` export.
AttoCubePLScanRealSpace
    Loads a sequence of real-space PL image CSVs swept over gate voltage.
_AttoCubeImage
    Internal base class shared by AttoCubeSampleImage and
    AttoCubeLaserReferenceImage.
AttoCubeSampleImage
    White-light reference image of the sample.
AttoCubeLaserReferenceImage
    Laser-spot reference image with fitted 1/e² radius.
"""

from __future__ import annotations

import inspect
import re
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
from skimage.morphology import white_tophat, disk
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
import matplotlib.patches as patches

from .constants import (
    E_CHARGE,
    EPS_0,
    EPS_HBN,
    EPS_TMDC,
    HC_EV_NM,
    SIGNAL_LABELS,
    SPECTROSCOPY_TYPES,
    _x_axis_name_unit,
)

from . import processing
from .processing import jacobian_correction_wvl2E, subtract_background

# ---------------------------------------------------------------------------
# StackLayer — one material slab in the heterostructure
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

@_dataclass
class StackLayer:
    """
    One TMDC (or generic dielectric) slab in a vdW heterostructure stack.

    Parameters
    ----------
    material : str
        Material name, e.g. ``"WS2"``, ``"MoSe2"``.  Used to look up
        default thickness and dielectric constant from ``constants.py``
        when *d_monolayer* or *eps* are not supplied explicitly.
    n_layers : int
        Number of monolayers of this material.  Default 1.
    d_monolayer : float, optional
        Monolayer thickness in **nm**.  If ``None``, looked up from
        :data:`~tmdc_optics_tools.constants.T_MONOLAYER`.
    eps : float, optional
        Out-of-plane dielectric constant.  If ``None``, looked up from
        :data:`~tmdc_optics_tools.constants.EPS_TMDC`.

    Examples
    --------
    >>> StackLayer("MoSe2")               # 1 ML, defaults from constants
    >>> StackLayer("WSe2", n_layers=2)    # 2 ML WSe2
    >>> StackLayer("WS2", d_monolayer=0.65, eps=7.2)   # explicit override
    """
    material    : str
    n_layers    : int   = 1
    d_monolayer : float = None   # resolved in __post_init__
    eps         : float = None   # resolved in __post_init__

    def __post_init__(self):
        from .constants import T_MONOLAYER, EPS_TMDC

        if self.d_monolayer is None:
            if self.material not in T_MONOLAYER:
                raise ValueError(
                    f"No monolayer thickness for '{self.material}' in T_MONOLAYER. "
                    f"Pass d_monolayer explicitly."
                )
            self.d_monolayer = T_MONOLAYER[self.material]

        if self.eps is None:
            if self.material not in EPS_TMDC:
                raise ValueError(
                    f"No dielectric constant for '{self.material}' in EPS_TMDC. "
                    f"Pass eps explicitly."
                )
            self.eps = EPS_TMDC[self.material]

    @property
    def thickness(self) -> float:
        """Total thickness of this slab in nm (n_layers × d_monolayer)."""
        return self.n_layers * self.d_monolayer

    def __repr__(self) -> str:
        return (
            f"StackLayer({self.material}, n_layers={self.n_layers}, "
            f"d={self.thickness:.3f} nm, ε={self.eps})"
        )


# ---------------------------------------------------------------------------
# DeviceGeometry
# ---------------------------------------------------------------------------

class DeviceGeometry:
    """
    Physical geometry and dielectric constants of a vdW heterostructure.

    The heterostructure is modelled as a vertical stack of dielectric slabs
    in series (series-capacitor model).  The effective dielectric constant
    is computed from the general formula:

        d_total / ε_eff = Σ_i  d_i / ε_i

    where the sum runs over every slab (hBN top, TMDC layers, hBN bottom).

    Parameters
    ----------
    tmdc_stack : list of StackLayer
        Ordered list of TMDC (or other dielectric) slabs between the two
        hBN layers.  For a simple monolayer use
        ``[StackLayer("WS2")]``; for a heterostructure use e.g.
        ``[StackLayer("MoSe2"), StackLayer("WSe2")]``.
    d_hbn_top : float or None
        Top hBN thickness in nm.  Pass ``None`` for a device without a
        top hBN encapsulation layer (e.g. no top gate dielectric).
    d_hbn_bottom : float or None
        Bottom hBN thickness in nm.  Pass ``None`` likewise.
    eps_hbn : float
        Out-of-plane hBN dielectric constant.  Defaults to
        :data:`~tmdc_optics_tools.constants.EPS_HBN`.
    label : str, optional
        Human-readable description of the stack, e.g.
        ``"hBN/MoSe2/WSe2/hBN"``.  For record-keeping only.

    Class methods
    -------------
    from_single(tmdc, d_hbn_top, d_hbn_bottom, ...)
        Convenience constructor for single-material stacks — preserves
        the old interface so existing code does not need to change.

    Examples
    --------
    **Simple monolayer (old-style, via classmethod):**

    >>> geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)

    **Heterobilayer MoSe2/WSe2:**

    >>> geom = DeviceGeometry(
    ...     tmdc_stack   = [StackLayer("MoSe2"), StackLayer("WSe2")],
    ...     d_hbn_top    = 53,
    ...     d_hbn_bottom = 46,
    ...     label        = "hBN/MoSe2/WSe2/hBN",
    ... )

    **Trilayer with non-default thicknesses:**

    >>> geom = DeviceGeometry(
    ...     tmdc_stack   = [
    ...         StackLayer("WS2",   n_layers=2),
    ...         StackLayer("MoS2",  n_layers=1),
    ...     ],
    ...     d_hbn_top    = 30,
    ...     d_hbn_bottom = 40,
    ... )

    **No top hBN (single-gated device):**

    >>> geom = DeviceGeometry(
    ...     tmdc_stack   = [StackLayer("WSe2")],
    ...     d_hbn_top    = None,
    ...     d_hbn_bottom = 50,
    ... )
    """

    def __init__(
        self,
        tmdc_stack   : list,          # list[StackLayer]
        d_hbn_top    : float = None,
        d_hbn_bottom : float = None,
        eps_hbn      : float = EPS_HBN,
        label        : str   = None,
    ):
        if not tmdc_stack:
            raise ValueError("tmdc_stack must contain at least one StackLayer.")

        self.tmdc_stack   = list(tmdc_stack)
        self.d_hbn_top    = d_hbn_top
        self.d_hbn_bottom = d_hbn_bottom
        self.eps_hbn      = eps_hbn
        self.label        = label
        self.slabs = self._slabs()  # precompute for efficiency

    # --- Classmethod for backward compatibility ----------------------------

    @classmethod
    def from_single(
        cls,
        tmdc         : str,
        d_hbn_top    : float = None,
        d_hbn_bottom : float = None,
        n_layers     : int   = 1,
        d_monolayer  : float = None,
        eps_tmdc     : float = None,
        eps_hbn      : float = EPS_HBN,
        label        : str   = None,
    ) -> "DeviceGeometry":
        """
        Convenience constructor for a single-material TMDC stack.

        Mirrors the old ``DeviceGeometry(tmdc=..., layers=..., ...)`` interface
        so existing code requires only minimal changes:

            # old
            DeviceGeometry(d_hbn_top=53, d_hbn_bottom=46, tmdc="WS2", layers=2)
            # new (equivalent)
            DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46, n_layers=2)

        Parameters
        ----------
        tmdc : str
            Material name, e.g. ``"WS2"``.
        d_hbn_top, d_hbn_bottom : float or None
            hBN thicknesses in nm.
        n_layers : int
            Number of monolayers.
        d_monolayer : float, optional
            Monolayer thickness in nm.  Looked up from constants if ``None``.
        eps_tmdc : float, optional
            Dielectric constant.  Looked up from constants if ``None``.
        eps_hbn : float
            hBN dielectric constant.
        label : str, optional
            Stack description string.
        """
        layer = StackLayer(tmdc, n_layers=n_layers,
                           d_monolayer=d_monolayer, eps=eps_tmdc)
        return cls(
            tmdc_stack   = [layer],
            d_hbn_top    = d_hbn_top,
            d_hbn_bottom = d_hbn_bottom,
            eps_hbn      = eps_hbn,
            label        = label,
        )

    # --- Class method for generating DeviceGeometry from dict -------------
    @classmethod
    def from_dict(cls, device_dict: dict) -> "DeviceGeometry":
        """
        Build a DeviceGeometry from a Python dictionary.

        Parameters
        ----------
        device_dict : dict
            Dictionary describing the device stack, with the structure
            shown in Examples below.  ``"tmdc_stack"`` is required;
            ``"d_hbn_top"``, ``"d_hbn_bottom"``, ``"eps_hbn"``, and
            ``"label"`` are optional and fall back to the same defaults
            as :meth:`DeviceGeometry.__init__` when omitted.

        Returns
        -------
        DeviceGeometry
            The constructed device geometry.

        See Also
        --------
        to_dict : Generate a Python dictionary from the DeviceGeometry class.

        Examples
        --------
        >>> device_dict = {
        ...     "tmdc_stack": [
        ...         {"material": "WSe2", "n_layers": 1, "d_monolayer": 0.65, "eps": 7.0},
        ...     ],
        ...     "d_hbn_top": 12.0,      # nm; None if that side has no hBN
        ...     "d_hbn_bottom": 18.0,   # nm
        ...     "eps_hbn": 3.0,
        ...     "label": "hBN/1L-WSe2/hBN on SiO2/Si, back-gated",
        ... }
        >>> geom = DeviceGeometry.from_dict(device_dict)
        """
        layers = []
        for material in device_dict["tmdc_stack"]:
            layers.append(StackLayer(**material))

        return cls(
            tmdc_stack   = layers,
            d_hbn_top    = device_dict.get("d_hbn_top"),
            d_hbn_bottom = device_dict.get("d_hbn_bottom"),
            eps_hbn      = device_dict.get("eps_hbn", EPS_HBN),
            label        = device_dict.get("label"),
        )
    
    # --- Internal: build the ordered slab list ----------------------------

    def _slabs(self) -> list:
        """
        Return an ordered list of ``(thickness_nm, epsilon)`` tuples for
        every slab in the stack, including the hBN layers if present.
        """
        slabs = []
        if self.d_hbn_top is not None:
            slabs.append((self.d_hbn_top, self.eps_hbn))
        for layer in self.tmdc_stack:
            slabs.append((layer.thickness, layer.eps))
        if self.d_hbn_bottom is not None:
            slabs.append((self.d_hbn_bottom, self.eps_hbn))
        return slabs

    # --- Derived quantities ------------------------------------------------

    @property
    def eps_stack(self) -> float:
        r"""
        Effective out-of-plane dielectric constant of the **whole gate stack**
        (TMDC layers *and* both hBN layers), from the series-capacitor
        (thickness-weighted harmonic-mean) model:

        .. math :: \frac{d_{\text{total}}}{\epsilon_{\text{stack}}} = \sum_{i}\frac{d_{i}}{\epsilon_{i}}

        This accounts for every slab in the stack — top hBN, each TMDC layer,
        and bottom hBN — with their individual thicknesses and dielectric
        constants.

        The mean is harmonic rather than arithmetic because, with no free
        charge between the gates, the displacement field is continuous, so
        :math:`\epsilon_i E_i` is constant and it is the *voltage drops*
        :math:`d_i/\epsilon_i` that add in series.

        See Also
        --------
        eps_2d : the same harmonic mean over the TMDC layers only.
        """
        slabs = self._slabs()  # slabs is a list of (thickness_nm, epsilon)
        d_total = sum(d for d, _ in slabs)
        return d_total / sum(d / eps for d, eps in slabs)
    
    @property
    def eps_2d(self) -> float:
        r"""
        Effective out-of-plane dielectric constant of the TMDC layers
        computed with the series-capacitor (harmonic-mean) model:

        .. math :: \frac{d_{\text{2D}}}{\epsilon_{\text{2D}}} = \sum_{i}\frac{d_{i}}{\epsilon_{i}}

        This accounts for only TMDC layers in the stack with their individual thicknesses and dielectric
        constants.  For a stack of a single material it returns that material's
        :data:`~tmdc_optics_tools.constants.EPS_TMDC` value unchanged; it only
        does work for a genuine heterostructure.

        See Also
        --------
        eps_stack : the same harmonic mean over every slab, hBN included.
        """
        # Thickness-weighted harmonic mean over the TMDC layers only.
        return self.d_2d / sum(
            layer.thickness / layer.eps for layer in self.tmdc_stack
        )

    @property
    def d_2d(self) -> float:
        """
        Total physical thickness of the TMDC layers in nm, :math:`d_{2D}`,
        excluding hBN.

        See Also
        --------
        d_stack : the same total over every slab, hBN included.
        """
        return sum(layer.thickness for layer in self.tmdc_stack)

    @property
    def d_stack(self) -> float:
        """
        Total physical thickness of the whole gate stack in nm, :math:`d_{TOT}`
        — every TMDC layer plus both hBN layers.

        See Also
        --------
        d_2d : the TMDC layers alone.
        """
        return sum(d for d, _ in self.slabs)

    @property
    def stack_label(self) -> str:
        """
        Human-readable description of the stack, e.g. ``"hBN/MoSe2/WSe2/hBN"``.
        If a custom label was provided at initialization, it is returned
        instead.
        """
        if self.label:
            return self.label
        parts = []
        if self.d_hbn_top is not None:
            parts.append(f"hBN({self.d_hbn_top:.0f} nm)")
        for layer in self.tmdc_stack:
            parts.append(f"{layer.material}({layer.n_layers} ML)")
        if self.d_hbn_bottom is not None:
            parts.append(f"hBN({self.d_hbn_bottom:.0f} nm)")
        return " / ".join(parts)

    def to_dict(self) -> dict:
        """
        Generate a Python dictionary from the DeviceGeometry class.

        Returns
        -------
        dict
            Dictionary describing the device stack, with the structure
            shown in Examples below.  ``"label"`` is the raw ``label``
            attribute, ``None`` unless a custom label was given at
            construction — not the derived :attr:`stack_label`.

        See Also
        --------
        from_dict : Build a DeviceGeometry from a Python dictionary.

        Examples
        --------
        >>> geom.to_dict()
        {
            "tmdc_stack": [
                {"material": "WSe2", "n_layers": 1, "d_monolayer": 0.65, "eps": 7.0},
            ],
            "d_hbn_top": 12.0,      # nm; None if that side has no hBN
            "d_hbn_bottom": 18.0,   # nm
            "eps_hbn": 3.0,
            "label": "hBN/1L-WSe2/hBN on SiO2/Si, back-gated",  # or None
        }
        """

        def format_tmdc_dict(
                material : str,
                n_layers : int,
                d_monolayer : float,
                eps : float
            ) -> dict:

            return {"material" : material, "n_layers" : n_layers, "d_monolayer" : d_monolayer, "eps" : eps}

        d = {}
        stacks = []
        d["tmdc_stack"] = stacks
        # if self.d_hbn_top is not None:
        #     stacks.append(format_material_dict("hBN", np.nan, self.d_hbn_top, self.eps_hbn))
        for layer in self.tmdc_stack:
            stacks.append(format_tmdc_dict(layer.material, layer.n_layers, layer.d_monolayer, layer.eps))
        # if self.d_hbn_bottom is not None:
        #     stacks.append(format_material_dict("hBN", np.nan, self.d_hbn_bottom, self.eps_hbn))
        d["d_hbn_top"] = self.d_hbn_top if self.d_hbn_top is not None else None
        d["d_hbn_bottom"] = self.d_hbn_bottom if self.d_hbn_bottom is not None else None
        d["eps_hbn"] = self.eps_hbn
        d["label"] = self.label
        return d

    def electric_field(
        self, v_top: np.ndarray, v_bot: np.ndarray
    ) -> np.ndarray:
        r"""
        Electrostatic field inside the TMDC layers, in mV/nm, from gate voltages.

        This is the field that enters the quantum-confined Stark shift, so its
        accuracy sets the accuracy of any dipole length extracted from one.
        Exact within the series-capacitor model:

        .. math ::
            \epsilon_{2D} E_{2D} = \epsilon_{\text{stack}} E_{\text{stack}},
            \qquad E_{\text{stack}} = \frac{V_{BG} - V_{TG}}{d_{TOT}}

        With no free charge between the gates the displacement field is
        continuous, so :math:`\epsilon_i E_i` is the same in every slab.
        :attr:`eps_stack` is by construction the dielectric constant of the
        homogeneous slab of thickness :math:`d_{TOT}` with the same capacitance
        as the real stack, so :math:`E_{\text{stack}}` is that slab's uniform
        field and the identity above is D-continuity restated.

        Parameters
        ----------
        v_top : array-like
            Top gate voltages in V.
        v_bot : array-like
            Bottom gate voltages in V.

        Returns
        -------
        np.ndarray
            Electrostatic field in the TMDC in mV/nm.  Positive when
            ``v_bot > v_top``.

        Raises
        ------
        ValueError
            If neither hBN layer is set, i.e. the device has no gate dielectric.

        Notes
        -----
        **Do not "simplify"** :attr:`eps_stack` to ``eps_hbn`` here.  That form,

        .. math :: E_{2D} \approx \frac{V_{BG}-V_{TG}}{d_{TOT}}
                   \cdot \frac{\epsilon_{hBN}}{\epsilon_{2D}}

        is the thin-TMDC approximation used by the group's earlier MATLAB
        scripts and by this function before 2026-07-30.  It is low by
        :math:`(d_{2D}/d_{hBN})(1 - \epsilon_{hBN}/\epsilon_{2D})` — 0.59 % for
        53/46 nm hBN around a MoSe2/WSe2 bilayer, growing for thicker TMDC
        stacks or thinner hBN — so fields from this function are ~0.6 % higher
        than pre-2026-07-30 results.  Note also that repairing that form by
        substituting :attr:`eps_stack` for ``eps_hbn`` *in the denominator*
        rather than the numerator is wrong by a factor of ~1.8; the arrangement
        above is the one to keep.

        The sign convention depends on which physical gate each voltage came
        from, which is per-session wiring that no instrument file records.
        Mapping the source channels onto *v_top* and *v_bot* is the caller's
        responsibility; transposing them mirrors the field axis and flips the
        sign of any extracted dipole.  The loaders take that mapping as their
        ``gates`` argument and refuse to guess one.
        """
        if self.d_hbn_top is None and self.d_hbn_bottom is None:
            raise ValueError(
                "Cannot compute electric_field: both d_hbn_top and "
                "d_hbn_bottom are None. At least one hBN layer is required."
            )
        vdiff = np.asarray(v_bot) - np.asarray(v_top)
        # eps_2d * E_2d = eps_stack * E_stack, with E_stack = vdiff / d_TOT the
        # uniform field of the equivalent homogeneous slab. 1000 -> mV/nm.
        return 1000.0 * (vdiff / self.d_stack) * (self.eps_stack / self.eps_2d)

    def gate_capacitance(self, gate: str = "bottom") -> float:
        r"""
        Geometric capacitance per unit area between a gate and the TMDC, in F/m².

        .. math :: C = \epsilon_0 \epsilon_{hBN} / d_{hBN}

        The TMDC is the counter-electrode of this capacitor, not a slab inside it,
        so only that gate's hBN enters — neither the TMDC thickness nor
        :attr:`eps_stack` appears.  Purely geometric: the quantum capacitance of
        the TMDC and any interface-trap capacitance are in series with this and
        make the effective value smaller, so a density derived from it is an upper
        bound.

        Parameters
        ----------
        gate : {"bottom", "top"}
            Which gate. Default ``"bottom"``.

        Returns
        -------
        float
            Capacitance per unit area in F/m².

        Raises
        ------
        ValueError
            If *gate* is not a recognised gate, or if that gate's hBN thickness is
            ``None`` — there is then no dielectric to define a capacitance.

        See Also
        --------
        carrier_density : the same capacitance turned into a sheet density.
        """
        thickness = {"top": self.d_hbn_top, "bottom": self.d_hbn_bottom}
        if gate not in thickness:
            raise ValueError(
                f"gate must be one of {sorted(thickness)}, got {gate!r}."
            )
        d_nm = thickness[gate]
        if d_nm is None:
            raise ValueError(
                f"Cannot compute the {gate} gate capacitance: d_hbn_{gate} is "
                f"None, so that gate has no dielectric. Supply its thickness."
            )
        return EPS_0 * self.eps_hbn / (d_nm * 1e-9)

    def carrier_density(
        self, v_top: np.ndarray = None, v_bot: np.ndarray = None,
        v_ref: float = 0.0,
    ) -> np.ndarray:
        r"""
        Sheet carrier density induced by the gates, in cm⁻².

        .. math :: \Delta n = \frac{1}{e}\sum_i C_i (V_i - V_{ref})

        summed over whichever gates are supplied, since each gate injects charge
        through its own capacitance.  Signed: positive is electron accumulation
        for a positive gate voltage.

        **This is a density difference, not an absolute density.** *v_ref* is a
        gate voltage, not a threshold, so the result is the density induced
        relative to that gate voltage.  Absolute density needs the threshold at
        which the channel starts to populate, which comes from a transfer curve or
        the PL charging step and is in no instrument file — pass it as *v_ref* if
        you have measured it, and read the result as absolute.

        Parameters
        ----------
        v_top, v_bot : array-like, optional
            Gate voltages in V.  Supply the gates the device has; omitting one
            leaves it out of the sum rather than treating it as zero.
        v_ref : float
            Reference gate voltage in V, subtracted from every supplied gate.
            Default ``0.0``.

        Returns
        -------
        np.ndarray
            Sheet density in cm⁻², broadcast over the supplied voltages.

        Raises
        ------
        ValueError
            If no gate voltage is supplied, or if a supplied gate has no hBN
            thickness (see :meth:`gate_capacitance`).

        Notes
        -----
        Density and field are independent quantities only in a dual-gated device,
        where an anti-symmetric sweep tunes the field at fixed density and a
        symmetric one the reverse.  A single gate is one degree of freedom and
        moves both together.
        """
        supplied = {"top": v_top, "bottom": v_bot}
        supplied = {gate: v for gate, v in supplied.items() if v is not None}
        if not supplied:
            raise ValueError(
                "carrier_density needs at least one gate voltage — pass v_top, "
                "v_bot, or both."
            )
        # Each gate contributes C_i·(V_i − V_ref) of charge per unit area; summed
        # and divided by e this is a sheet number density in m^-2, then 1e-4 -> cm^-2.
        sigma = sum(self.gate_capacitance(gate) * (np.asarray(v, float) - v_ref)
                    for gate, v in supplied.items())
        return sigma / E_CHARGE * 1e-4

    # --- Dunder methods ----------------------------------------------------

    def __repr__(self) -> str:
        hbn_top_str = (f"{self.d_hbn_top} nm" if self.d_hbn_top is not None
                       else "None")
        hbn_bot_str = (f"{self.d_hbn_bottom} nm" if self.d_hbn_bottom is not None
                       else "None")
        stack_str   = " / ".join(repr(s) for s in self.tmdc_stack)
        label_str   = f"\n  Label         : {self.label}" if self.label else ""
        return (
            f"DeviceGeometry\n"
            f"  hBN top       : {hbn_top_str}\n"
            f"  TMDC stack    : {stack_str}\n"
            f"  hBN bottom    : {hbn_bot_str}\n"
            f"  ε_stack       : {self.eps_stack:.4f}\n"
            f"  ε_2D          : {self.eps_2d:.4f}\n"
            f"  d_stack       : {self.d_stack:.2f} nm\n"
            f"  d_2D          : {self.d_2d:.2f} nm"
            f"{label_str}"
        )


# ---------------------------------------------------------------------------
# Sweep-axis registry
# ---------------------------------------------------------------------------

# Recognised sweep types: key -> (curated attribute carrying the axis, axis
# label, unit).  A ``None`` source means "no physical axis recorded", so the
# sweep index is used.  Anything *not* in this table is looked up as a raw CSV
# row label instead, which is why an unforeseen swept quantity ("Galvo_Y",
# "T", …) needs no change here to be usable as a sweep axis.
#
# A ``None`` *unit* means "read it from the curated registry", which is where a
# ``curated_units`` override lands.  Only the three axes with no curated row
# behind them spell a unit here: the sweep index, and the two quantities derived
# from gate voltages rather than promoted from one.  Keeping a literal beside a
# curated-backed axis is what let a rescaled row keep its old label.
_SWEEP_TYPES = {
    "index"          : (None,        "Sweep index",        ""),
    "electric_field" : ("ef",        r"$E_F$",             "mV/nm"),
    "carrier_density": ("carrier_density", r"$\Delta n$",  r"cm$^{-2}$"),
    "top_voltage"    : ("v_top",     r"$V_\mathrm{top}$",  None),
    "bottom_voltage" : ("v_bot",     r"$V_\mathrm{bot}$",  None),
    "power"          : ("power",     "Power",              None),
    # The scanners are piezos and the rows carry their drive voltage, so these
    # two axes are in V.  Converting to a distance needs a per-stage µm/V
    # calibration the file does not contain; pass one via ``curated_scales``,
    # and say what the numbers then are via ``curated_units``.
    "piezo_x"        : ("scanner_x", r"Piezo $x$",         None),
    "piezo_y"        : ("scanner_y", r"Piezo $y$",         None),
}

# Which curated rows each sweep type depends on.  Checked at load time so a
# declared sweep fails immediately, with the available labels listed, rather
# than at the first plot or fit.
_SWEEP_REQUIRES = {
    "electric_field" : ("v_top", "v_bot"),
    "top_voltage"    : ("v_top",),
    "bottom_voltage" : ("v_bot",),
    "power"          : ("power",),
    "piezo_x"        : ("scanner_x",),
    "piezo_y"        : ("scanner_y",),
}

# Which electrode each gate-backed curated entry belongs to.  Which acquisition
# channel reached which electrode is per-session wiring that no export records, so
# these are declared through ``gates=`` and nowhere else; any sweep type requiring
# one of them therefore needs that declaration too, which _resolve_sweep derives
# from this table rather than relisting.
_GATE_ROLE_CURATED = {"top": "v_top", "bottom": "v_bot"}
_GATE_CURATED      = tuple(_GATE_ROLE_CURATED.values())
_ROLE_FOR_CURATED  = {attr: role for role, attr in _GATE_ROLE_CURATED.items()}

# Electrodes that gate the TMDC across a dielectric.  A potential *difference*
# between the two is what defines a displacement field, so a field needs both.
_GATE_ELECTRODES = tuple(_GATE_ROLE_CURATED)

# Every role a ``gates`` mapping may name.  ``"channel"`` is a contact to the TMDC
# itself — the ground reference of a doping measurement — not a gate: it sits
# inside the stack rather than across a dielectric from it, so it carries no
# thickness and enters no field.  Naming it records that the device is contacted
# and is what makes a single-gate declaration unambiguous.
_GATE_ROLES = _GATE_ELECTRODES + ("channel",)

# Voltage row -> the current row measured at the same terminal.  A source-meter
# channel applies a bias and reports the current it sources, so both rows describe
# one electrode and a declaration naming either names both.  This is a property of
# the export, fixed across every file seen, whereas which electrode that channel
# reached is per-session wiring — so it belongs here rather than in ``gates``.
_CHANNEL_SIBLING_CURRENT = {"V_A": "I_A", "V_B": "I_B"}

# Which curated entry carries each role's current.  Covers all three roles, unlike
# the voltage map above: a current flows at the channel contact just as it does at a
# gate, and only the *field* is restricted to the two gate electrodes.
_ROLE_CURRENT_CURATED = {"top": "i_top", "bottom": "i_bot", "channel": "i_channel"}
_CURRENT_CURATED      = tuple(_ROLE_CURRENT_CURATED.values())

# Curated entries whose label comes from ``gates`` and nowhere else.  Rejected as
# ``curated_labels`` keys, and dropped from an HDF5 file's curated dump on read, so
# that a written label can never be mistaken for a declared mapping.
_DECLARED_CURATED = _GATE_CURATED + _CURRENT_CURATED


# ---------------------------------------------------------------------------
# Spectra-source registry
# ---------------------------------------------------------------------------

# Correction state → the array holding it, as (wavelength axis, energy axis).  A
# source names the state and x_axis picks the column, so every state is reachable
# on either axis and a source cannot be served on the wrong one.  Contrast is
# opt-in by name: "best" never returns it, ΔR/R₀ being a different physical
# quantity from the counts rather than a better-corrected version of them.
_SPECTRA_SOURCES = {
    "raw"      : ("spectra",    "energy_spectra"),
    "cr"       : ("spectra_cr", "energy_spectra_cr"),
    "bg"       : ("spectra_bg", "energy_spectra_bg"),
    "contrast" : ("contrast",   "energy_contrast"),
}

# Served on the energy axis whatever x_axis says.  The Jacobian is a property of
# the energy representation rather than a state of the counts, so it is not a
# column of the table above — the wavelength arrays *are* the pre-Jacobian values.
_SPECTRA_SOURCES_ENERGY_ONLY = {"pre_jacobian": "energy_spectra_pre_jacobian"}

# "best" asks for the most-corrected rung the object has, which each class answers
# through its own best_* pair, so it is a request rather than a row of the table.
_SPECTRA_SOURCE_BEST = "best"

# Which argument would make an absent source available — one message per source,
# each naming the argument that fixes it rather than guessing between two.
_SOURCE_REQUIRES = {
    "cr"       : "cosmic_rays=",
    "bg"       : "bg_region_nm=, bg_region_eV= or bg_spectrum=",
    "contrast" : "a reference= spectrum",
}


def _spectra_source_names() -> list:
    """Every accepted ``spectra_source``, for a message."""
    return ([_SPECTRA_SOURCE_BEST] + list(_SPECTRA_SOURCES)
            + list(_SPECTRA_SOURCES_ENERGY_ONLY))


def _resolve_spectra(scan, spectra_source: str, x_axis: str) -> np.ndarray:
    """
    Return the ``(n_pixels, n_sweeps)`` array for *spectra_source*.

    A source names a **correction state** — ``"raw"``, ``"cr"``, ``"bg"``,
    ``"contrast"`` — and *x_axis* picks the axis it is served on, so the two
    cannot disagree.  ``"best"`` asks for the most-corrected state the object
    holds.  ``"pre_jacobian"`` is the one exception: the Jacobian belongs to the
    energy representation rather than to the counts, so it exists on that axis
    only.

    Reads *scan* by attribute name, so it serves anything mirroring
    :class:`AttoCubeSpectralSweep`, and distinguishes a correction the class does
    not offer from one that was simply not requested.

    Raises ``ValueError`` when *x_axis* names no spectral axis, when the source is
    unrecognised, when the class has no such correction (a
    :class:`SingleSpectrum` has no cosmic-ray repair), when the correction exists
    but was not requested at load time, or when ``"pre_jacobian"`` is asked for on
    the wavelength axis.
    """
    # Ahead of the source lookup: the axis chooses the column below, so an
    # unrecognised one would be read as wavelength.
    _x_axis_name_unit(x_axis)

    src = spectra_source.lower()

    if src == _SPECTRA_SOURCE_BEST:
        # Each class decides what its own "best" is, so this serves a
        # SingleSpectrum's background-corrected array and a sweep's most-corrected
        # rung through one expression.
        arr = (scan.best_energy_spectra if x_axis == "energy"
               else scan.best_spectra)
        return np.asarray(arr, dtype=float)

    if src in _SPECTRA_SOURCES_ENERGY_ONLY:
        if x_axis != "energy":
            raise ValueError(
                f"spectra_source={src!r} exists on the energy axis only — the "
                f"Jacobian is a property of that representation, and the "
                f"wavelength-space arrays already hold the values it is applied "
                f"to. Ask for 'raw', 'cr' or 'bg' on the wavelength axis."
            )
        attr = _SPECTRA_SOURCES_ENERGY_ONLY[src]
    elif src in _SPECTRA_SOURCES:
        attr = _SPECTRA_SOURCES[src][0 if x_axis == "wavelength" else 1]
    else:
        raise ValueError(
            f"spectra_source {src!r} is not recognised. "
            f"Choose from: {_spectra_source_names()}."
        )

    # Two different failures, and the advice differs: a class that never offers
    # this correction cannot be fixed by passing an argument at load time.
    if not hasattr(scan, attr):
        raise ValueError(
            f"a {type(scan).__name__} has no {src!r} spectra ({attr!r}), so that "
            f"correction cannot be asked of it. Available: "
            f"{_spectra_source_names()}."
        )
    arr = getattr(scan, attr)
    if arr is None:
        raise ValueError(
            f"spectra_source={src!r} is not available on this scan. Check that "
            f"{_SOURCE_REQUIRES[src]} was set at load time."
        )

    return np.asarray(arr, dtype=float)


# Mapping of string names → the accessor that reads a real-space frame.  The
# frame counterpart of _SPECTRA_SOURCES, and `None` is the same sentinel: "best"
# is resolved at call time because which array it names depends on how the scan
# was loaded.
_FRAME_SOURCES = {
    "best" : None,             # bg-subtracted where available, else raw
    "raw"  : "load_frame",
    "bg"   : "load_frame_bg",
}

# Statistics _apply_bg_region can estimate a pedestal with.  Kept next to the
# sources because the two together are the whole frame-correction vocabulary.
_BG_STATS = ("median", "mean")


def _resolve_frame(scan, idx: int, frame_source: str = "best") -> np.ndarray:
    """
    Return frame *idx* of *scan* as the array *frame_source* names.

    Reads *scan* by accessor name, so it serves anything exposing
    ``load_frame(idx)`` — a scan that offers no background-corrected frames is
    not an error under ``"best"``, it simply has one fewer array to choose from.

    Parameters
    ----------
    scan : AttoCubePLScanRealSpace or object exposing ``load_frame(idx)``
    idx : int
        Frame index.
    frame_source : {``"best"``, ``"raw"``, ``"bg"``}
        ``"raw"`` is the file's own counts; ``"bg"`` is background-subtracted;
        ``"best"`` gives ``"bg"`` when a *bg_region* was set at load time and
        ``"raw"`` otherwise.

    Returns
    -------
    np.ndarray
        2-D frame, ``float``.

    Raises
    ------
    ValueError
        If *frame_source* is not recognised, or if ``"bg"`` is asked for on a
        scan carrying no *bg_region*.
    """
    src = frame_source.lower()
    if src not in _FRAME_SOURCES:
        raise ValueError(
            f"frame_source {src!r} is not recognised. "
            f"Choose from: {list(_FRAME_SOURCES)}."
        )

    # "best" prefers the corrected frame; asking for it by name makes it required.
    attr   = _FRAME_SOURCES["bg"] if src == "best" else _FRAME_SOURCES[src]
    getter = getattr(scan, attr, None)
    if src != "raw" and (getter is None or getattr(scan, "bg_region", None) is None):
        if src == "bg":
            raise ValueError(
                f"frame_source={src!r} is not available on this scan.  "
                f"Check that bg_region was set at load time."
            )
        getter = getattr(scan, _FRAME_SOURCES["raw"])

    return np.asarray(getter(idx), dtype=float)


# Recognised input formats, dispatched on the file suffix.
_CSV_SUFFIXES  = (".csv",)
_HDF5_SUFFIXES = (".h5", ".hdf5", ".he5")


# ---------------------------------------------------------------------------
# Export block layout
# ---------------------------------------------------------------------------

# The AttoCube export writes one column *block* per sweep point.  Two block
# shapes exist, told apart by the header's field names:
#
#   spectral  [Par_i, Wavelength{i}, ExpROI1_{i}, ExpROI2_{i}]   PL, R, RC …
#   temporal  [Par_i, Wavelength{i}, Exp_{i}]                    TRPL
#
# In the temporal layout the column *named* "Wavelength" actually holds **time**
# — an acquisition-software misnomer, not something to correct on read.
#
# The names are the only honest source for the layout: the block count cannot be
# recovered from the column count, because the exporter over-allocates the header
# and then fills the surplus blocks with zeros (see _drop_unwritten_blocks).
_BLOCK_LAYOUTS = {
    ("Par", "Wavelength", "ExpROI1", "ExpROI2"): "spectral",
    ("Par", "Wavelength", "Exp")               : "temporal",
}

# Which loader reads which layout.  Names rather than classes, since this table
# sits above their definitions; used only to point a mis-aimed load at the right
# class by name.
_CLASS_FOR_KIND = {
    "spectral": "AttoCubeSpectralSweep",
    "temporal": "AttoCubeTRPLSweep",
}

# Trailing digits, with or without a separating underscore: "Par_0" -> "Par",
# "Wavelength12" -> "Wavelength".  "ExpROI1_0" -> "ExpROI1" — the ROI number is
# kept because the underscore-and-index suffix is what gets stripped.
_BLOCK_FIELD_INDEX = re.compile(r"_?\d+$")

# A block's first column, e.g. "Par_0".  Matching the *indexed* form matters: the
# header's own leading label column is "Parameters Labels", which a bare
# startswith("Par") would mistake for a block start.
_BLOCK_START = re.compile(r"^Par_?\d+$")


def _n_rows_upto(fh, limit: int) -> int:
    """
    Count the non-blank lines remaining in *fh*, reading no further than *limit*.

    Bounded so classifying a file costs one short read however large it is.  The
    result is therefore **not** the file's row count and must not be reported as
    one — only the comparison against *limit* is meaningful.

    Parameters
    ----------
    fh : file object
        Open text handle, read from its current position.
    limit : int
        Stop after this many lines, blank ones included.

    Returns
    -------
    int
        Non-blank lines seen, at most *limit*.
    """
    return sum(1 for _, line in zip(range(limit), fh) if line.strip())


def _read_block_layout(path) -> dict:
    """
    Determine an export's block layout from its **header line alone**.

    Reading one line keeps this cheap on the large exports (a real reflectance
    raster is 314 MB), and makes it usable as a file *classifier* — which is how
    a directory of TRPL files is sorted into decays and their metadata companion
    without parsing any of them.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
        ``kind`` : ``"spectral"`` or ``"temporal"``.
        ``block_width`` : columns per sweep point.
        ``n_blocks`` : sweep points *declared* by the header — not necessarily
            the number actually written; see :func:`_drop_unwritten_blocks`.
        ``roles`` : the de-indexed field names of one block, in order.

    Raises
    ------
    ValueError
        If the file has no ``Par`` columns (not an AttoCube spectral export at
        all — a real-space image CSV, for instance), or if the block's field
        names match no known layout.
    """
    with open(path, "r") as fh:
        header = fh.readline() # Reads the header
        # Only whether a third row exists matters, so two lines past the header
        # settle it.
        two_rows_only = _n_rows_upto(fh, 2) < 2
    names = [name.strip() for name in header.split(",")] # Splits into the column names

    # Every block starts with a Par column, so the Par positions give both the
    # block count and the stride — no arithmetic on the total column count.
    par_at = [i for i, name in enumerate(names) if _BLOCK_START.match(name)]
    if not par_at:
        # No header at all: the first line is already data.  Both other CSV kinds
        # look like this, and only the row count separates them, so name the class
        # that fits rather than guessing at one.
        if two_rows_only:
            shape, better = (
                "two rows",
                "SingleSpectrum, which reads exactly this shape "
                "(row 0 wavelength in nm, row 1 counts)",
            )
        else:
            shape, better = (
                "more than two rows",
                "AttoCubePLScanRealSpace, which reads a directory of these as an "
                "image sequence, or SingleImage for one frame",
            )
        raise ValueError(
            f"'{path}' has no 'Par' columns in its header, so it is not an "
            f"AttoCube spectral export — it is a bare numeric grid of {shape}. "
            f"Load it with {better}."
        )

    block_width = (par_at[1] - par_at[0]) if len(par_at) > 1 else (
        # A single-block file: the block runs to the last non-empty field -> gets number of columns after the first block start match
        sum(1 for name in names[par_at[0]:] if name) )

    # Get the column header format for matching with known types
    roles = tuple(
        # Iterates over the entries in one block. .sub() removes the underscore (if any) + numeric suffix from the column name.
        _BLOCK_FIELD_INDEX.sub("", name) 
        for name in names[par_at[0]:par_at[0] + block_width] 
    )
    if roles not in _BLOCK_LAYOUTS:
        raise ValueError(
            f"'{path}' has an unrecognised block layout {roles}. Known layouts:\n"
            + "\n".join(f"  {kind:<8} {fields}"
                        for fields, kind in _BLOCK_LAYOUTS.items())
        )

    return {
        "kind"        : _BLOCK_LAYOUTS[roles],
        "block_width" : block_width,
        "n_blocks"    : len(par_at),
        "roles"       : roles,
    }


def _drop_unwritten_blocks(axis_col: np.ndarray, path) -> tuple:
    """
    Strip blocks the exporter declared and zero-filled but never wrote.

    The acquisition software over-allocates: a 2091-point reflectance raster is
    exported with 4182 declared blocks, the surplus half filled with literal
    ``0.0`` for *every* field.  Those columns are numeric, not empty, so the
    all-NaN padding strip does not touch them — and keeping them fabricates
    thousands of measurements that were never taken.  Removing them is decoding,
    not a correction: there is no data there to preserve.

    The sentinel is the **axis column being identically zero**.  Sound for both
    layouts: a spectrometer axis never contains zeros, and a time axis has only
    its first bin at zero, never the whole column.

    Parameters
    ----------
    axis_col : np.ndarray, shape (n_axis, n_blocks)
        The Wavelength column of every block.
    path : str or Path
        Used in messages only.

    Returns
    -------
    keep : np.ndarray of bool, shape (n_blocks,)
        Blocks to retain. Nothing is modified here; the caller applies this
        mask to its own arrays.
    n_declared : int
    axis_block : int
        Index of the first block holding a real axis.  Returned separately from
        *keep* because in the interleaved case every block is kept, and block 0
        may then be a zero-filled one — taking the axis from it would give an
        all-zero wavelength axis and infinite energies.

    Warns
    ------
    UserWarning
        If the unwritten blocks are *interleaved* rather than strictly trailing.
        Nothing is dropped in that case: interleaving would mean this model of
        the export format is wrong, and guessing which columns to discard could
        silently misalign every sweep point against its parameters.
    """
    n_declared = axis_col.shape[1]
    # A block counts as written if its axis column holds any real non-zero value.
    # The finiteness test is not decoration: exports carry a trailing NaN row, and
    # `NaN != 0` is True, so a bare `!= 0` marks every block as written.
    written = np.any(np.isfinite(axis_col) & (axis_col != 0), axis=0)  # (n_blocks,). Finds truly written blocks by checking if any value in the axis column is finite and non-zero.
    axis_block = int(np.flatnonzero(written)[0]) if written.any() else 0

    if written.all(): # Simple case: all written columns are real columns. Keep all.
        return written, n_declared, axis_block

    # Strictly trailing means: once written stops, it never resumes.
    n_written = int(written.sum())
    if not written[:n_written].all(): # Finds the first n_written blocks, checks if all of them are true.
        # If condition not true: something is strange about how the file is written
        warnings.warn(
            f"'{path}' has {n_declared - n_written} zero-filled block(s) "
            f"*interleaved* with real ones (first unwritten at index "
            f"{int(np.flatnonzero(~written)[0])}, but written blocks continue "
            f"after it). Expected any unwritten blocks to be trailing, so this "
            f"file does not match the known export layout. Keeping all "
            f"{n_declared} blocks rather than guessing which to drop — check "
            f"the file before trusting the sweep axis.",
            UserWarning, stacklevel=3,
        )
        return np.ones(n_declared, dtype=bool), n_declared, axis_block

    # If yes, then the real blocks are all at the front and the unwritten ones are strictly trailing.
    return written, n_declared, axis_block


# ---------------------------------------------------------------------------
# Per-point file ordering
# ---------------------------------------------------------------------------

# The per-point index in a directory export's filenames, e.g. "..._iter_12.csv".
# Ordering must be numeric on this: plain lexicographic sorting puts iter_10
# before iter_2, which would silently pair every point with the wrong parameters.
# Exports are usually zero-padded, but the width varies between them, so the
# padding is not something to rely on.
_ITER_INDEX = re.compile(r"_iter_(\d+)$", re.IGNORECASE)


def _order_by_iter(files: list, path, *, stacklevel: int) -> list:
    """
    Sort per-point export files by the integer in their ``_iter_N`` suffix.

    Lexicographic order places ``iter_10`` before ``iter_2``, pairing every point
    with the wrong index.  A gap in the sequence, and an index claimed by more than
    one file, are both reported rather than repaired: a missing point means an
    aborted or partly copied acquisition, a repeated one usually means two
    acquisitions share a directory, and closing up or discarding either would
    misalign the axis silently.

    Parameters
    ----------
    files : list of Path
        Per-point files, in any order.
    path : str or Path
        Directory they came from.  Used in messages only.
    stacklevel : int
        Stack level for the warnings below, counted from inside this function, so
        ``2`` attributes them to the immediate caller.  A caller reached through
        several private layers passes the depth that lands on its own entry point.

    Returns
    -------
    list of Path
        The same files, ordered by iteration index.  Nothing is dropped, and
        neither a gap nor a repeat is repaired, so index *i* is not necessarily
        iteration *i*.

    Warns
    -----
    UserWarning
        If any file carries no ``_iter_N`` suffix.  Filename order is returned
        instead, which need not be acquisition order.
    UserWarning
        If more than one file claims the same index.  Both are kept, so the count
        exceeds the number of distinct points and every point from the collision
        onward is offset against a per-point variable array.
    UserWarning
        If iterations are missing between the lowest and highest present.
    """
    indexed = []
    for f in files:
        m = _ITER_INDEX.search(f.stem)
        indexed.append((int(m.group(1)) if m else None, f))

    if any(idx is None for idx, _ in indexed):
        warnings.warn(
            f"Some files in '{path}' carry no '_iter_N' suffix "
            f"({', '.join(f.name for idx, f in indexed if idx is None)}); "
            f"falling back to filename order, which may not be acquisition "
            f"order.",
            UserWarning, stacklevel=stacklevel,
        )
        return [f for _, f in indexed]

    indexed.sort(key=lambda item: item[0])

    # Grouped by index so the message can name the colliding files, which is what
    # says whether two acquisitions were merged. The gap check below cannot see a
    # repeat: it compares against the *set* of indices, which a repeat leaves
    # unchanged.
    by_index = {}
    for idx, f in indexed:
        by_index.setdefault(idx, []).append(f)
    collisions = {idx: fs for idx, fs in by_index.items() if len(fs) > 1}
    if collisions:
        detail = "; ".join(f"iter_{idx}: {', '.join(f.name for f in fs)}"
                           for idx, fs in sorted(collisions.items()))
        warnings.warn(
            f"'{path}' has {len(collisions)} iteration index(es) claimed by more "
            f"than one file ({detail}). Index i is therefore not iteration i, and a "
            f"per-point variable array will be mispaired from the first collision "
            f"onward. Nothing is dropped — narrow the prefix if two acquisitions "
            f"share a directory.",
            UserWarning, stacklevel=stacklevel,
        )

    seen    = [idx for idx, _ in indexed]
    missing = sorted(set(range(seen[0], seen[-1] + 1)) - set(seen))
    if missing:
        warnings.warn(
            f"'{path}' is missing iteration(s) {missing} between iter_"
            f"{seen[0]} and iter_{seen[-1]} — an aborted or partly copied "
            f"export. The {len(seen)} file(s) present are loaded in order, so "
            f"index i is not iteration i.",
            UserWarning, stacklevel=stacklevel,
        )
    return [f for _, f in indexed]


# ---------------------------------------------------------------------------
# _Sweep — shared machinery
# ---------------------------------------------------------------------------

class SweepGrid(NamedTuple):
    """
    A 2-D raster detected in the parameter rows: its fast axis and its slow one.

    What :meth:`_Sweep.sweep_grid` reports.  Names the same two axes a
    nest is declared with, so a detection reads directly as the ``fast_sweep=`` /
    ``slow_sweep=`` to pass — but it is a row-level guess, and a nest whose axis
    is a derived quantity will be reported through the rows that carry it.
    """
    fast_label : str
    n_fast     : int
    slow_label : str
    n_slow     : int

    def __str__(self) -> str:
        return (f"{self.fast_label} ({self.n_fast}) × "
                f"{self.slow_label} ({self.n_slow}) = "
                f"{self.n_fast * self.n_slow}")


# Fraction by which two readings of a *driven* axis may differ and still count as the
# same value, and — the same number used the other way round — the fraction of its own
# magnitude a row must travel before it counts as driven at all.  Relative rather than
# absolute so it carries across volts, microwatts and kelvin without a table.
#
# This describes and matches the *sweep axis*; it does not resolve a nest.  One
# tolerance for a whole axis has to sit above the scatter within a level and below
# the step between levels, and on a measured axis no such value need exist: a power
# meter's scatter grows with the reading while the smallest step it must resolve
# does not.  ``_levels_separate`` compares each level against its neighbour instead.
_AXIS_RTOL = 1e-3

# Fraction of its own travel by which consecutive readings may step and still be a
# sweep progressing through its range, rather than scatter jumping about inside one.
# A driven axis of n points steps about 1/(n-1) of its travel each time — 10% at 11
# points, 0.5% at 201 — whereas scatter sits near 40% however many readings there are.
_AXIS_STEP_FRAC = 0.1

# Non-zero steps a row must take before the *direction* of those steps is evidence of
# anything.  Below this, scatter runs one way often enough by chance to be worth
# nothing: on a held 5 V gate, 9% of four-point rows never reverse, against 2% at five
# readings and 0.2% at six.
_AXIS_MIN_MOVES = 4


def _axis_driven(values: np.ndarray, rtol: float = _AXIS_RTOL) -> bool:
    """
    Whether *values* record a driven axis rather than one setting plus scatter.

    Three independent signs, any one of which is enough, because each is blind where
    another sees:

    ==========================  ==============================  ======================
    sign                        draws on                        blind to
    ==========================  ==============================  ======================
    travel against size         how far it went for how large   a fine sweep on a
                                it is                           large offset — 300.0
                                                                to 300.2 K travels
                                                                0.07% of its readings
    how far it steps            a sweep progresses where        a coarse sweep, whose
                                scatter jumps about             few points step as far
                                                                as scatter would
    which way it steps          a sweep does not turn round;    a sweep that revisits
                                scatter does                    values, which restarts
                                                                every row of a nest
    ==========================  ==============================  ======================

    Requiring all three to fail before calling an axis undriven is what keeps a real
    sweep from being collapsed: the safe error here is leaving scatter uncollapsed,
    which is what happens without any of them.  The third sign is what covers the two
    blind spots that *overlap* — a sweep both coarse and narrow for its offset, such as
    six points from 300.0 K to 300.2 K, defeats the first two together.

    That safety sets the reach.  Because the first sign fires as soon as a row's span
    exceeds *rtol* of its magnitude, a held setting is recognised only while its
    read-back is stable to better than that — reliably so for a commanded row (a
    source-meter reports a 5 V gate to ~20 µV) and not at all for a power meter, whose
    scatter runs to a few parts per thousand of the reading.  A noisy instrument holding
    one setting therefore still reads as many, exactly as it did before.  Read-back that
    *drifts* rather than jitters is seen by the last two signs at any amplitude, so a
    held row is recognised only while its scatter is directionless.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return False
    travel = float(np.ptp(finite))
    if travel == 0.0:
        return False        # every reading identical: one setting, nothing to judge

    # RMS rather than |mean|, because a row can sit astride zero — an anti-symmetric
    # gate pair sweeping the field does — and its mean is then no measure of how large
    # it is.  Floored so an identically zero row cannot divide by nothing.
    size = max(float(np.sqrt(np.mean(finite ** 2))), np.finfo(float).tiny)
    if travel > rtol * size:
        return True

    # (n_finite - 1,) consecutive differences in *acquisition order*, which is what
    # scatter destroys and a sweep preserves.  Median rather than mean, so the jump
    # back at the end of each row of a flattened nest is outvoted rather than read as
    # scatter.
    if float(np.median(np.abs(np.diff(finite)))) < _AXIS_STEP_FRAC * travel:
        return True

    # The same differences with the exact repeats dropped, leaving (n_moves,) actual
    # moves: a slow axis sits still between its steps, and standing still is not
    # turning round.  Judged on direction alone, so this sign carries no magnitude and
    # no point count — which is why it still sees a sweep too coarse for the second
    # sign and too narrow for the first.
    steps = np.diff(finite)
    steps = steps[steps != 0.0]
    if steps.size < _AXIS_MIN_MOVES:
        return False
    return bool(np.all(steps > 0) or np.all(steps < 0))


def _axis_atol(values: np.ndarray, rtol: float = _AXIS_RTOL) -> float:
    """
    How far apart two readings may be and still count as the same setting.

    A driven axis is judged on a fraction of its travel, which is a fair proxy for how
    finely it steps.  An undriven one is judged on the whole of it: there the travel
    *is* the instrument's scatter, so everything inside it is the one setting that was
    set.  Taking a fraction of the travel in that case is circular — the tolerance
    shrinks with the scatter it exists to absorb, so it never absorbs it, and a quieter
    instrument does not help.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.finfo(float).tiny
    travel = float(np.ptp(finite))
    if _axis_driven(values, rtol):
        return max(rtol * travel, np.finfo(float).tiny)
    return max(travel, np.finfo(float).tiny)


def _level_labels(values: np.ndarray, atol: float) -> tuple:
    """
    Index the grid level each value sits on, and count the levels.

    Two readings share a level when the gap between them, in sorted order, is at
    most *atol* — so a level is a run of readings each within *atol* of its
    neighbour, however far the ends of the run are from each other.  Non-finite
    entries sit on no level and are labelled ``-1``.

    Returns ``(labels, n_levels)``, *labels* shaped like *values*.
    """
    labels = np.full(values.shape, -1, dtype=int)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return labels, 0

    # Sort the finite entries, cut the run wherever a consecutive gap exceeds
    # atol, and scatter the running cut count back to where each value came
    # from: one vectorised pass rather than a pairwise comparison.
    order         = finite[np.argsort(values[finite], kind="stable")]
    cuts          = np.diff(values[order]) > atol         # (n_finite - 1,)
    labels[order] = np.concatenate(([0], np.cumsum(cuts)))
    return labels, int(labels[order[-1]]) + 1


def _count_distinct(values: np.ndarray, atol: float) -> int:
    """
    Number of distinct values in *values*, treating gaps ≤ *atol* as equal.

    Non-finite entries are not counted.
    """
    return _level_labels(values, atol)[1]


def _render_indices(indices: np.ndarray, limit: int = 6) -> str:
    """Render matched sweep positions for a message, capped so it stays readable."""
    shown = ", ".join(str(i) for i in indices[:limit])
    if indices.size > limit:
        return f"{indices.size} sweep points (indices {shown}, …)"
    return f"{indices.size} sweep points (indices {shown})"


# Which way a grid is collapsed to recover one axis's levels.  A slow axis holds
# still while the fast axis runs to completion, so its levels are the grid's *rows*;
# a fast axis repeats the same run of settings in every row, so its levels are the
# grid's *columns*.
_SLOW_LEVELS_AXIS = 1
_FAST_LEVELS_AXIS = 0


def _level_bounds(grid: np.ndarray, axis: int) -> tuple:
    """
    Lowest and highest reading of each grid level, ordered by value.

    *grid* is one axis's readings reshaped to ``(n_slow, n_fast)``, and *axis* is
    :data:`_SLOW_LEVELS_AXIS` or :data:`_FAST_LEVELS_AXIS`.

    Returns ``(lo, hi)``, both ``(n_levels,)``, sorted by *lo*.  A level holding a
    non-finite reading gets a non-finite bound, which no comparison below accepts,
    so such an axis is refused rather than grouped on the finite readings alone.
    """
    lo, hi = grid.min(axis), grid.max(axis)          # (n_levels,) each
    order  = np.argsort(lo, kind="stable")           # levels in value order
    return lo[order], hi[order]


def _levels_separate(grid: np.ndarray, axis: int) -> bool:
    """
    True when the readings of different grid levels never interleave.

    Every level must begin above where the level below it ended.  This is what makes
    a reshape trustworthy without a tolerance: the only comparison made is between
    one level's own scatter and the step to its immediate neighbour, at the same
    place on the axis.  A single tolerance for the whole axis cannot do that, which
    is why a log-spaced power series read back by a meter defeats one — the scatter
    at the top of the axis exceeds the step at the bottom.

    Separable is all this checks, not tight.  A level whose readings are spread far
    wider than expected still passes when that spread happens to fall in a gap.
    Testing tightness needs a width to compare against, and the only scale-free
    candidate — demanding the neighbour gap exceed the level's own spread — refuses
    genuine log-spaced power sweeps.
    """
    lo, hi = _level_bounds(grid, axis)
    # (n_levels - 1,): clearance from each level's top to the next level's floor.
    return bool(np.all(lo[1:] - hi[:-1] > 0))


def _nest_candidates(fast: np.ndarray, slow: np.ndarray, n_sweeps: int) -> list:
    """
    Every ``(n_fast, n_slow)`` under which *fast* nests inside *slow*.

    Both arrays are the flattened ``(n_sweeps,)`` readings of the two declared axes.
    The exporter writes the fast axis running fastest, so each slow step gets one
    complete run of the fast axis and ``n_fast * n_slow == n_sweeps`` exactly: only
    the divisors of *n_sweeps* can be shapes, and both counts must reach 2 for there
    to be a nest at all.  A shape survives when both axes' levels are separable
    under it.

    Normally exactly one does.  Returning all of them lets the caller refuse an
    ambiguous file by naming the shapes rather than choosing one of them.
    """
    # Trial division to the square root, pairing each divisor with its cofactor, so
    # a long sweep does not walk every integer below n_sweeps.
    divisors = set()
    for d in range(1, int(np.sqrt(n_sweeps)) + 1):
        if n_sweeps % d == 0:
            divisors.update((d, n_sweeps // d))

    hits = []
    for n_fast in sorted(divisors):
        n_slow = n_sweeps // n_fast
        if n_fast < 2 or n_slow < 2:
            continue
        if (_levels_separate(fast.reshape(n_slow, n_fast), _FAST_LEVELS_AXIS)
                and _levels_separate(slow.reshape(n_slow, n_fast), _SLOW_LEVELS_AXIS)):
            hits.append((n_fast, n_slow))
    return hits


def _nest_shape(fast: np.ndarray, slow: np.ndarray, n_sweeps: int) -> tuple:
    """
    Return the one ``(n_fast, n_slow)`` describing *fast* nested inside *slow*.

    ``None`` when no shape fits, and also when more than one does — a file that
    reshapes two ways is not one to pick between.  Callers that need to tell those
    apart, or to report them, use :func:`_nest_candidates` directly.
    """
    candidates = _nest_candidates(fast, slow, n_sweeps)
    return candidates[0] if len(candidates) == 1 else None


def _level_coordinates(grid: np.ndarray, axis: int) -> tuple:
    """
    One coordinate per grid level, with the spread of the readings behind it.

    The **median** represents a level.  Every reading in a level was taken at the same
    setting, so the scatter between them is the instrument's, not the experiment's, and
    a level's coordinate should not move because one reading was bad — a stray reading
    small enough to pass :func:`_levels_separate` still drags a mean.  The median is
    also unbiased when a level *drifts* through its readings, which a read-back power
    does while the laser settles after a setpoint change; the first reading of the level
    is the one taken before it settled.

    Levels stay in acquisition order — index *i* is the *i*-th level as measured, not the
    *i*-th smallest — so a descending sweep stays descending.

    Returns ``(coordinate, spread)``, both ``(n_levels,)``.  *spread* is each level's
    peak-to-peak range, which is what makes a level that is not really one visible.
    """
    return np.median(grid, axis=axis), np.ptp(grid, axis=axis)


def _overlap_report(values: np.ndarray, axis: int, shape: tuple, unit: str = "") -> str:
    """
    Describe the first pair of grid levels whose readings interleave.

    *values* is the flat ``(n_sweeps,)`` axis and *shape* the ``(n_fast, n_slow)``
    candidate being explained.  Returns ``""`` when the levels are in fact separate.
    """
    n_fast, n_slow = shape
    lo, hi = _level_bounds(values.reshape(n_slow, n_fast), axis)
    # ~(gap > 0) rather than (gap <= 0), so a non-finite bound counts as a clash.
    clash = np.flatnonzero(~(lo[1:] - hi[:-1] > 0))
    if clash.size == 0:
        return ""
    i = int(clash[0])
    suffix = f" {unit}" if unit else ""
    return (f"one covers {lo[i]:.6g}–{hi[i]:.6g}{suffix} and the next "
            f"{lo[i + 1]:.6g}–{hi[i + 1]:.6g}{suffix}")


@_dataclass(frozen=True, eq=False)
class SweepNesting:
    """
    A declared 2-D sweep: a fast (inner) axis run to completion inside a slow one.

    Held by :attr:`_Sweep.nesting`.  Each coordinate array holds one value per
    level of that axis, in acquisition order, so a descending sweep stays descending;
    they are *not* sorted.  A coordinate is the **median** of the readings taken at
    that level, and ``fast_spread`` / ``slow_spread`` carry the peak-to-peak range
    those medians came from — zero for a commanded axis read back exactly, and the
    thing to look at when a level is not as flat as it should be.

    ``asserted`` records that the shape was named with ``n_fast=`` / ``n_slow=``
    rather than established from the readings, so at least one axis may not be able
    to tell its own settings apart.  A caller reading coordinates off such a nest is
    reading what the instrument happened to report at each level.

    ``fast_group`` / ``slow_group`` name the row that established each axis's levels.
    Normally that is the axis itself; it differs when ``fast_group_by=`` /
    ``slow_group_by=`` pointed the grouping at a commanded setpoint while the
    coordinates stayed on a measured quantity.
    """
    fast_type   : str
    fast_label  : str
    fast_unit   : str
    fast_axis   : np.ndarray
    slow_type   : str
    slow_label  : str
    slow_unit   : str
    slow_axis   : np.ndarray
    asserted    : bool = False
    fast_spread : np.ndarray = None
    slow_spread : np.ndarray = None
    fast_group  : str = None
    slow_group  : str = None

    @property
    def n_fast(self) -> int:
        """Points along the fast axis."""
        return int(self.fast_axis.size)

    @property
    def n_slow(self) -> int:
        """Points along the slow axis."""
        return int(self.slow_axis.size)

    @property
    def shape(self) -> tuple:
        """``(n_slow, n_fast)`` — the grid shape a flat sweep reshapes to."""
        return (self.n_slow, self.n_fast)

    @property
    def fast_axis_label(self) -> str:
        """Label for :attr:`fast_axis`, with its unit when one is known."""
        return f"{self.fast_label} ({self.fast_unit})" if self.fast_unit \
            else self.fast_label

    @property
    def slow_axis_label(self) -> str:
        """Label for :attr:`slow_axis`, with its unit when one is known."""
        return f"{self.slow_label} ({self.slow_unit})" if self.slow_unit \
            else self.slow_label

    def __str__(self) -> str:
        # A grouping row is named where it differs, since it is what decided which
        # spectra share a setting and a reader cannot infer it from the coordinates.
        def side(kind, n, group):
            axis = getattr(self, f"{kind}_type")
            via  = f" via {group}" if group is not None and group != axis else ""
            return f"{axis} ({n}, {kind}{via})"
        shape = "" if not self.asserted else ", shape asserted"
        return (f"{side('fast', self.n_fast, self.fast_group)} × "
                f"{side('slow', self.n_slow, self.slow_group)} = "
                f"{self.n_fast * self.n_slow}{shape}")


class _Sweep:
    """
    Shared machinery for a parameter sweep, whatever the measured axis.

    Not constructed directly — see :class:`AttoCubeSpectralSweep` (wavelength /
    energy) and :class:`AttoCubeTRPLSweep` (time).  Everything here is
    independent of what the spectral axis *is*: the parameter store, the curated
    registry, sweep-axis resolution, measurement-type metadata, and HDF5 export.

    Subclasses declare which export layout they read (``_LAYOUT_KIND``) and where
    their axis and signal arrays live (``_AXIS_ATTR``, ``_SIGNAL_ATTR``), then
    drive construction explicitly:

    .. code-block:: python

        payload = self._decode_and_describe(path, spectra_type=..., ...)
        # ... set the axis and signal arrays, apply corrections ...
        self._bind_sweep_axis(sweep, sweep_label, sweep_unit)
        self._bind_nesting(fast_sweep, slow_sweep)

    That sequence is spelled out in each subclass rather than hidden behind a
    template method, because the ordering matters — the sweep axis cannot be
    resolved until ``n_sweeps`` is known — and a reader should be able to see it.
    """

    # Set by subclasses.
    _LAYOUT_KIND = None     # "spectral" | "temporal": which export it accepts
    _AXIS_ATTR   = None     # attribute holding the (n_points,) axis
    _SIGNAL_ATTR = None     # attribute holding the (n_points, n_sweeps) signal

    # No nest until one is declared, so a subclass that never calls
    # _bind_nesting still answers is_nested.
    _nesting = None

    # Canonical curated parameters: attribute name -> (CSV row label, scale, unit).
    # These are the analysis-primary quantities promoted to first-class
    # properties (scaled views into :attr:`parameters`).  The label, scale and/or
    # unit of any entry can be overridden per-instance via ``curated_labels`` /
    # ``curated_scales`` / ``curated_units``; everything else in the file is
    # reached through the generic :attr:`parameters` store.  A row listed here
    # that a given file does not contain is not an error — the property raises
    # only if accessed.
    #
    # The unit is not decoration: it is what ``_SWEEP_TYPES`` reads for a
    # curated-backed axis, so it reaches every axis label, ``__repr__`` line and
    # legend entry.  A scale and its unit are one fact and move together.
    #
    # The role-backed entries are the exception: their labels come from ``gates``
    # alone, and the rows below are never read without one.  They are listed here
    # so that the file's own candidate rows can be named in that error, and so
    # ``curated_scales`` still reaches them.
    #
    # The two voltages carry a default row because ``_gate_candidates`` and
    # ``gate_mode`` both describe an *undeclared* scan and need somewhere to look;
    # the currents have no such reader, so a default there would be a guess nothing
    # consults.  Their label is filled in from ``gates`` via
    # ``_CHANNEL_SIBLING_CURRENT``, and access is refused until it is.
    _CURATED = {
        "v_top":     ("V_A",              1.0,      "V"),
        "v_bot":     ("V_B",              1.0,      "V"),
        "power":     ("Excitation Power", 0.303e6,  "µW"),
        "i_top":     (None,               1e9,      "nA"),
        "i_bot":     (None,               1e9,      "nA"),
        "i_channel": (None,               1e9,      "nA"),
        "scanner_x": ("Scanner X",        1.0,      "V"),
        "scanner_y": ("Scanner Y",        1.0,      "V"),
    }

    # --- Construction helpers ----------------------------------------------

    def _decode_and_describe(
        self,
        path,
        *,
        spectra_type   : str  = None,
        geometry              = None,
        gates          : dict = None,
        curated_labels : dict = None,
        curated_scales : dict = None,
        curated_units  : dict = None,
    ) -> dict:
        """
        Decode *path* and settle everything that does not depend on the axis.

        Returns the payload so the subclass can pull its axis and signal arrays
        out of it.  See the class docstring for where this sits in construction.
        """
        self.path = str(path)

        payload = self._decode(path)
        meta    = payload["metadata"]
        self.source_metadata = dict(meta)

        # Every labeled row -> per-sweep array in the file's raw units.  Exposes
        # the full instrument state (Scanner X/Y, Galvo X/Y, T, laser settings,
        # …), not just the curated few.
        self.parameters = payload["parameters"]
        # Sweep points the header declared; exceeds n_sweeps when the exporter
        # over-allocated and zero-filled the surplus.  None -> no over-allocation.
        self._n_declared = payload.get("n_declared")

        self.spectra_type = self._resolve_spectra_type(spectra_type, meta)
        self.geometry     = geometry if geometry is not None else meta.get("geometry")

        # Which acquisition channel reached which electrode, and thereby which
        # electrodes the device has at all.  The explicit argument first, then
        # whatever an HDF5 file recorded.  ``None`` means the session's wiring was
        # never stated, and every role-dependent quantity then refuses rather than
        # guessing.  Resolved before the curated registry because it supplies two of
        # that registry's labels.
        self._gates = self._resolve_gates(
            gates if gates is not None else meta.get("gates")
        )

        # First creates a curated registry using defaults
        # Then updates the registry with values from the file's metadata and any explicit overrides provided during construction.
        # Raises an error if an unknown curated parameter name is encountered.
        self._curated = {name: list(cfg) for name, cfg in self._CURATED.items()}
        # An HDF5 file dumps the *resolved* label of every curated entry, gates
        # included.  Those are the writer's bookkeeping rather than a statement of
        # wiring, so drop them here and let the file's own ``gates`` attribute be
        # the only thing that can declare a role — a caller passing them is a
        # different case, and raises below.
        meta_labels = {
            name: value
            for name, value in (meta.get("curated_labels") or {}).items()
            if name not in _DECLARED_CURATED
        }
        # Curated config is (label, scale, unit) and all three can be overridden,
        # so idx runs 0..2.  ``from_caller`` marks the stores that came from the
        # constructor rather than from the file, because only a scale the *caller*
        # changed can be missing its unit: an HDF5 file dumps a value for every
        # entry, so a key is never absent on that side.
        rescaled_by_caller = {}
        for store, idx, from_caller in ((meta_labels,                0, False),
                                        (curated_labels,             0, True),
                                        (meta.get("curated_scales"), 1, False),
                                        (curated_scales,             1, True),
                                        (meta.get("curated_units"),  2, False),
                                        (curated_units,              2, True)):
            for name, value in (store or {}).items():
                if name not in self._curated:
                    raise ValueError(
                        f"'{name}' is not a curated parameter. "
                        f"Valid names: {sorted(self._CURATED)}."
                    )
                if idx == 0 and name in _DECLARED_CURATED:
                    raise ValueError(
                        f"'{name}' cannot be set through curated_labels. Which "
                        f"acquisition channel drove which physical electrode is "
                        f"per-session wiring, declared through gates= so that the "
                        f"mapping is recorded once and travels with the scan: "
                        f"gates={{'top': '<row>', 'bottom': '<row>'}}. A declared "
                        f"row names that electrode's current row too, so the "
                        f"currents are not set separately either."
                    )
                # A curated entry the file's metadata and the arguments both leave
                # alone keeps its default from _CURATED.
                if idx == 0:
                    # Verbatim: a current's label is legitimately None until gates
                    # names it, so this slot is not always a string.
                    self._curated[name][idx] = value
                elif idx == 1:
                    if from_caller and float(value) != self._curated[name][idx]:
                        rescaled_by_caller[name] = (self._curated[name][idx],
                                                    float(value))
                    self._curated[name][idx] = float(value)
                else:
                    if value is None:
                        raise ValueError(
                            f"curated_units['{name}'] is None. A unit is a string; "
                            f"pass \"\" for a dimensionless quantity, which is how "
                            f"the sweep index spells its own."
                        )
                    self._curated[name][idx] = str(value)

        # A scale the caller changed without saying what the numbers now are leaves
        # a label that misstates them — this registry is what every axis label,
        # __repr__ line and legend entry reads the unit through.  Restating the
        # unit silences this, which is what a polarity flip does: -1e9 nA is still
        # nA.  Not raised, because the researcher may be mid-calibration and the
        # numbers are correct either way; not silent, because nothing downstream
        # can detect it.
        unstated = sorted(set(rescaled_by_caller) - set(curated_units or {}))
        if unstated:
            detail = "; ".join(
                f"'{name}' {rescaled_by_caller[name][0]:g} -> "
                f"{rescaled_by_caller[name][1]:g}, unit still "
                f"{self._curated[name][2]!r}"
                for name in unstated
            )
            warnings.warn(
                f"curated_scales rescaled {detail}. Pass curated_units="
                f"{{{', '.join(repr(n) + ': ...' for n in unstated)}}} to state "
                f"what the numbers now are — the unit reaches every axis label, "
                f"__repr__ line and legend entry, so it is a misread otherwise.",
                # Measured, not counted off def lines: 1 is here, 2 is the
                # constructor that called this, 3 is the researcher's own line.
                # AttoCubePLVabScan adds a frame and so points at its own
                # super().__init__ instead; it is deprecated, and its comment
                # there says so.  This is a new site, not one of A11's fifteen.
                UserWarning, stacklevel=3,
            )

        # The declared mapping is what backs the gate roles.  A role left out, or
        # present with a None row (an electrode tied to ground that no channel
        # records), leaves its _CURATED entry alone — nothing will read it, because
        # the raise lives on ``self._gates`` so that a defaulted label can never be
        # mistaken for a stated one.
        for role, attr in _GATE_ROLE_CURATED.items():
            label = (self._gates or {}).get(role)
            if label is not None:
                self._curated[attr][0] = label

        # The same declaration also names each electrode's current row, because a
        # source-meter channel's bias and current are one terminal.  A role declared
        # on a row outside _CHANNEL_SIBLING_CURRENT leaves its current label None,
        # and _role_current raises rather than guessing a sibling from the spelling.
        for role, attr in _ROLE_CURRENT_CURATED.items():
            label = (self._gates or {}).get(role)
            if label is not None:
                self._curated[attr][0] = _CHANNEL_SIBLING_CURRENT.get(label)

        # Conver to tuple - not mutable, so that the curated parameters cannot be changed after initialization.
        self._curated = {name: tuple(cfg) for name, cfg in self._curated.items()}

        return payload

    @staticmethod
    def _resolve_gates(gates) -> dict:
        """
        Validate a declared electrode mapping, or pass ``None`` through.

        The roles *present* describe the device: two gate electrodes is a
        dual-gated stack, one gate plus a channel contact is a single-gated one.
        Returns a copy in canonical role order, so that mutating the caller's dict
        afterwards cannot change what the scan recorded.
        """
        if gates is None:
            return None
        if not isinstance(gates, dict):
            raise ValueError(
                f"gates must be a dict mapping electrode roles to parameter rows, "
                f"got {type(gates).__name__}. Valid roles: {sorted(_GATE_ROLES)}."
            )

        unknown = sorted(set(gates) - set(_GATE_ROLES))
        if unknown:
            raise ValueError(
                f"gates names unknown role(s) {unknown}. Valid roles: "
                f"{sorted(_GATE_ROLES)} — 'top'/'bottom' are gate electrodes, "
                f"'channel' is a contact to the TMDC itself."
            )

        electrodes = set(gates) & set(_GATE_ELECTRODES)
        if not electrodes:
            raise ValueError(
                f"gates must name at least one gate electrode "
                f"({' or '.join(_GATE_ELECTRODES)}); got {sorted(gates)}. A channel "
                f"contact alone describes no gated device."
            )
        # One gate and no channel is ambiguous in the way this argument exists to
        # remove: it cannot be told from a two-gate device whose other gate was
        # forgotten.  Requiring the contact makes the single-gate case a statement.
        if len(electrodes) == 1 and "channel" not in gates:
            missing = next(e for e in _GATE_ELECTRODES if e not in electrodes)
            raise ValueError(
                f"gates names only the {electrodes.pop()} gate, which is ambiguous: "
                f"a single-gated device is declared by naming its channel contact "
                f"too — gates={{..., 'channel': '<row>'}}, or 'channel': None when "
                f"the TMDC is hard-grounded and no row records it. If the device is "
                f"dual-gated, name the {missing!r} gate as well (None if it is tied "
                f"to ground). A lone gate with a floating TMDC defines neither a "
                f"field nor a density."
            )
        return {role: gates[role] for role in _GATE_ROLES if role in gates}

    def _bind_sweep_axis(self, sweep, sweep_label, sweep_unit) -> None:
        """
        Resolve the sweep axis, falling back to what the file recorded.

        Called last in construction: resolving a sweep validates against
        ``n_sweeps``, which is not known until the signal array is in place.
        """
        meta = self.source_metadata
        (self.sweep_type, self._sweep_source,
         self._sweep_label, self._sweep_unit) = self._resolve_sweep(
            sweep       if sweep       is not None else meta.get("sweep"),
            sweep_label if sweep_label is not None else meta.get("sweep_label"),
            sweep_unit  if sweep_unit  is not None else meta.get("sweep_unit"),
        )

    def _bind_nesting(self, fast_sweep, slow_sweep, *,
                      n_fast=None, n_slow=None,
                      fast_group_by=None, slow_group_by=None) -> None:
        """
        Resolve the declared nest, then check the sweep axis against it.

        Called after :meth:`_bind_sweep_axis`: both nest axes are length
        ``n_sweeps`` and are verified against it, and a curated axis may read a
        property that the sweep resolution has already checked the rows for.
        """
        self._nesting = self._resolve_nesting(
            fast_sweep, slow_sweep, n_fast=n_fast, n_slow=n_slow,
            fast_group_by=fast_group_by, slow_group_by=slow_group_by)
        self._warn_if_sweep_axis_repeats()

    def _warn_if_sweep_axis_repeats(self) -> None:
        """
        Warn when the declared sweep axis does not give each point its own value.

        The sweep axis is what a map positions its spectra along, so points
        sharing a value land on top of each other and only one of them is drawn.
        ``__repr__`` prints the endpoints and reads perfectly well either way, so
        nothing else announces it.

        Both ways a flattened nest does this are caught: the inner quantity
        restarts every row and the outer holds still through one, and both leave
        repeats.  Repeat measurements at each setting collapse a map the same
        way, and are warned about for the same reason.

        Not an error — one quantity of a nest is a legitimate axis when the
        caller means to slice, and the declaration is theirs to make.
        """
        if self.sweep_type == "index":
            return
        axis   = self.sweep_axis
        finite = axis[np.isfinite(axis)]
        if finite.size < 2:
            return
        n_distinct = _count_distinct(finite, _axis_atol(finite))
        if n_distinct >= finite.size:
            return

        # For the message only: naming the nest, when there is one to name, turns
        # "this is wrong" into "here is what to use instead".  A single value is its
        # own case — the row was held still, so it is not a nest axis at all and
        # pointing at fast_sweep= would send the reader the wrong way.
        structure = self._nesting if self._nesting is not None else self.sweep_grid()
        if n_distinct == 1:
            fix = (" This row was held at one setting for the whole file, so these "
                   "are repeat measurements rather than a sweep: varying_parameters()"
                   " lists the rows that did move, and sweep=None gives the flat "
                   "index.")
        elif structure is not None:
            fix = (f" These points are a 2-D nest ({structure}) — address it with "
                   f"as_grid() and get_spectrum_at(fast=..., slow=...).")
        else:
            fix = (" If these points are a nest, declare it with fast_sweep= and "
                   "slow_sweep=; if they are repeat measurements at each setting, "
                   "sweep=None gives the flat index.")

        unit  = f" {self._sweep_unit}" if self._sweep_unit else ""
        value = "value" if n_distinct == 1 else "different values"
        warnings.warn(
            f"sweep={self.sweep_type!r} takes only {n_distinct} {value} "
            f"across {finite.size} sweep points in '{self.path}' "
            f"({finite.min():.4g} → {finite.max():.4g}{unit}), so it does not "
            f"label them individually. Plotted against it, points sharing a value "
            f"land in the same place and only one of them is drawn.{fix}",
            UserWarning, stacklevel=4,
        )

    def _resolve_nesting(self, fast_sweep, slow_sweep, *,
                         n_fast=None, n_slow=None,
                         fast_group_by=None, slow_group_by=None) -> SweepNesting:
        """Build the declared nest, or return ``None`` when none was declared."""
        meta = self.source_metadata
        fast = fast_sweep if fast_sweep is not None else meta.get("fast_sweep")
        slow = slow_sweep if slow_sweep is not None else meta.get("slow_sweep")
        n_fast = n_fast if n_fast is not None else meta.get("n_fast")
        n_slow = n_slow if n_slow is not None else meta.get("n_slow")
        fast_group_by = (fast_group_by if fast_group_by is not None
                         else meta.get("fast_group_by"))
        slow_group_by = (slow_group_by if slow_group_by is not None
                         else meta.get("slow_group_by"))

        if fast is None and slow is None:
            # An asserted shape says how many points each axis has, and a grouping row
            # says which points share a setting.  Neither says what was scanned, so
            # neither can stand in for the declaration.
            for name, value in (("n_fast/n_slow", n_fast if n_fast is not None
                                 else n_slow),
                                ("fast_group_by/slow_group_by",
                                 fast_group_by if fast_group_by is not None
                                 else slow_group_by)):
                if value is not None:
                    raise ValueError(
                        f"{name} was given without fast_sweep= and slow_sweep=. It "
                        f"says how the {self.n_sweeps} sweep points divide up, not "
                        f"what was scanned along each axis. Name both axes as well."
                    )
            return None

        # One without the other cannot be told from a forgotten second axis, and
        # which of the two is inner is the fact the declaration exists to carry.
        if fast is None or slow is None:
            given, missing = (("slow_sweep", "fast_sweep") if fast is None
                              else ("fast_sweep", "slow_sweep"))
            raise ValueError(
                f"{given}={slow if fast is None else fast!r} was declared without "
                f"{missing}. A nest needs both axes named: the inner one as "
                f"fast_sweep (it runs to completion at each point of the outer), "
                f"the outer as slow_sweep. Pass neither to leave the sweep flat."
            )

        if fast == slow:
            raise ValueError(
                f"fast_sweep and slow_sweep are both {fast!r}. A nest needs two "
                f"different axes."
            )

        resolved = {}
        for param, declared in (("fast_sweep", fast), ("slow_sweep", slow)):
            key, source, label, unit = self._resolve_sweep(
                declared, None, None, param=param)
            if key == "index":
                raise ValueError(
                    f"{param}='index' is not an axis of a nest — the sweep index "
                    f"is the flat position the nest is being declared over. Name "
                    f"the parameter that was scanned."
                )
            resolved[param] = (key, self._axis_for_source(source), label, unit)

        (fast_key, fast_flat, fast_label, fast_unit) = resolved["fast_sweep"]
        (slow_key, slow_flat, slow_label, slow_unit) = resolved["slow_sweep"]

        # Which row establishes each axis's levels.  Normally the axis itself; a
        # separate row when the labelled quantity is measured and cannot resolve its
        # own settings while a commanded one can.
        fast_group_key, fast_group = self._resolve_grouping(
            "fast_group_by", fast_group_by, fast_key, fast_flat)
        slow_group_key, slow_group = self._resolve_grouping(
            "slow_group_by", slow_group_by, slow_key, slow_flat)

        fast_side = ("fast_sweep" if fast_group_key == fast_key else "fast_group_by",
                     fast_group_key, fast_group, fast_unit if
                     fast_group_key == fast_key else "", _FAST_LEVELS_AXIS)
        slow_side = ("slow_sweep" if slow_group_key == slow_key else "slow_group_by",
                     slow_group_key, slow_group, slow_unit if
                     slow_group_key == slow_key else "", _SLOW_LEVELS_AXIS)

        asserted = n_fast is not None or n_slow is not None
        if asserted:
            # Asserted, so the checks report rather than decide: the whole point of
            # naming a shape is to load a file whose readings cannot establish one.
            n_fast, n_slow = self._asserted_shape(n_fast, n_slow)
            self._warn_if_asserted_levels_overlap(
                fast_side, slow_side, (n_fast, n_slow))
        else:
            candidates = _nest_candidates(fast_group, slow_group, self.n_sweeps)
            if len(candidates) != 1:
                raise ValueError(
                    self._nesting_failure(fast_side, slow_side, candidates))
            n_fast, n_slow = candidates[0]

        # One coordinate per level, taken from the *labelled* row rather than the row
        # that grouped the points — the grouping settles which spectra share a
        # setting, the label settles what that setting is called.
        fast_axis, fast_spread = _level_coordinates(
            fast_flat.reshape(n_slow, n_fast), _FAST_LEVELS_AXIS)
        slow_axis, slow_spread = _level_coordinates(
            slow_flat.reshape(n_slow, n_fast), _SLOW_LEVELS_AXIS)

        # A grouped axis is expected to be messy — that is why it was grouped by
        # something else — but a coordinate whose levels overlap cannot be plotted or
        # looked up meaningfully, and only the caller can decide whether that matters.
        for param, key, label, unit, values, axis, shape in (
            ("fast_sweep", fast_key, fast_label, fast_unit, fast_flat,
             _FAST_LEVELS_AXIS, (n_fast, n_slow)),
            ("slow_sweep", slow_key, slow_label, slow_unit, slow_flat,
             _SLOW_LEVELS_AXIS, (n_fast, n_slow)),
        ):
            grouped_by = fast_group_key if axis == _FAST_LEVELS_AXIS else slow_group_key
            if grouped_by == key:
                continue                    # not grouped separately; already checked
            if _levels_separate(values.reshape(n_slow, n_fast), axis):
                continue
            warnings.warn(
                f"{param}={key!r} is the coordinate of a nest grouped by "
                f"{grouped_by!r}, and its levels overlap: "
                f"{_overlap_report(values, axis, shape, unit)}. The grouping is sound, "
                f"so the grid is right and every spectrum is on the setting it was "
                f"measured at. What is not sound is this axis as a coordinate — "
                f"plotting against it misleads and get_spectrum_at() on it is "
                f"ambiguous. Use {grouped_by!r} as the axis if you need to address the "
                f"nest by value.",
                UserWarning, stacklevel=4,
            )

        return SweepNesting(
            fast_type   = fast_key,
            fast_label  = fast_label,
            fast_unit   = fast_unit,
            fast_axis   = fast_axis,
            slow_type   = slow_key,
            slow_label  = slow_label,
            slow_unit   = slow_unit,
            slow_axis   = slow_axis,
            asserted    = asserted,
            fast_spread = fast_spread,
            slow_spread = slow_spread,
            fast_group  = fast_group_key,
            slow_group  = slow_group_key,
        )

    def _resolve_grouping(self, param, declared, axis_key, axis_flat) -> tuple:
        """
        Resolve a ``*_group_by=`` row, defaulting to the axis it groups.

        Returns ``(key, values)``.  The key equals *axis_key* when nothing separate was
        declared, which is what every caller tests to tell the two cases apart.
        """
        if declared is None:
            return axis_key, axis_flat
        key, source, _, _ = self._resolve_sweep(declared, None, None, param=param)
        if key == "index":
            raise ValueError(
                f"{param}='index' cannot group a nest — the sweep index is a "
                f"different value at every point, so it puts each spectrum on a level "
                f"of its own. Name the row that was commanded."
            )
        return key, self._axis_for_source(source)

    def _asserted_shape(self, n_fast, n_slow) -> tuple:
        """
        Validate an asserted nest shape against ``n_sweeps``, filling in the other half.

        Either count alone is enough, since the total is known exactly; giving both is
        allowed and cross-checked, which is what catches a half-remembered scan.  Each
        count is a number of grid points, so neither may be below 2 — a nest with one
        row is not a nest.
        """
        n = self.n_sweeps
        for name, value in (("n_fast", n_fast), ("n_slow", n_slow)):
            if value is None:
                continue
            if int(value) != value or value < 2:
                raise ValueError(
                    f"{name}={value!r} is not a usable axis length. It counts grid "
                    f"points along one axis of the nest, so it must be a whole "
                    f"number of at least 2."
                )

        if n_fast is not None and n_slow is not None:
            if int(n_fast) * int(n_slow) != n:
                raise ValueError(
                    f"n_fast={int(n_fast)} × n_slow={int(n_slow)} = "
                    f"{int(n_fast) * int(n_slow)}, but '{self.path}' holds {n} sweep "
                    f"points. Every point of the slow axis carries one complete run "
                    f"of the fast axis, so the two counts multiply to the total. If "
                    f"the scan was aborted part-way through a row, the completed rows "
                    f"have to be sliced out first."
                )
            return int(n_fast), int(n_slow)

        # One given: the other follows from the total, which also rejects a count that
        # leaves a partial row rather than silently truncating one.
        given_name, given = (("n_fast", n_fast) if n_fast is not None
                             else ("n_slow", n_slow))
        given = int(given)
        if n % given:
            raise ValueError(
                f"{given_name}={given} does not divide the {n} sweep points in "
                f"'{self.path}' — it leaves {n % given} over. Every point of the slow "
                f"axis carries one complete run of the fast axis, so each count "
                f"divides the total. If the scan was aborted part-way through a row, "
                f"the completed rows have to be sliced out first."
            )
        other = n // given
        if other < 2:
            raise ValueError(
                f"{given_name}={given} leaves {other} point(s) on the other axis of "
                f"the {n}-point sweep, which is not a nest. Leave fast_sweep= and "
                f"slow_sweep= off to keep the sweep flat."
            )
        return (given, other) if given_name == "n_fast" else (other, given)

    def _warn_if_asserted_levels_overlap(self, fast_side, slow_side, shape) -> None:
        """
        Warn for each axis whose levels interleave under an asserted *shape*.

        An asserted shape exists to load a file whose readings cannot establish one, so
        this cannot refuse.  What it can do is say which axis does not hold apart, and
        that is worth doing in both directions: on the file this was built for the
        warning names the axis already known to be bad, while a shape asserted the
        wrong way round names an axis known to be good — which is the loud signal that
        the two counts were transposed.
        """
        n_fast, n_slow = shape
        for param, key, values, unit, axis in (fast_side, slow_side):
            if _levels_separate(values.reshape(n_slow, n_fast), axis):
                continue
            warnings.warn(
                f"{param}={key!r} does not hold apart at the asserted "
                f"{n_fast} × {n_slow}: "
                f"{_overlap_report(values, axis, shape, unit)}. The reshape was done "
                f"as asserted, so the grid is the shape you named, but this axis "
                f"cannot tell its own settings apart. If it is one you expect to step "
                f"cleanly, check that n_fast and n_slow are not the wrong way round.",
                UserWarning, stacklevel=5,
            )

    def _nesting_failure(self, fast_side, slow_side, candidates) -> str:
        """
        Explain why a declared nest did not resolve to one shape.

        Each side is ``(param, key, values, unit, levels_axis)`` for the row that was
        actually tested, which is the declared axis unless a ``*_group_by=`` replaced
        it.  *candidates* is what :func:`_nest_candidates` returned — empty when nothing
        fitted, longer than one when the file reshapes more than one way.
        """
        n = self.n_sweeps
        (fast_param, fast_key, fast_flat, fast_unit, _) = fast_side
        (slow_param, slow_key, slow_flat, slow_unit, _) = slow_side
        head = (f"{fast_param}={fast_key!r} inside {slow_param}={slow_key!r} does not "
                f"describe the {n} sweep points in '{self.path}'.")

        # More than one shape fits, so the file does not say which was measured.
        if candidates:
            shapes = ", ".join(f"{f} × {s}" for f, s in candidates)
            return (f"{head}\n  {len(candidates)} shapes fit it equally well "
                    f"({shapes} as fast × slow), and nothing in the file says which "
                    f"was measured. Name it with n_fast= or n_slow=, declare an axis "
                    f"whose readings distinguish the shapes, or leave the sweep flat.")

        # The declaration exists to settle which axis is inner, so the reversed
        # reading is the one mistake worth naming outright.
        if _nest_shape(slow_flat, fast_flat, n) is not None:
            return (f"{head}\n  Swapping them does: pass "
                    f"fast_sweep={slow_key!r}, slow_sweep={fast_key!r}. The fast "
                    f"axis is the one that runs to completion at each point of "
                    f"the slow one.")

        # Nothing fitted.  What is worth reporting is the shape the file *does* look
        # to have, and which axis broke it.  Whichever axis separates pins that
        # shape — a fast axis that separates fixes the period of the inner loop, a
        # slow axis that separates fixes the outer one — and the axis that does not
        # is then the one at fault, so its overlapping readings can be quoted.
        # Anchoring on an arbitrary divisor instead would blame a healthy axis at a
        # shape the caller never declared.
        shapes = [(d, n // d) for d in range(2, n // 2 + 1)
                  if n % d == 0 and n // d >= 2]

        # The fast axis is tried as the anchor first: it is the declared inner loop,
        # and it is the axis a commanded parameter is usually on.
        for anchor, culprit in ((fast_side, slow_side), (slow_side, fast_side)):
            a_param, a_key, a_values, _, a_axis = anchor
            c_param, c_key, c_values, c_unit, c_axis = culprit
            for n_fast, n_slow in shapes:
                if not _levels_separate(a_values.reshape(n_slow, n_fast), a_axis):
                    continue
                detail = _overlap_report(c_values, c_axis, (n_fast, n_slow), c_unit)
                return (
                    f"{head}\n  At {n_fast} × {n_slow}, {a_param}={a_key!r} holds "
                    f"apart but {c_param}={c_key!r} does not: {detail}. Readings "
                    f"that overlap cannot be told apart, so those spectra cannot be "
                    f"assigned to a setting by value. Declare a row that steps "
                    f"exactly — a commanded setpoint rather than a measured "
                    f"read-back — and compare against varying_parameters() to find "
                    f"one. To reshape anyway and accept the axis as it is, assert the "
                    f"shape with n_fast={n_fast} (or n_slow={n_slow})."
                )

        return (f"{head}\n  No shape fits: {n} sweep points do not divide into two "
                f"axes whose readings hold apart. Either one of these is not a nest "
                f"axis, or the scan was aborted part-way through a row — a partial "
                f"final row cannot be reshaped. Compare against sweep_grid() and "
                f"varying_parameters(); slice the completed rows, or leave the "
                f"sweep flat.")

    def _validate_axis_and_signals(self, axis, signals: dict) -> None:
        """
        Check the decoded arrays are self-consistent before anything uses them.

        *signals* maps a display name (as it appears in the export header) to its
        ``(n_points, n_sweeps)`` array, so the error message can name the field
        the caller would recognise.

        - Checks that the measured axis is a non-empty 1-D array.
        - Checks that every signal array has the same number (the right number) of rows as the axis.
        - Checks that every signal array has the same number of columns (same array shape).
        - Checks that every parameter row has exactly one value per sweep point.
        """

        # Checks that the measured axis is a non-empty 1-D array.
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(
                f"'{self.path}' yielded an axis of shape {axis.shape}; "
                f"expected a non-empty 1-D array."
            )
        # Checks that every signal array has the same number (the right number) of rows as the axis.
        n_points = axis.size
        for name, arr in signals.items():
            if arr.ndim != 2 or arr.shape[0] != n_points:
                raise ValueError(
                    f"'{self.path}': {name} has shape {arr.shape}, which does "
                    f"not match the {n_points}-point axis. Expected "
                    f"({n_points}, {n_sweeps})."
                )
        # Checks that every signal array has the same number of columns (same array shape).
        shapes = {name: arr.shape for name, arr in signals.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"'{self.path}': mismatched signal shapes {shapes}.")

        n_sweeps = next(iter(signals.values())).shape[1]
        # Checks that every parameter row has exactly one value per sweep point.
        bad = {lbl: arr.shape for lbl, arr in self.parameters.items()
               if arr.shape != (n_sweeps,)}
        if bad:
            raise ValueError(
                f"'{self.path}': parameter rows {bad} do not have one value per "
                f"sweep point (expected shape ({n_sweeps},))."
            )

    # --- Decoding ----------------------------------------------------------

    def _decode(self, path) -> dict:
        """
        Read *path* into the payload every format decodes to.

        The payload carries the axis, the signal array(s), ``parameters`` and
        ``metadata``.  Only the decoders differ between formats; everything after
        this point is shared, which is why HDF5 input needs no second class.
        """
        path = Path(path)
        if path.is_dir():
            return self._decode_dir(path)
        suffix = path.suffix.lower()
        if suffix in _HDF5_SUFFIXES:
            from . import hdf5 as _hdf5      # local: h5py is only needed here
            payload = _hdf5.read_sweep(path)
            # An .h5 carries its axis kind, so the wrong class is caught here just
            # as a mismatched header is for a CSV.
            expected = _hdf5._AXIS_KIND_FOR_LAYOUT[self._LAYOUT_KIND]["name"]
            found    = payload.pop("axis_kind")
            if found != expected:
                raise ValueError(
                    f"'{path}' stores a '{found}' axis, but {type(self).__name__} "
                    f"reads '{expected}'. Use "
                    f"{_hdf5._CLASS_FOR_AXIS_KIND[found]} instead."
                )
            return payload
        if suffix in _CSV_SUFFIXES:
            return self._decode_csv(path)
        raise ValueError(
            f"Unrecognised input format '{suffix}' for '{path}'. Expected a raw "
            f"AttoCube export ({', '.join(_CSV_SUFFIXES)}) or a file written by "
            f"to_hdf5 ({', '.join(_HDF5_SUFFIXES)})."
        )

    @classmethod
    def _decode_csv(cls, path) -> dict:
        raise NotImplementedError(f"{cls.__name__} defines no CSV decoder.")

    def _decode_dir(self, path) -> dict:
        raise ValueError(
            f"'{path}' is a directory, but {type(self).__name__} reads a single "
            f"file. Only TRPL sweeps are stored as a directory of per-point files."
        )

    @staticmethod
    def _resolve_spectra_type(spectra_type: str, meta: dict) -> str:
        """
        Resolve and validate :attr:`spectra_type` from argument and file metadata.

        Required for a raw export, which records nothing.  Taken from the file
        otherwise; if both are present and disagree, the argument wins but the
        disagreement is not swallowed — a relabelled measurement is exactly the
        kind of error that survives into every downstream figure.
        """
        from_file = meta.get("spectra_type")
        if spectra_type is None:
            if from_file is None:
                raise ValueError(
                    "spectra_type is required: this file records no measurement "
                    "type, and it cannot be inferred from the data. Pass one of "
                    f"{sorted(SPECTROSCOPY_TYPES)} — e.g. spectra_type='PL'."
                )
            return from_file

        if spectra_type not in SPECTROSCOPY_TYPES:
            raise ValueError(
                f"spectra_type={spectra_type!r} is not a recognised measurement "
                f"type. Choose from {sorted(SPECTROSCOPY_TYPES)}."
            )
        if from_file is not None and from_file != spectra_type:
            warnings.warn(
                f"This file records spectra_type={from_file!r} but "
                f"spectra_type={spectra_type!r} was passed; using the argument. "
                f"Re-export with to_hdf5 to correct the stored metadata.",
                UserWarning, stacklevel=4,
            )
        return spectra_type

    def _resolve_sweep(self, sweep, label, unit, *, param: str = "sweep") -> tuple:
        """
        Resolve a declared axis to ``(key, source, label, unit)``.

        *source* is a ``(kind, name)`` pair consumed by :attr:`sweep_axis`:
        ``("curated", attr)`` for a registry entry, ``("row", label)`` for a raw
        CSV row, ``("index", None)`` when nothing was declared.

        *param* is the name of the argument being resolved, interpolated into
        every message so the error names the argument the caller actually passed.
        """
        if sweep is None:
            sweep = "index"

        if sweep in _SWEEP_TYPES:
            source, default_label, default_unit = _SWEEP_TYPES[sweep]
            # A sweep resolving through a gate role needs the electrode declared
            # first, so this comes before the row check: without a mapping there is
            # no row to look for. Derived from _SWEEP_REQUIRES via
            # _ROLE_FOR_CURATED rather than listed again, so a new gate-backed
            # sweep type inherits the requirement.
            grounded = []
            for name in _SWEEP_REQUIRES.get(sweep, ()):
                role = _ROLE_FOR_CURATED.get(name)
                if role is None:
                    continue
                self._require_role(role, f"{param}={sweep!r}")
                if self._gates[role] is None:
                    grounded.append(role)
            if grounded:
                raise ValueError(
                    f"{param}={sweep!r} resolves through the "
                    f"{', '.join(repr(r) for r in grounded)} electrode, which gates "
                    f"declared as tied to ground — its voltage is zero at every "
                    f"sweep point, so it is not an axis. Sweep the driven electrode "
                    f"instead, or pass {param}=None."
                )
            # Fail on the row the *declared* sweep needs, not on every curated
            # row: the requirement follows what the caller said was measured.
            for name in _SWEEP_REQUIRES.get(sweep, ()):
                csv_label = self._curated[name][0]
                if csv_label not in self.parameters:
                    # A gate row is named through gates=, everything else through
                    # curated_labels; point at whichever one applies.
                    fix = (f"Re-declare it with gates={{'top': '<row>', "
                           f"'bottom': '<row>'}}"
                           if name in _GATE_CURATED else
                           f"Pass curated_labels={{'{name}': '<row>'}}")
                    raise KeyError(
                        f"{param}={sweep!r} needs the curated parameter '{name}' "
                        f"(row '{csv_label}'), which '{self.path}' does not "
                        f"contain. Available rows: {self.parameter_labels}. "
                        f"{fix} if it is under another name."
                    )
            if sweep == "electric_field" and self.geometry is None:
                raise ValueError(
                    f"{param}='electric_field' needs a DeviceGeometry to convert "
                    f"gate voltages into a field — pass geometry=. To use a "
                    f"raw gate voltage instead, use {param}='top_voltage' or "
                    f"'bottom_voltage'."
                )
            # carrier_density sums over whichever gates the device declares, so its
            # requirement is not a fixed row list and cannot live in
            # _SWEEP_REQUIRES.  Checked here so it fails at load, like the rest.
            if sweep == "carrier_density":
                if self.geometry is None:
                    raise ValueError(
                        f"{param}='carrier_density' needs a DeviceGeometry for the "
                        f"gate capacitance — pass geometry= with the hBN thickness "
                        f"of each gate. To use a raw gate voltage instead, use "
                        f"{param}='top_voltage' or 'bottom_voltage'."
                    )
                # Charge has to come from somewhere: a density is defined against
                # the contact that supplies it, so the contact must be declared.
                self._require_role("channel", f"{param}='carrier_density'")
                for role in _GATE_ELECTRODES:
                    if role in self._gates:
                        self.geometry.gate_capacitance(role)
            # A curated-backed axis takes its unit from the curated registry, so a
            # curated_units override reaches the axis label rather than only the
            # registry's own view.  _SWEEP_TYPES spells a unit only for the axes
            # with no curated row behind them; a None beside a source that is not
            # a registry key is a table error, and the KeyError names it.
            if default_unit is None:
                default_unit = self._curated[source][2]
            kind = ("index", None) if source is None else ("curated", source)
            return (sweep, kind,
                    label if label is not None else default_label,
                    unit  if unit  is not None else default_unit)

        if sweep in self.parameters:
            # A raw row used directly as the axis.  The file does not state its
            # unit, so the label defaults to the row name and the unit is blank
            # until the caller supplies one.
            return (sweep, ("row", sweep),
                    label if label is not None else sweep,
                    unit  if unit  is not None else "")

        raise ValueError(
            f"{param}={sweep!r} is neither a known sweep type nor a parameter row "
            f"in '{self.path}'.\n"
            f"  Sweep types : {sorted(_SWEEP_TYPES)}\n"
            f"  File rows   : {self.parameter_labels}"
        )

    # --- The parameter store -----------------------------------------------

    def get_parameter(self, label: str, scale: float = 1.0) -> np.ndarray:
        """
        Return the per-sweep values for *label* as a NumPy array.

        Parameters
        ----------
        label : str
            Exact row label as it appears in the CSV, e.g. ``"Galvo_X"``,
            ``"Scanner Y"``, ``"Excitation Power"``.  See
            :attr:`parameter_labels` for the available names.
        scale : float
            Multiplicative factor applied to the raw values (e.g. unit
            conversion).  Default ``1.0`` (raw units, as stored in the file).

        Returns
        -------
        np.ndarray, shape (n_sweeps,)

        Raises
        ------
        KeyError
            If *label* is not present, with the available labels listed.
        """
        if label not in self.parameters:
            raise KeyError(
                f"Parameter '{label}' not found in CSV rows. "
                f"Available: {self.parameter_labels}"
            )
        return self.parameters[label] * scale

    def _curated_value(self, name: str) -> np.ndarray:
        """Return a curated quantity as a scaled view into :attr:`parameters`."""
        label, scale, _ = self._curated[name]
        return self.get_parameter(label, scale)

    def _curated_or_none(self, name: str) -> np.ndarray:
        """A curated quantity, or ``None`` if its row is absent from this file."""
        label = self._curated[name][0]
        return self._curated_value(name) if label in self.parameters else None

    # --- Curated parameter properties (scaled views into self.parameters) ---

    @property
    def gates(self) -> dict:
        """
        Which parameter row reached each electrode, or ``None`` if undeclared.

        As given to *gates* at load time, in canonical role order. The roles
        present describe the device — ``{"top", "bottom"}`` a dual-gated stack,
        ``{"bottom", "channel"}`` a bottom-gated one with a contacted TMDC — and a
        value of ``None`` means that electrode is tied to ground with no parameter
        row recording it.

        ``None`` means the wiring was never stated, in which case :attr:`v_top`,
        :attr:`v_bot`, :attr:`v_channel`, :attr:`ef` and the gate sweep types all
        raise.
        """
        return dict(self._gates) if self._gates is not None else None

    @property
    def is_dual_gated(self) -> bool:
        """
        Whether *gates* declared both gate electrodes, so a field is defined.

        ``False`` both for a single-gated device and for a scan whose wiring was
        never declared — in neither case can :attr:`ef` be computed.
        """
        return (self._gates is not None
                and all(role in self._gates for role in _GATE_ELECTRODES))

    def _require_gates(self, what: str) -> None:
        """Raise unless the electrode mapping was declared at all."""
        if self._gates is not None:
            return
        raise ValueError(
            f"{what} is defined per electrode, but which acquisition channel "
            f"reached which electrode was not declared for '{self.path}'. The "
            f"electrodes can be wired either way round and no export records "
            f"which, so transposing them mirrors the field axis and flips the sign "
            f"of any extracted dipole. Pass gates={{'top': '<row>', 'bottom': "
            f"'<row>'}} for a dual-gated device, or gates={{'bottom': '<row>', "
            f"'channel': '<row>'}} for a single-gated one; candidate rows in this "
            f"file: {self._gate_candidates()}. To work in channel terms instead, "
            f"read the row directly — scan['<row>'], or sweep='<row>' for an axis."
        )

    def _require_role(self, role: str, what: str) -> None:
        """Raise unless *role* is one of the electrodes this device declared."""
        self._require_gates(what)
        if role in self._gates:
            return
        # The device does not have this electrode.  For a gate that means there is
        # no gate-to-gate potential difference and so no displacement field; the
        # single knob such a device has controls carrier density instead.
        extra = (" A single-gated device has no gate-to-gate potential difference "
                 "and so no displacement field: carrier density is the quantity "
                 "it controls."
                 if role in _GATE_ELECTRODES and not self.is_dual_gated else "")
        raise ValueError(
            f"{what} needs the {role!r} electrode, which '{self.path}' does not "
            f"have: gates declared {sorted(self._gates)}.{extra}"
        )

    def _gate_value(self, role: str) -> np.ndarray:
        """
        The voltage on a declared electrode, in V.

        A role declared as ``None`` is tied to ground with no row recording it, so
        its voltage is zero at every sweep point — that is what the declaration
        says, not an assumption about a missing row.
        """
        label = self._gates[role]
        if label is None:
            return np.zeros(self.n_sweeps)
        attr = _GATE_ROLE_CURATED.get(role)
        # Gate electrodes go through the curated registry so a curated_scales
        # override still applies; the channel has no curated entry of its own.
        return (self._curated_value(attr) if attr is not None
                else self.get_parameter(label))

    def _role_current(self, role: str) -> np.ndarray:
        """
        The current at a declared electrode, in nA.

        Unlike :meth:`_gate_value`, a role declared ``None`` raises: grounding an
        electrode is what holds its potential at zero, but it says nothing about the
        current flowing through it, which is merely unrecorded.
        """
        row = self._gates[role]
        if row is None:
            raise ValueError(
                f"i_{role} needs a recorded current, but the {role!r} electrode was "
                f"declared as tied to ground with no row for it. Grounding fixes "
                f"that electrode's potential, not its current — the current still "
                f"flows and simply was not measured, so there is no value to "
                f"return. Declare the row that drove it if one exists: "
                f"gates={{..., '{role}': '<row>'}}."
            )
        attr  = _ROLE_CURRENT_CURATED[role]
        label = self._curated[attr][0]
        if label is None:
            raise ValueError(
                f"i_{role} is not available: the {role!r} electrode is declared on "
                f"row '{row}', which is not a source-meter channel, so no row "
                f"records the current at that terminal. Channels carrying both a "
                f"bias and a current in this format: "
                f"{sorted(_CHANNEL_SIBLING_CURRENT)}. The voltage is unaffected."
            )
        return self._curated_value(attr)

    def _gate_candidates(self) -> list:
        """
        Rows that plausibly carry an electrode voltage, for the undeclared error.

        The conventional gate rows first if this file has them, then any other row
        that varied — a row held at a constant 0 V is a grounded electrode and
        cannot be told from an unused channel, so it is not proposed.
        """
        varying = self.varying_parameters()
        default = [self._CURATED[name][0] for name in _GATE_CURATED
                   if self._CURATED[name][0] in self.parameters]
        return default + [lbl for lbl in varying if lbl not in default]

    @property
    def v_top(self) -> np.ndarray:
        """
        Top gate voltage in V (per sweep).

        Raises ``ValueError`` unless *gates* declared a top gate and which row
        reached it; see :attr:`gates`.
        """
        self._require_role("top", "v_top")
        return self._gate_value("top")

    @property
    def v_bot(self) -> np.ndarray:
        """
        Bottom gate voltage in V (per sweep).

        Raises ``ValueError`` unless *gates* declared a bottom gate and which row
        reached it; see :attr:`gates`.
        """
        self._require_role("bottom", "v_bot")
        return self._gate_value("bottom")

    @property
    def v_channel(self) -> np.ndarray:
        """
        Voltage on the contact to the TMDC itself, in V (per sweep).

        Zero throughout for a grounded channel, which is the usual case. Raises
        ``ValueError`` unless *gates* declared a ``"channel"`` role.
        """
        self._require_role("channel", "v_channel")
        return self._gate_value("channel")

    @property
    def power(self) -> np.ndarray:
        """Excitation power in µW (per sweep)."""
        return self._curated_value("power")

    @property
    def i_top(self) -> np.ndarray:
        """
        Leakage current at the top gate in nA (per sweep).

        Raises ``ValueError`` unless *gates* declared a top gate on a source-meter
        channel; see :attr:`gates`.
        """
        self._require_role("top", "i_top")
        return self._role_current("top")

    @property
    def i_bot(self) -> np.ndarray:
        """
        Leakage current at the bottom gate in nA (per sweep).

        Raises ``ValueError`` unless *gates* declared a bottom gate on a source-meter
        channel; see :attr:`gates`.
        """
        self._require_role("bottom", "i_bot")
        return self._role_current("bottom")

    @property
    def i_channel(self) -> np.ndarray:
        """
        Current sourced into the contact on the TMDC itself, in nA (per sweep).

        Transport through the flake rather than leakage across a dielectric, which
        is what distinguishes it from :attr:`i_top` / :attr:`i_bot`.

        Raises ``ValueError`` unless *gates* declared a ``"channel"`` role on a
        source-meter channel — including when the contact is declared grounded with
        no row, since that fixes its potential but leaves its current unmeasured.
        """
        self._require_role("channel", "i_channel")
        return self._role_current("channel")

    @property
    def scanner_x(self) -> np.ndarray:
        """Piezo scanner X drive voltage in V (per sweep), not a distance."""
        return self._curated_value("scanner_x")

    @property
    def scanner_y(self) -> np.ndarray:
        """Piezo scanner Y drive voltage in V (per sweep), not a distance."""
        return self._curated_value("scanner_y")

    @property
    def ef(self) -> np.ndarray:
        """
        Displacement field in mV/nm (per sweep), or ``None`` if no
        :class:`DeviceGeometry` was supplied.  Computed from the curated
        :attr:`v_top` / :attr:`v_bot`.

        Raises ``ValueError`` when a geometry *was* supplied but the device is not
        :attr:`is_dual_gated` — an undeclared wiring leaves the field's sign
        undefined, and a single gate has no field at all. Saying no field was
        computed needs neither, so the ungated case still returns ``None``.
        """
        if self.geometry is None:
            return None
        # Named explicitly so the error says "ef", not "v_top" — the caller asked
        # for a field and the reason it cannot have one is about the device.
        for role in _GATE_ELECTRODES:
            self._require_role(role, "ef")
        return self.geometry.electric_field(self.v_top, self.v_bot)

    @property
    def carrier_density(self) -> np.ndarray:
        """
        Gate-induced sheet carrier density in cm⁻² (per sweep), or ``None`` if no
        :class:`DeviceGeometry` was supplied.

        Summed over the gate electrodes *gates* declared, referenced to zero gate
        voltage — so this is the density induced **relative to 0 V**, not an
        absolute density, which needs a threshold no instrument file records. Call
        :meth:`DeviceGeometry.carrier_density` directly with *v_ref* to reference it
        elsewhere, or to a measured threshold.

        Raises ``ValueError`` unless *gates* declared a ``"channel"`` role: a
        density is defined against the contact that supplies the charge. Warns when
        that contact's row varies across the sweep, since the reference then moves
        with the axis and is not accounted for.
        """
        if self.geometry is None:
            return None
        self._require_role("channel", "carrier_density")

        # The density is referenced to the contact, so a contact that is itself
        # being driven moves the reference under the axis.  Legitimate for a
        # source-drain bias measurement, wrong for a doping sweep, and the file
        # cannot tell which — so say what was seen rather than pick.
        channel_row = self._gates["channel"]
        if channel_row is not None and channel_row in self.varying_parameters():
            span = float(np.ptp(self.parameters[channel_row]))
            warnings.warn(
                f"carrier_density references the channel contact, but its row "
                f"'{channel_row}' varies by {span:.4g} V across the sweep, so the "
                f"reference moves with the axis. The returned density is relative "
                f"to 0 V on the gates alone and does not account for it. Expected "
                f"for a source-drain bias; for a doping sweep the channel should "
                f"be at a fixed potential.",
                UserWarning, stacklevel=2,
            )

        # One kwarg per declared gate, keyed by the same v_top / v_bot names
        # DeviceGeometry takes; a gate the device lacks is left out of the sum
        # rather than passed as zero.
        volts = {_GATE_ROLE_CURATED[role]: self._gate_value(role)
                 for role in _GATE_ELECTRODES if role in self._gates}
        return self.geometry.carrier_density(**volts)

    @property
    def curated_parameters(self) -> dict:
        """
        Mapping ``attr -> (csv_label, scale, unit)`` for the curated quantities.

        Documents which CSV rows are promoted to first-class properties, the
        scale applied (e.g. raw amps → nA), and the resulting unit — making the
        raw-vs-scaled distinction with :attr:`parameters` explicit.

        The unit is also what a curated-backed sweep axis is labelled with, so
        this is where :attr:`sweep_axis_label` gets "V" or "µW" from.
        """
        return dict(self._curated)

    @property
    def parameter_labels(self) -> list:
        """Sorted list of every parameter label available via :attr:`parameters`."""
        return sorted(self.parameters)

    # --- Shape -------------------------------------------------------------

    @property
    def n_sweeps(self) -> int:
        """Number of sweep points."""
        return getattr(self, self._SIGNAL_ATTR).shape[1]

    @property
    def n_points(self) -> int:
        """Length of the measured axis — detector pixels, or time bins."""
        return getattr(self, self._AXIS_ATTR).size

    @property
    def n_declared_sweeps(self) -> int:
        """
        Sweep points the export *declared*.

        Exceeds :attr:`n_sweeps` when the acquisition software over-allocated its
        header and zero-filled the surplus blocks — see
        :func:`_drop_unwritten_blocks`.
        """
        return self._n_declared if self._n_declared is not None else self.n_sweeps

    # --- The sweep axis ----------------------------------------------------

    @property
    def sweep_axis(self) -> np.ndarray:
        """
        The declared sweep axis, shape ``(n_sweeps,)``.

        A registry sweep returns its curated quantity (converted, e.g. a field in
        mV/nm); a raw-row sweep returns that row in file units; an undeclared
        sweep returns ``arange(n_sweeps)`` — never a guess at which parameter was
        meant.  See :meth:`varying_parameters`.
        """
        return self._axis_for_source(self._sweep_source)

    def _axis_for_source(self, source: tuple) -> np.ndarray:
        """Read the ``(n_sweeps,)`` array a ``(kind, name)`` source points at."""
        kind, name = source
        if kind == "index":
            return np.arange(self.n_sweeps, dtype=float)
        if kind == "row":
            return self.get_parameter(name)
        return getattr(self, name)

    @property
    def sweep_axis_label(self) -> str:
        """Axis label for :attr:`sweep_axis`, with its unit when one is known."""
        if self._sweep_unit:
            return f"{self._sweep_label} ({self._sweep_unit})"
        return self._sweep_label

    @property
    def sweep_label(self) -> str:
        """Bare axis label for :attr:`sweep_axis`, without the unit."""
        return self._sweep_label

    @property
    def sweep_unit(self) -> str:
        """Unit of :attr:`sweep_axis`; empty when the file does not state one."""
        return self._sweep_unit

    # --- The declared nest ---------------------------------------------------

    @property
    def nesting(self) -> SweepNesting:
        """
        The declared 2-D nest, or ``None`` when the sweep is flat.

        Carries both coordinate axes with their labels and units.  A nest is
        declared with ``fast_sweep=`` and ``slow_sweep=`` at load time and is
        never inferred; :meth:`sweep_grid` reports what a file looks like it
        contains, which is the diagnostic for deciding what to declare.
        """
        return self._nesting

    @property
    def is_nested(self) -> bool:
        """Whether a 2-D nest was declared — the predicate :meth:`as_grid` needs."""
        return self._nesting is not None

    def as_grid(self, array: np.ndarray) -> np.ndarray:
        """
        Reshape a flat-sweep array onto the declared nest.

        Parameters
        ----------
        array : np.ndarray
            Either a signal array of shape ``(n_points, n_sweeps)`` — the spectra,
            any corrected variant of them, the decays — or a per-sweep-point row
            of shape ``(n_sweeps,)``, such as anything from :attr:`parameters`.

        Returns
        -------
        np.ndarray
            ``(n_points, n_slow, n_fast)`` for a signal array,
            ``(n_slow, n_fast)`` for a per-point row.  The fast axis is last,
            because the sweep was written with it running fastest.

        Raises
        ------
        ValueError
            If no nest was declared, or if *array* does not have ``n_sweeps`` as
            its trailing dimension.

        Notes
        -----
        A **view**, not a copy: the flat sweep is already in the right memory
        order, so nothing moves.  It therefore shares storage with the array it
        came from, and the never-mutate-after-load rule reaches it.

        Examples
        --------
        >>> cube = scan.as_grid(scan.spectra)            # doctest: +SKIP
        >>> cube.shape                                   # doctest: +SKIP
        (1340, 51, 41)
        >>> scan.as_grid(scan["Scanner X"]).shape        # doctest: +SKIP
        (51, 41)

        See Also
        --------
        tmdc_optics_tools.processing.reorder_grid : flatten a grid built by
            this method back into a sequence, in a chosen traversal order
            and direction rather than only the one it was written in.
        """
        nest  = self._require_nesting("as_grid()")
        array = np.asarray(array)
        n     = self.n_sweeps

        if array.ndim not in (1, 2) or array.shape[-1] != n:
            raise ValueError(
                f"as_grid() takes an array whose last axis is the {n} sweep "
                f"points — either (n_points, {n}) or ({n},) — and got shape "
                f"{array.shape}. An array indexed some other way has no nest to "
                f"be put onto."
            )
        return array.reshape(array.shape[:-1] + nest.shape)

    def as_image_grid(self, image_scan) -> np.ndarray:
        """
        Reshape an image sequence's frames onto this sweep's declared nest.

        :meth:`as_grid` only accepts a 1-D or 2-D array; an image stack is
        3-D, so this flattens it to ``(height * width, n_frames)`` first and
        reshapes back after. :meth:`as_grid`'s own frame-count check still
        does the real work: this raises exactly the error it would for a
        spectral array if ``image_scan.n_frames != self.n_sweeps`` — not a
        second, differently-worded one written just for images.

        Parameters
        ----------
        image_scan : object exposing ``n_frames`` and ``load_frame(idx)``
            E.g. :class:`AttoCubePLScanRealSpace`, or anything else
            :class:`~tmdc_optics_tools.plotting.ImageSequencePanel` accepts.

        Returns
        -------
        np.ndarray
            ``(height, width, n_slow, n_fast)``.

        Raises
        ------
        ValueError
            If no nest was declared, or if ``image_scan.n_frames`` does not
            equal :attr:`n_sweeps` — see :meth:`as_grid`.
        """
        frames = [image_scan.load_frame(i) for i in range(image_scan.n_frames)]
        height, width = frames[0].shape
        flat = np.stack(frames, axis=-1).reshape(height * width, image_scan.n_frames)
        return self.as_grid(flat).reshape(height, width, *self.nesting.shape)

    # --- Locating a sweep point ----------------------------------------------

    def _lookup_axis(self, axis: str, param : str = "axis") -> tuple:
        """
        Return ``(values, label, unit)`` for an axis name.

        ``"sweep"`` is the declared sweep axis; ``"fast"`` and ``"slow"`` are the
        nest coordinates.  Anything else resolves the way ``sweep=`` does — a
        registry key or a raw row label — so any per-sweep-point quantity can be
        looked up, whether or not it is what the sweep was declared with.

        *param* is the name of the argument being resolved, interpolated into
        every message so the error names the argument the caller actually passed.
        """
        if axis == "sweep":
            return self.sweep_axis, self.sweep_label, self.sweep_unit
        if axis in ("fast", "slow"):
            nest = self._require_nesting(f"{param}={axis!r}")
            return (getattr(nest, f"{axis}_axis"),
                    getattr(nest, f"{axis}_label"),
                    getattr(nest, f"{axis}_unit"))
        try:
            _, source, label, unit = self._resolve_sweep(
                axis, None, None, param=param)
        except ValueError as exc:
            raise ValueError(
                f"{exc}\n  Also accepted : 'sweep' (the declared sweep axis), "
                f"'fast' and 'slow' (the nest coordinates)."
            ) from None
        return self._axis_for_source(source), label, unit

    def nearest_index(self, value: float, axis: str = "sweep") -> int:
        """
        Position along *axis* of the point closest to *value*.

        Parameters
        ----------
        value : float
            The coordinate wanted, in the axis's own units.
        axis : str
            Which axis to search.  ``"sweep"`` is the flat sweep axis, so the
            result indexes the columns of :attr:`spectra` directly.  ``"fast"``
            and ``"slow"`` are the nest coordinates and need a declared nest.
            Any other name is a quantity to search *instead* of the declared
            axis, spelled as ``sweep=`` spells it — a registry key such as
            ``"top_voltage"`` or ``"power"``, or a raw row label such as
            ``"V_A"``.  Useful when a sweep is declared in one coordinate and
            you want to locate a point by another: a field sweep driven by both
            gates at a fixed ratio can be searched by ``"top_voltage"``.

        Returns
        -------
        int
            An index into that axis — always a single position, never a set.

        Warns
        -----
        UserWarning
            When the closest point is further than half the axis's median step
            from *value*.  A nearest-value lookup cannot fail, so asking for a
            coordinate the scan does not hold returns a real spectrum from
            somewhere else; the warning names what was asked for and what was
            used.  A request landing between two real points is silent.
        UserWarning
            When the value found occurs at more than one sweep point, so the
            request does not identify one.  Only a quantity that changes
            monotonically along the sweep is guaranteed not to: a hysteresis
            loop passes the same gate voltage twice.
        """
        idx, values, label, unit, matches = self._nearest(value, axis, depth=4)
        if matches.size > 1:
            warnings.warn(
                f"{label} holds {float(values[idx]):.6g}"
                f"{f' {unit}' if unit else ''} at {_render_indices(matches)}, so "
                f"looking it up by value does not identify one; using index "
                f"{idx}. Address these by index, or search a quantity that "
                f"changes monotonically along the sweep.",
                UserWarning, stacklevel=3,
            )
        return idx

    def _nearest(self, value: float, axis: str, depth: int = 3) -> tuple:
        """
        Locate *value* on *axis*, warning when nothing lies near it.

        Returns ``(idx, values, label, unit, matches)``, *matches* being every
        index sharing the coordinate found — one element when the lookup is
        unambiguous.  What to do about more than one is the caller's policy:
        :meth:`nearest_index` warns because a single index is its whole contract,
        while an accessor returning spectra refuses, since dropping all but one
        of them would be a silent partial answer.
        """
        values, label, unit = self._lookup_axis(axis)
        value = float(value)

        # inf where the axis is not finite, so a NaN row cannot win the argmin.
        distance = np.where(np.isfinite(values), np.abs(values - value), np.inf)
        if not np.isfinite(distance).any():
            raise ValueError(
                f"The {axis!r} axis holds no finite values, so there is no "
                f"nearest point to {value:.6g}."
            )
        idx   = int(np.argmin(distance))
        found = float(values[idx])

        finite = values[np.isfinite(values)]
        step   = (float(np.median(np.abs(np.diff(np.sort(finite)))))
                  if finite.size > 1 else 0.0)
        suffix = f" {unit}" if unit else ""

        # Half a typical gap is the distance at which a request has clearly missed a
        # grid point.  On an undriven axis those gaps are scatter-sized, so that alone
        # would report a value as absent when it is the only value the axis holds; the
        # tolerance floors it, and on a driven axis it is far the smaller of the two
        # and changes nothing.
        if abs(found - value) > max(0.5 * step, _axis_atol(values)):
            warnings.warn(
                f"Looking up {value:.6g}{suffix} on {label} (axis={axis!r}) found "
                f"no point there; using index {idx} at {found:.6g}{suffix}, which "
                f"is {abs(found - value):.6g}{suffix} away. The axis spans "
                f"{finite.min():.6g} to {finite.max():.6g}{suffix} in "
                f"{finite.size} points.",
                UserWarning, stacklevel=depth,
            )

        # A quantity that is not what the sweep was ordered by need not label its
        # points individually — a hysteresis loop passes the same gate voltage
        # twice, and every quantity of a nest repeats.  Compared on the
        # *coordinate* rather than the distance, so a request landing midway
        # between two distinct points is not a tie.
        matches = np.flatnonzero(np.abs(values - found) <= _axis_atol(values))
        return idx, values, label, unit, matches

    def _index_for_value(self, value: float, axis: str, what: str) -> int:
        """
        Locate *value* on *axis*, refusing when it does not identify one point.

        The accessors' policy.  Where :meth:`nearest_index` owes the caller an
        integer and can only warn, an accessor has a complete answer to point at:
        a declared nest addresses every match at once through ``fast=`` / ``slow=``.
        """
        idx, values, label, unit, matches = self._nearest(value, axis, depth=5)
        if matches.size == 1:
            return idx

        grid = self.sweep_grid()
        if matches.size == self.n_sweeps:
            # Every point matched, so the axis holds one setting and nothing to select
            # between.  A nest is the wrong suggestion here: a row that never moved is
            # not an axis of one.
            fix = (" This row was held at one setting for the whole file, so no value "
                   "on it picks out a spectrum. varying_parameters() lists the rows "
                   "that did move.")
        elif grid is not None:
            fix = (f" This file looks like {grid} — declare it with "
                   f"fast_sweep='{grid.fast_label}', slow_sweep='{grid.slow_label}', "
                   f"and fast= / slow= then return every spectrum at a coordinate "
                   f"rather than one of them.")
        else:
            fix = (" If these points are a nest, declare it with fast_sweep= and "
                   "slow_sweep=, and address it with fast= / slow=.")
        raise ValueError(
            f"{what}: {label} holds {float(values[idx]):.6g}"
            f"{f' {unit}' if unit else ''} at {_render_indices(matches)}, so it "
            f"does not identify one spectrum.{fix} To take a single one of them, "
            f"use get_spectrum_by_index()."
        )

    def _positional_index(self, index, axis: str, length: int) -> int:
        """Validate an integer position, accepting Python-style negatives."""
        try:
            index = int(index)
        except (TypeError, ValueError):
            raise TypeError(
                f"A {axis!r} index must be an integer, got {index!r}. To look up "
                f"by value instead, use the *_at accessor."
            ) from None
        if not -length <= index < length:
            raise IndexError(
                f"{axis!r} index {index} is out of range for {length} points."
            )
        return index % length

    def _sweep_selector(self, value=None, *, axis: str = "sweep",
                        fast=None, slow=None, by_value: bool, what: str):
        """
        Resolve a point request into an ``int`` or ``slice`` over the sweep axis.

        An ``int`` drops the sweep dimension, a ``slice`` keeps it — and both are
        basic indexing, so the signal array is viewed rather than copied.  For a
        nest, pinning the slow axis takes one contiguous run of columns while
        pinning the fast axis strides across every run.
        """
        # By value the accessors refuse an ambiguous coordinate rather than
        # returning one of the points it names; see _index_for_value.
        locate = ((lambda v, ax: self._index_for_value(v, ax, what)) if by_value
                  else (lambda v, ax: self._positional_index(
                      v, ax, len(self._lookup_axis(ax)[0]))))
        arg = "value" if by_value else "index"

        # axis= says what the positional argument is read against, so it is an
        # alternative to naming the nest axes rather than a modifier of them.
        if axis != "sweep" and (fast is not None or slow is not None):
            raise ValueError(
                f"{what}: axis={axis!r} names the quantity the positional {arg} "
                f"is read against, so it cannot be combined with fast= or "
                f"slow=. To address a nest by another quantity, declare the nest "
                f"in it: fast_sweep={axis!r}."
            )

        if not self.is_nested:
            if fast is not None or slow is not None:
                raise ValueError(
                    f"{what}: fast= and slow= need a declared nest, and this "
                    f"sweep is flat ({self.n_sweeps} points). Give the {arg} "
                    f"positionally to index the sweep axis, or declare the nest "
                    f"with fast_sweep= and slow_sweep= at load time."
                )
            if value is None:
                raise ValueError(f"{what} needs a {arg}.")
            return locate(value, axis)

        nest = self._nesting
        if value is not None:
            raise ValueError(
                f"{what}: this sweep is a declared nest ({nest}), so a single "
                f"{arg} does not locate a point. Name the axes: fast= and/or "
                f"slow=."
            )
        if fast is None and slow is None:
            raise ValueError(
                f"{what}: name fast= and/or slow=. Naming both gives one "
                f"spectrum, naming one gives the line of spectra along the "
                f"other. For the whole grid at once, use as_grid()."
            )

        n_fast = nest.n_fast
        if fast is not None and slow is not None:
            return locate(slow, "slow") * n_fast + locate(fast, "fast")
        if slow is not None:
            start = locate(slow, "slow") * n_fast
            return slice(start, start + n_fast)
        return slice(locate(fast, "fast"), None, n_fast)

    def _require_nesting(self, what: str) -> SweepNesting:
        """Return the nest, or raise naming what needed it."""
        if self._nesting is None:
            detected = self.sweep_grid()
            hint = (f" This file looks like {detected}; declare it with "
                    f"fast_sweep= and slow_sweep=."
                    if detected is not None else
                    " Declare one with fast_sweep= and slow_sweep= at load time.")
            raise ValueError(
                f"{what} needs a declared nest, and this sweep is flat "
                f"({self.n_sweeps} points).{hint}"
            )
        return self._nesting

    # --- What the measurement is -------------------------------------------

    @property
    def signal_name(self) -> str:
        """Name of the measured signal, from :attr:`spectra_type`."""
        return SIGNAL_LABELS[self.spectra_type][0]

    @property
    def signal_unit(self) -> str:
        """
        Native unit of :attr:`signal_name`; empty for dimensionless ratios.

        Exposed separately from :attr:`signal_label` so a caller that rescales
        the data can substitute its own unit rather than parse one out of a
        composed string.
        """
        return SIGNAL_LABELS[self.spectra_type][1]

    @property
    def signal_label(self) -> str:
        """
        Y-axis label for the measured signal, e.g. ``"PL intensity (counts)"``.

        Use this rather than hardcoding "PL" in a plot that may one day be
        handed reflectance; the unit is omitted for dimensionless ratios.
        """
        name, unit = SIGNAL_LABELS[self.spectra_type]
        return f"{name} ({unit})" if unit else name

    @property
    def spectroscopy(self) -> str:
        """Human-readable :attr:`spectra_type`, e.g. ``"Photoluminescence"``."""
        return SPECTROSCOPY_TYPES[self.spectra_type]

    # --- What actually varied ----------------------------------------------

    def varying_parameters(self, rtol: float = 1e-3) -> dict:
        """
        Report which parameter rows actually changed across the sweep.

        Evidence for choosing a *sweep*, and the check for "was only one gate
        driven?".  A row counts as varying on any of three signs: its span exceeds
        *rtol* times its own RMS magnitude, or its readings step through that span a
        small part at a time in acquisition order, or those readings only ever move
        one way.  Directionless jitter on a nominally static channel satisfies none
        of the three, so it does not register; a fine sweep sitting on a large offset
        satisfies the second or the third even though it fails the first.  RMS rather
        than mean, so that a row straddling zero — an anti-symmetric gate pair, say —
        is measured by how large it is rather than by how nearly it cancels.

        Membership is decided by the same helper the loader groups readings with, so
        this report and the tolerance behind :meth:`get_spectrum_at` cannot disagree
        about which rows are only jittering.  A held row whose read-back *drifts*
        rather than jitters therefore registers here, at any amplitude.

        Parameters
        ----------
        rtol : float
            Relative span above which the **first** sign fires.  Default ``1e-3``.
            It does not reach the other two, which are fixed: a row satisfying either
            of them is reported whatever *rtol* is set to.  Raising it therefore
            narrows the report only among rows that are recognised by their span
            alone.

        Returns
        -------
        dict
            ``label -> (min, max, span)``, ordered by span relative to RMS,
            largest first.

        Notes
        -----
        The ordering ranks how much a channel moved relative to its own
        magnitude; it does **not** identify the swept quantity, and should not be
        read as doing so.  A small channel that swings across its whole range —
        a leakage current, or a gate driven symmetrically about zero — outranks a
        large one stepped through part of its range, however clearly the second
        is the axis of the experiment.  This returns the evidence; which axis was
        swept is the caller's to declare via ``sweep=``.

        Examples
        --------
        >>> scan.varying_parameters()
        {'V_A': (0.0, 1.0, 1.0), 'V_B': (-1.0, 0.0, 1.0), 'Scanner Y': ...}
        """
        found = []
        for label, arr in self.parameters.items():
            finite = arr[np.isfinite(arr)]
            if finite.size < 2:
                continue
            span = float(np.ptp(finite))
            # Ranked by span relative to the row's own magnitude.  RMS rather than
            # |mean|, because a row can sit astride zero — an anti-symmetric gate
            # pair sweeping the field does, and is routine — and its mean is then no
            # measure of how large it is.  Flooring a vanishing mean at finfo.tiny
            # turned the division by zero into a division by 2.2e-308, i.e. into
            # inf.  RMS cannot vanish unless the row is identically zero, which a
            # zero span already excludes; the floor below is belt and braces.
            scale = max(float(np.sqrt(np.mean(finite ** 2))),
                        np.finfo(float).tiny)
            # Membership comes from the same helper the loader groups readings with,
            # so this report and the tolerance behind get_spectrum_at() cannot
            # disagree about which rows are only jittering.  It is wider than the
            # ranking: a fine sweep on a large offset fails span-against-RMS and is
            # still caught by the other two signs, which the ranking below then places
            # low, correctly.
            if _axis_driven(arr, rtol):
                found.append((span / scale, label,
                              (float(finite.min()), float(finite.max()), span)))
        found.sort(reverse=True)
        return {label: values for _, label, values in found}

    @property
    def gate_mode(self) -> str:
        """
        How the gates were driven, read off the data.

        ``None`` when no gate row is available to look at.  Purely descriptive:
        the correlation sign distinguishes an anti-symmetric (field-like) sweep
        from a symmetric (doping-like) one.

        Never raises, and never needs *gates*: how many channels moved together is
        a property of the data, and the correlation is symmetric in the two rows,
        so the verdict is unchanged by transposing them. Only the wording differs —
        with a declared mapping a single driven gate is named by its role
        (``"bottom-gate only"``), and without one by its channel
        (``"single gate driven ('V_B')"``), because which electrode that channel
        reached is not in the file.

        A single-gated device reports on its one gate; the channel contact is not
        a gate and is not included.
        """
        # Read the rows straight from the store rather than through v_top / v_bot,
        # which require a declared mapping this property deliberately does without.
        # A grounded electrode (declared None) has no row and cannot be described.
        if self._gates is not None:
            rows = {role: self._gates[role] for role in _GATE_ELECTRODES
                    if self._gates.get(role) is not None}
        else:
            rows = {role: self._CURATED[attr][0]
                    for role, attr in _GATE_ROLE_CURATED.items()}
        rows = {role: lbl for role, lbl in rows.items() if lbl in self.parameters}
        if not rows:
            return None

        varying = self.varying_parameters()
        driven  = [role for role, lbl in rows.items() if lbl in varying]

        if not driven:
            return "gates static"

        # One gate available or one gate driven: describe that gate alone.  Naming
        # its role needs the declaration; without one, name the channel, which is
        # true whatever the wiring.
        if len(rows) == 1 or len(driven) == 1:
            role = driven[0]
            if self._gates is None:
                return f"single gate driven ('{rows[role]}')"
            return f"{role}-gate only"

        if self.n_sweeps < 3:
            return "dual-gate"
        first, second = (rows[role] for role in driven[:2])
        # Pearson coefficient of the two gate channels.  [0, 1] is the cross term;
        # it is symmetric in the two, so the sign does not depend on the wiring.
        r = float(np.corrcoef(self.parameters[first],
                              self.parameters[second])[0, 1])
        if r < -0.95:
            return "dual-gate, anti-correlated (field-like)"
        if r > 0.95:
            return "dual-gate, correlated (doping-like)"
        return "dual-gate, independent"

    def sweep_grid(self, rtol: float = 1e-3) -> SweepGrid:
        """
        Detect a 2-D raster flattened into the sweep axis, or return ``None``.

        A spatial map arrives as one long sweep: the reflectance example is 41
        ``Scanner X`` positions nested inside 51 ``Scanner Y`` positions, 2091
        points in all.  This reports that shape so it is visible in
        :func:`repr`; it does **not** reshape anything, and the sweep axis is
        unaffected.

        The signature looked for is strong: two varying rows whose distinct-value
        counts multiply to exactly :attr:`n_sweeps`, with the inner row taking
        every one of its values before repeating.

        Returns
        -------
        SweepGrid or None
            ``(fast_label, n_fast, slow_label, n_slow)``.  ``None`` when the
            sweep is not a raster — the normal case for a gate or power sweep.

        See Also
        --------
        nesting : the nest actually declared, which is what reshaping uses.

        Notes
        -----
        Reports; does not decide.  Both axes are looked for among the raw
        parameter rows, so a nest whose axis is a *derived* quantity — an
        anti-symmetric gate pair sweeping the field, say — is reported through
        the channels that carry it rather than through the field.  Declare the
        nest with ``fast_sweep=`` / ``slow_sweep=``, which accept the same
        vocabulary as ``sweep=``.
        """
        n = self.n_sweeps
        if n < 4: # Checks if less than 4 points -- not a grid sweep
            return None

        counts = {}
        for label in self.varying_parameters(rtol=rtol):
            arr = self.parameters[label]
            # Find finite values, then find unique values, then count the unique values
            n_unique = np.unique(arr[np.isfinite(arr)]).size
            if 1 < n_unique < n:
                # If n_unique is more than 1 but less than n, it could be a sweep grid
                counts[label] = n_unique

        for fast, n_fast in counts.items():
            for slow, n_slow in counts.items():
                if fast == slow or n_fast * n_slow != n:
                    # Same row twice, or the two counts do not account for every
                    # sweep point: not this pair.
                    continue
                values = self.parameters[fast]
                # The fast axis runs through all its values, then starts over.
                # The first n_fast values holding every distinct value with no
                # repeat, and value n_fast returning to the first, is that.
                if (np.unique(values[:n_fast]).size == n_fast
                        and np.isclose(values[0], values[n_fast])):
                    return SweepGrid(fast, n_fast, slow, n_slow)
        return None

    # --- Export ------------------------------------------------------------

    def to_hdf5(self, path, **kwargs):
        """
        Write this sweep to a self-describing HDF5 file.

        Thin delegation to :func:`tmdc_optics_tools.hdf5.write_sweep`; see there
        for the layout and for what is deliberately *not* stored.

        Parameters
        ----------
        path : str or Path
            Destination ``.h5`` file.
        **kwargs
            Forwarded to :func:`~tmdc_optics_tools.hdf5.write_sweep`
            (``compression``, ``overwrite``).

        Returns
        -------
        pathlib.Path
            The file written.
        """
        from . import hdf5 as _hdf5
        return _hdf5.write_sweep(self, path, **kwargs)

    # --- Dunder methods ----------------------------------------------------

    def __getitem__(self, label: str) -> np.ndarray:
        """Sugar for :meth:`get_parameter`.
        Makes scan.get_parameter('V_A') equivalent to scan['V_A'].
        """
        return self.get_parameter(label)

    def _repr_axis_lines(self, w: int) -> list:
        """Subclass hook: lines describing the measured axis."""
        return []

    def _repr_extra_lines(self, w: int) -> list:
        """Subclass hook: lines describing corrections and source selection."""
        return []

    def __repr__(self) -> str:
        w = 12
        lines = [
            f"{type(self).__name__} — {self.n_sweeps} "
            f"{'sweep' if self.n_sweeps == 1 else 'sweeps'} × "
            f"{self.n_points} {self._POINT_NOUN}  [{self.spectroscopy}]",
            f"  {'File':<{w}}: {self.path}",
        ]
        lines += self._repr_axis_lines(w)

        if self.sweep_type == "index":
            lines.append(
                f"  {'Sweep':<{w}}: not declared — sweep index used as the axis"
            )
        else:
            axis = self.sweep_axis
            unit = f" {self._sweep_unit}" if self._sweep_unit else ""
            lines.append(
                f"  {'Sweep':<{w}}: {self.sweep_type} — "
                f"{axis.min():.4g} → {axis.max():.4g}{unit}"
            )

        # Gate wiring is per-session and unrecorded, so state which row was read
        # as which gate: transposing them flips the sign of any extracted dipole.
        # Undeclared, say so here rather than anywhere else — this is where someone
        # looks after loading, and the failure being guarded against is a plot that
        # looks entirely normal.
        if self.gate_mode is not None:
            if self._gates is not None:
                # Every declared role, so the device topology is visible too: a
                # missing 'top' is what makes this a single-gated device.
                wiring = "(" + ", ".join(
                    f"{role} ← " + ("grounded" if lbl is None else f"'{lbl}'")
                    for role, lbl in self._gates.items()
                ) + ")"
            else:
                wiring = ("— wiring not declared; pass "
                          "gates={'top': ..., 'bottom': ...} for v_top/v_bot/E_F")
            lines.append(f"  {'Gates':<{w}}: {self.gate_mode}  {wiring}")

        # Context the sweep axis does not already show; skip E_F when the field
        # *is* the swept axis, which the Sweep line has just reported, and when the
        # device cannot define one — an undeclared wiring leaves its sign undefined,
        # a single gate leaves it undefined outright.  A repr must render whatever
        # state the object is in.
        # The power unit comes from the registry, not a literal: a curated_scales
        # override with its curated_units changes what these numbers are.
        extra = [("Power", self._curated_or_none("power"), self._curated["power"][2])]
        if self.sweep_type != "electric_field" and self.is_dual_gated:
            extra.append(("E_F", self.ef, "mV/nm"))
        for label, arr, unit in extra:
            if arr is not None:
                lines.append(
                    f"  {label:<{w}}: {arr.min():.1f} → {arr.max():.1f} {unit}"
                )

        varying = [lbl for lbl in self.varying_parameters()][:4]
        if varying:
            lines.append(f"  {'Varying':<{w}}: {', '.join(varying)}")

        # A declared nest is a fact about the scan; a detected one is only a
        # reading of the rows, so say which of the two this is.  Same split as
        # the gate wiring above.
        if self._nesting is not None:
            lines.append(f"  {'Nesting':<{w}}: {self._nesting}")
        else:
            grid = self.sweep_grid()
            if grid is not None:
                lines.append(
                    f"  {'Grid':<{w}}: {grid}  — detected, not declared; pass "
                    f"fast_sweep='{grid.fast_label}', "
                    f"slow_sweep='{grid.slow_label}'"
                )

        if self.n_declared_sweeps != self.n_sweeps:
            lines.append(
                f"  {'Blocks':<{w}}: {self.n_declared_sweeps} declared, "
                f"{self.n_declared_sweeps - self.n_sweeps} zero-filled and dropped"
            )
        lines += self._repr_extra_lines(w)
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# AttoCubeSpectralSweep
# ---------------------------------------------------------------------------

# Accepted keys of the `cosmic_rays=` declaration, read off the function they are
# forwarded to so the two cannot drift apart as that signature grows.  `spectra`
# is the array being loaded and `axis` is fixed by this class's
# (n_pixels, n_sweeps) convention, so neither is the caller's to set — passing an
# axis would transpose the detection without changing the stored shape.
_COSMIC_RAY_KEYS = frozenset(
    inspect.signature(processing.remove_cosmic_rays).parameters
) - {"spectra", "axis"}


class AttoCubeSpectralSweep(_Sweep):
    """
    A sweep of spectra from the AttoCube cryogenic confocal.

    One spectrum per sweep point, over any scanned parameter — displacement
    field, a single gate, excitation power, sample position, or a raw
    instrument channel.  What was measured (:attr:`spectra_type`) and what was
    swept (:attr:`sweep_type`) are recorded metadata rather than assumptions
    baked into the parser, so the same class serves a Stark-shift map and a
    power series.

    Two input formats, dispatched on the file suffix:

    * ``.csv`` — the raw AttoCube export.  The **first column** is a row label
      (e.g. ``"V_A"``, ``V_B``), every **sweep point** occupies four consecutive columns
      ``[Par, Wavelength, ExpROI1, ExpROI2]``, and the file is padded with
      empty columns beyond the last sweep point.
    * ``.h5`` / ``.hdf5`` — written by :meth:`to_hdf5`.  Carries the parameter
      rows verbatim plus the measurement metadata, so a re-read reproduces the
      object without the caller re-supplying ``spectra_type``, ``sweep``, or a
      :class:`DeviceGeometry`.

    .. note::
       The CSV parser is still the PL-shaped 4-column layout.  ``spectra_type``
       currently drives **labelling and metadata only** — it does not change how
       the file is decoded, because no reflectance or reflectance-contrast
       export has been characterised yet (see E9 in ``dev/defects.md``).
       Loading an ``"R"`` file works only if it happens to share the PL layout.

    Parameters
    ----------
    path : str or Path
        Path to the ``.csv`` or ``.h5`` file.
    spectra_type : str
        What the spectra *are*, one of
        :data:`~tmdc_optics_tools.constants.SPECTROSCOPY_TYPES`
        (``"PL"``, ``"R"``, ``"RC"``, ``"T"``, ``"A"``, ``"TRPL"``).
        Required for CSV input; optional when reading HDF5, where it is taken
        from the file unless given (a mismatch warns and the argument wins).
        Keyword-only and deliberately without a default: the value is written
        into exported metadata and trusted thereafter, so a guess would outlive
        the session that made it.
    sweep : str, optional
        What was scanned.  Either a key of :data:`_SWEEP_TYPES`
        (``"electric_field"``, ``"carrier_density"``, ``"top_voltage"``,
        ``"bottom_voltage"``, ``"power"``, ``"piezo_x"``, ``"piezo_y"``) or any raw CSV row
        label (``"Galvo_Y"``, ``"T"``, …), which is then used in its file
        units.  ``None`` (default) means **no axis is assumed**: the sweep axis
        becomes the sweep index (for sweep of length N, [1,2,3,…,N]).  Use :meth:`varying_parameters` to see which
        rows actually changed before committing to one.  The three gate sweeps
        name a physical electrode, so they also need *gates*.
    sweep_label, sweep_unit : str, optional
        Override the axis label / unit for the resolved sweep.  Needed mainly
        for raw-row sweeps, whose units the file does not state.
    fast_sweep, slow_sweep : str, optional
        Declare that the sweep points are a 2-D nest: *fast_sweep* is the inner
        axis, which runs to completion at every point of *slow_sweep*.  Both take
        the same vocabulary as *sweep*, a derived quantity included, and both are
        needed — one alone raises, since which axis is inner is the fact the
        declaration exists to carry.  Omit both to leave the sweep flat, which is
        the default: :attr:`spectra` keeps shape ``(n_points, n_sweeps)`` either
        way, and :meth:`as_grid` is how the grid is reached.

        The shape is worked out from the readings.  Only the divisors of
        :attr:`n_sweeps` can be shapes, since every point of the slow axis carries
        one complete run of the fast axis, and the shape that fits is the one under
        which neither axis's settings overlap each other.  A file whose readings
        fit no shape, or fit more than one, is refused rather than guessed at.
    n_fast, n_slow : int, optional
        Assert the nest's shape instead of resolving it from the readings, for a
        file that cannot establish one — a laser that plateaued, so two commanded
        settings read back the same power.  Either count alone is enough, since
        :attr:`n_sweeps` is known exactly; giving both cross-checks them.  A count
        that does not divide :attr:`n_sweeps` raises, so a scan aborted part-way
        through a row is still refused rather than truncated.

        Requires *fast_sweep* and *slow_sweep*: a shape says how the points divide
        up, not what was scanned along each axis.  The overlap checks still run and
        **warn** for every axis that cannot tell its own settings apart, which is
        also what surfaces the two counts being given the wrong way round — that
        shows up as a warning about the axis you expect to be clean.

        Asserting the shape settles the reshape only.  An overlapping axis's
        coordinates are still whatever was read at each level, so plotting against
        them misleads and :meth:`get_spectrum_at` on them stays ambiguous; prefer
        declaring a row that steps exactly where the file has one.
    fast_group_by, slow_group_by : str, optional
        Establish that axis's levels from a **different row** than the one being
        plotted.  Takes the same vocabulary as *fast_sweep*.

        The case this exists for is a power series: the commanded setpoint steps
        exactly and so says which spectra share a power, while the meter reading
        carries the physical µW the axis should be labelled in.  One row cannot do
        both, so::

            slow_sweep    = "power",                 # coordinate, in µW
            slow_group_by = "Fianium_Select_A4",     # the exact setpoint

        The grouping row is what the shape is resolved from and what must hold
        apart.  The labelled row supplies the coordinates and is **not** required to:
        each level's coordinate is the median of the readings taken at it, with the
        peak-to-peak range in :attr:`SweepNesting.slow_spread`.  If the labelled
        row's own levels overlap the load warns, because the grid is then right while
        the axis is not usable for plotting or for :meth:`get_spectrum_at`.

        Which row drives an instrument is per-session configuration, so it is named
        here and nowhere else — there is no default and no per-instrument shortcut.
    geometry : DeviceGeometry, optional
        Device geometry used to convert gate voltages to a displacement field.
        Without it :attr:`ef` is ``None``, and ``sweep="electric_field"``
        raises.
    cosmic_rays : dict, optional
        Opts into cosmic-ray repair, and carries the keyword arguments forwarded
        to :func:`~tmdc_optics_tools.processing.remove_cosmic_rays` — e.g.
        ``{"sigma_threshold": 4.0}``, or ``{}`` to accept that function's
        defaults.  ``None`` (default) leaves the counts alone.  An unknown key
        raises; ``spectra`` and ``axis`` are not accepted, being the array this
        loader read and the axis convention it stores it in.

        Runs in wavelength space and **before every other correction**, so it
        feeds the background estimate and the contrast: a spike inside the
        *bg_region* window would otherwise bias the pedestal estimate, and a spike
        in either array of a contrast biases the ratio non-linearly.
        :attr:`spectra` is left as the file wrote it — the repaired counts are
        :attr:`spectra_cr` and :attr:`energy_spectra_cr`, and the pixels replaced
        are :attr:`cosmic_ray_mask`.
    bg_region_nm : tuple of (wl_min, wl_max), optional
        Wavelength range in **nm** used to estimate the background level.
        The mean counts in this window are subtracted from every sweep
        point *before* any Jacobian correction is applied, which is the
        correct order of operations.  Mutually exclusive with
        *bg_region_eV*; passing both raises ``ValueError``.
    bg_region_eV : tuple of (E_min, E_max), optional
        Same as *bg_region_nm* but specified in **eV**.  Internally
        converted to a wavelength range (with the order flipped, since
        energy and wavelength are inversely related) before subtraction.
        Mutually exclusive with *bg_region_nm*.
    bg_spectrum : path, SingleSpectrum or array, optional
        A separately **measured** background spectrum, subtracted from every
        sweep point.  Unlike *bg_region_nm*, which removes a scalar estimated
        from a window of the spectrum itself, this removes wavelength-dependent
        structure — a dark frame or stray-light reference.  Applies to PL and
        reflectance alike.  Must be on this scan's wavelength axis; a mismatch
        raises rather than resampling.
    reference : path, SingleSpectrum or array, optional
        Reference spectrum against which to form a **contrast** — for
        reflectance, the bare-substrate measurement.  Supplying it is what opts
        into :attr:`contrast` / :attr:`energy_contrast`; nothing is computed
        otherwise.  Background-corrected first if *bg_spectrum* is also given.
    reference_scale : float, optional
        Multiplies *reference* before the contrast is formed.  Needed when the
        reference was **not** acquired at the same integration time and
        excitation power as the sample, because for a reference scaled by *k*,
        ``(S − kR)/(kR)`` is a *biased* contrast rather than a rescaled one.  A
        2-row reference CSV carries no parameter rows, so the package cannot
        check this and cannot correct for it: matching the acquisition, or
        supplying the ratio here, is the caller's responsibility.
    contrast : {"contrast", "ratio"}
        Which contrast to form when *reference* is given — ``(S − R)/R`` or
        ``S/R``.  See :func:`~tmdc_optics_tools.processing.spectral_contrast`.
    apply_jacobian : bool
        If ``True``, the Jacobian correction ``dλ/dE = λ²/(hc)`` is applied
        when building the energy-axis spectra, so integrated intensity is
        conserved under the wavelength → energy change of variables.  Default
        ``False``: it redistributes spectral weight, so it is opt-in, and peak
        *positions* do not need it.  It is **never** applied to
        :attr:`energy_contrast`: the factor cancels identically in a ratio.
        Warns when ``True`` and no background was supplied — neither
        *bg_region_nm* / *bg_region_eV* nor *bg_spectrum* — because the λ²
        factor scales a dark pedestal into a curved baseline instead of
        leaving it a flat offset.
    gates : dict, optional
        Which parameter row reached each electrode.  The roles *present* describe
        the device, so this declares its topology as well as its wiring:

        - ``{"top": "V_A", "bottom": "V_B"}`` — dual-gated. :attr:`ef` available.
        - ``{"bottom": "V_A", "channel": "V_B"}`` — bottom-gated with a contact to
          the TMDC. One gate is one degree of freedom, so there is no independent
          field: :attr:`v_top` and :attr:`ef` raise.

        ``"top"`` and ``"bottom"`` are gate electrodes across a dielectric;
        ``"channel"`` is a contact to the TMDC itself, the ground reference of a
        doping measurement.  A value of ``None`` means that electrode is tied to
        ground and no row records it, giving zero at every sweep point.  At least
        one gate is required, and a lone gate must be accompanied by its
        ``"channel"`` — one gate with a floating TMDC defines neither a field nor a
        density, so the declaration would be ambiguous rather than informative.

        Naming a source-meter row declares that electrode's **current** as well as
        its voltage: the instrument applies a bias and reports the current it
        sources at the same terminal, so ``"bottom": "V_A"`` also makes ``I_A``
        available as :attr:`i_bot`.  An electrode declared on a row that is not
        such a channel keeps its voltage and has no current.

        The electrodes can be wired either way round and no export records which
        way, so this is per-session information the file cannot supply.  Without
        it, :attr:`v_top`, :attr:`v_bot`, :attr:`v_channel`, :attr:`i_top`,
        :attr:`i_bot`, :attr:`i_channel`, :attr:`ef` and
        ``sweep="electric_field"`` / ``"top_voltage"`` / ``"bottom_voltage"`` all
        raise rather than assume one: transposing two gates mirrors the field axis
        and flips the sign of any extracted dipole.  Work in channel terms with
        ``scan["V_A"]`` or ``sweep="V_A"`` if the roles are not needed.

        Recorded on the scan (:attr:`gates`), shown in :func:`repr`, and written
        into exported HDF5.
    curated_labels : dict, optional
        Override which CSV row backs a curated attribute, e.g.
        ``{"power": "Laser Power"}``.  Keys are curated attribute names — the names
        listed under *Attributes*, not CSV row labels.  Unknown keys raise, as do
        the role-backed ones (``"v_top"``, ``"v_bot"``, ``"i_top"``, ``"i_bot"``,
        ``"i_channel"``): those are declared through *gates*, so that one fact has
        one spelling.
    curated_scales : dict, optional
        Override the scale factor of a curated attribute, e.g.
        ``{"power": 1.0}`` to keep raw power.  The scale *replaces* the registry's
        rather than multiplying it.  Keys are the same curated attribute names as
        *curated_labels*, except that the role-backed ones are accepted here: a
        unit conversion claims nothing about which channel reached which
        electrode.  Changing a scale without also passing *curated_units* warns.
    curated_units : dict, optional
        Override the unit of a curated attribute, e.g.
        ``{"scanner_x": "µm"}`` beside a µm/V *curated_scales* entry.  Same keys as
        *curated_scales*.  Pass ``""`` for a dimensionless quantity.

        This is not cosmetic.  The unit reaches :attr:`sweep_axis_label`, the
        :func:`repr` and every plot legend that names the swept coordinate, so a
        rescaled row left with its old unit reads as a measurement in the wrong
        units rather than as an untidy figure.
    roi : {1, 2}
        Which spectrometer ROI :attr:`spectra` points at.  Default ``1``.
        Both are always loaded — see :attr:`spectra_roi1` / :attr:`spectra_roi2`.

    Attributes
    ----------
    wavelength : np.ndarray, shape (n_pixels,)
        Spectrometer wavelength axis in nm (original, ascending order).
    energy : np.ndarray, shape (n_pixels,)
        Photon energy axis in eV (ascending order).
    spectra : np.ndarray, shape (n_pixels, n_sweeps)
        The file's own counts in wavelength space. Never modified after loading.
    spectra_cr : np.ndarray or None, shape (n_pixels, n_sweeps)
        :attr:`spectra` with cosmic rays replaced by local medians, or ``None``
        when *cosmic_rays* was not given.  Where it exists it is what the
        corrections below are computed from, :attr:`contrast` included.
    spectra_bg : np.ndarray or None, shape (n_pixels, n_sweeps)
        Background-subtracted counts in wavelength space, or ``None`` when neither
        *bg_region_nm* / *bg_region_eV* nor *bg_spectrum* was given.  Carries the
        cosmic-ray repair as well where one was declared, so
        ``spectra - spectra_bg`` is the pedestal alone only when no repair ran.
    cosmic_ray_mask : np.ndarray[bool] or None, shape (n_pixels, n_sweeps)
        Which pixels :attr:`spectra_cr` replaced, ``None`` when no repair was
        asked for.  ``cosmic_ray_mask.mean(axis=1)`` localises a detector defect
        or a narrow spectral feature flagged in most sweeps, which a cosmic ray
        cannot be.
    cosmic_rays : dict or None
        The repair arguments as declared, or ``None``.
    energy_spectra : np.ndarray, shape (n_pixels, n_sweeps)
        :attr:`spectra` on the ascending energy axis, Jacobian-corrected when
        *apply_jacobian* is ``True``.  No repair and no background subtraction.
    energy_spectra_cr : np.ndarray or None, shape (n_pixels, n_sweeps)
        :attr:`spectra_cr` on the energy axis, or ``None`` when no repair was
        declared.
    energy_spectra_pre_jacobian : np.ndarray, shape (n_pixels, n_sweeps)
        :attr:`energy_spectra` without the Jacobian, regardless of
        *apply_jacobian* — the same object when it is off.  Useful for
        peak-position fitting, where the density correction moves a centre.  The
        repaired and background-subtracted rungs have no pre-Jacobian variant of
        their own: their wavelength-space arrays hold exactly those values.
    energy_spectra_bg : np.ndarray or None, shape (n_pixels, n_sweeps)
        :attr:`spectra_bg` on the energy axis, or ``None`` when neither
        *bg_region_nm* / *bg_region_eV* nor *bg_spectrum* was given.  The
        background comes off in wavelength space *before* the Jacobian, so the
        correction does not curve the residual baseline.
    contrast : np.ndarray or None, shape (n_pixels, n_sweeps)
        Contrast against *reference* in wavelength space, or ``None`` when no
        reference was supplied.  See :attr:`contrast_label`.
    energy_contrast : np.ndarray or None, shape (n_pixels, n_sweeps)
        :attr:`contrast` on the ascending energy axis, built with the Jacobian
        **off** regardless of *apply_jacobian* — ``(S·λ²/hc)/(R·λ²/hc) = S/R``,
        so it cancels exactly in a ratio.
    reference_guarded : np.ndarray or None, shape (n_pixels,)
        Which reference pixels were non-positive and so excluded from the
        contrast (``NaN`` there).  Returned so the gaps can be handled knowingly.
    bg_region_nm : tuple or None
        The background window actually used, always in nm (even if the
        caller supplied *bg_region_eV*).
    apply_jacobian : bool
        Whether the Jacobian correction was applied.
    spectra_roi1, spectra_roi2 : np.ndarray, shape (n_pixels, n_sweeps)
        Both spectrometer ROIs in wavelength space, always loaded.  ``ExpROI2``
        is frequently all zeros in PL exports; for a measurement that carries a
        reference channel it is where that channel would sit.  :attr:`spectra`
        is whichever of the two *roi* selected.
    sweep_axis : np.ndarray, shape (n_sweeps,)
        The declared sweep axis in its own units, or ``arange(n_sweeps)`` when
        no *sweep* was declared.  Read-only property.
    sweep_axis_label : str
        Matching axis label, e.g. ``"$E_F$ (mV/nm)"``.
    sweep_type : str
        The resolved sweep key — a :data:`_SWEEP_TYPES` name, a raw CSV row
        label, or ``"index"``.
    spectra_type : str
        The declared measurement type (see *spectra_type* above).
    signal_label : str
        Y-axis label for the measured signal, from :attr:`spectra_type` — e.g.
        ``"PL intensity (counts)"``, ``"$\\Delta R/R_0$"``.  Use this instead of
        hardcoding "PL" in a plot that may be handed reflectance.
    signal_name : str
        The quantity alone, without the unit — e.g. ``"PL intensity"``.
    signal_unit : str
        The unit alone, empty for a dimensionless ratio — e.g. ``"counts"``.
    gates : dict or None
        The declared electrode mapping, or ``None`` if the wiring was never
        stated.  Read-only property.
    is_dual_gated : bool
        Whether *gates* declared both gate electrodes, so a field is defined.
        ``False`` for a single-gated device and for an undeclared wiring alike.
        Read-only property.
    v_top, v_bot : np.ndarray, shape (n_sweeps,)
        Top / bottom gate voltages in V.  Read-only properties (scaled views
        into :attr:`parameters`).  Raise unless *gates* declared that electrode.
    v_channel : np.ndarray, shape (n_sweeps,)
        Voltage on the contact to the TMDC itself in V, zero throughout for the
        usual grounded channel.  Raises unless *gates* declared a ``"channel"``.
        Read-only property.
    power : np.ndarray, shape (n_sweeps,)
        Excitation power in µW.  Read-only property (scaled view).
    i_top, i_bot : np.ndarray, shape (n_sweeps,)
        Leakage current at each gate in nA.  Raise unless *gates* declared that
        electrode on a source-meter channel.  Read-only properties (scaled views).
    i_channel : np.ndarray, shape (n_sweeps,)
        Current sourced into the contact on the TMDC in nA — transport, not
        leakage.  Raises unless *gates* declared a ``"channel"`` on a source-meter
        channel; a contact declared grounded with no row has no current to report.
        Read-only property (scaled view).
    scanner_x, scanner_y : np.ndarray, shape (n_sweeps,)
        Piezo scanner X / Y drive voltage in V — a drive level, not a distance.
        Converting to µm needs a per-stage µm/V calibration that is not in the
        file; supply one through *curated_scales*, and say what the numbers then
        are through *curated_units*, or the axis will still be labelled in volts.
        Read-only properties (views).
    ef : np.ndarray or None, shape (n_sweeps,)
        Displacement field in mV/nm, or ``None`` if no geometry supplied.
        Read-only property computed from :attr:`v_top` / :attr:`v_bot`, so it
        raises when a geometry was supplied and the device is not
        :attr:`is_dual_gated` — either because *gates* was not given, or because
        the device has one gate and therefore no field to report.
    carrier_density : np.ndarray or None, shape (n_sweeps,)
        Gate-induced sheet density in cm⁻² relative to zero gate voltage, summed
        over the declared gates, or ``None`` if no geometry supplied.  Raises
        unless *gates* declared a ``"channel"``.  Read-only property; see
        :meth:`DeviceGeometry.carrier_density` for a different reference.
    gate_mode : str or None
        Description of how the gates were driven — ``"dual-gate,
        anti-correlated (field-like)"``, ``"bottom-gate only"``, … — or ``None``
        when the gate rows are absent.  Descriptive, from the data; see
        :meth:`varying_parameters`.  Needs no *gates*: without one, a single
        driven gate is named by its channel rather than by a role.
    source_metadata : dict
        Metadata recorded in the source file, empty for CSV input.  Includes the
        ``apply_jacobian`` / ``bg_region_nm`` / ``cosmic_rays`` of the session
        that wrote an HDF5 file — provenance only.  Those corrections are **not**
        replayed on read; the stored spectra are raw, and a correction stays the
        caller's decision.  Pass ``cosmic_rays=`` again to repeat a repair.
    curated_parameters : dict[str, tuple]
        Mapping ``attr -> (csv_label, scale, unit)`` documenting which rows are
        promoted to the curated properties above, the scale applied, and the
        resulting unit.  Configurable via the class-level :attr:`_CURATED`
        registry and the constructor *curated_labels* / *curated_scales* /
        *curated_units* overrides.  A curated row that this file does not contain
        is simply absent here, and its property raises only if accessed.
    parameters : dict[str, np.ndarray]
        Every labeled parameter row from the CSV, mapped to its per-sweep
        array in the file's **raw** units (one entry per row, shape
        ``(n_sweeps,)``).  Exposes the full instrument state — e.g.
        ``"Scanner X"``, ``"Scanner Y"``, ``"Galvo_X"``, ``"Galvo_Y"``,
        ``"Excitation Power"``, ``"T"`` — beyond the curated attributes above.
        The curated attributes (:attr:`v_top`, :attr:`power`, …) are scaled
        views into this store; the values here are unscaled.
    geometry : DeviceGeometry or None
    path : str

    Notes
    -----
    The spectra come in one three-rung ladder per axis, and the two axes mirror
    each other: :attr:`spectra` / :attr:`spectra_cr` / :attr:`spectra_bg` in
    wavelength space, and :attr:`energy_spectra` / :attr:`energy_spectra_cr` /
    :attr:`energy_spectra_bg` on the ascending energy axis.  Each rung is the one
    above it plus one further correction, so a suffix names the **last** correction
    applied rather than the only one — a background-subtracted array also carries
    the cosmic-ray repair where one was declared.  A rung is ``None`` when its
    correction was not asked for; the first rung always exists.
    :attr:`best_spectra` / :attr:`best_energy_spectra` return the most-corrected
    rung present, so downstream code need not know which corrections ran.
    :attr:`contrast` / :attr:`energy_contrast` sit outside the ladder: a different
    quantity, not a better-corrected one.

    Use :attr:`parameter_labels` to list every available row name and
    :meth:`get_parameter` (or ``scan["label"]``) to pull any one of them, with
    an optional ``scale`` factor for unit conversion.

    Beware the raw-vs-scaled distinction: :attr:`parameters` holds the file's
    **raw** values, whereas the curated properties apply a scale — e.g.
    ``scan.power`` (µW) differs from ``scan.parameters["Excitation Power"]``
    (raw).  See :attr:`curated_parameters` for the exact label/scale/unit map.

    Examples
    --------
    >>> geom = DeviceGeometry.from_single("WS2", d_hbn_top=53, d_hbn_bottom=46)

    **A displacement-field PL sweep.**  The field's sign depends on which channel
    reached which electrode, so that mapping is stated rather than assumed:

    >>> scan = AttoCubeSpectralSweep(
    ...     "myscan.csv", spectra_type="PL",
    ...     sweep="electric_field", geometry=geom,
    ...     gates={"top": "V_A", "bottom": "V_B"},
    ... )
    >>> scan.sweep_axis_label
    '$E_F$ (mV/nm)'

    **A power series and a piezo line-scan — same class, same file layout:**

    >>> pwr  = AttoCubeSpectralSweep("power.csv", spectra_type="PL", sweep="power")
    >>> line = AttoCubeSpectralSweep("line.csv",  spectra_type="PL", sweep="piezo_y")

    **Don't know what was swept?  Load it and ask:**

    >>> scan = AttoCubeSpectralSweep("unknown.csv", spectra_type="PL")
    >>> scan.varying_parameters()             # {'V_A': (0.0, 1.0, 1.0), ...}
    >>> scan.gate_mode
    'dual-gate, anti-correlated (field-like)'

    **The same device rewired the other way round**, which mirrors the field axis:

    >>> scan = AttoCubeSpectralSweep(
    ...     "myscan.csv", spectra_type="PL", sweep="electric_field",
    ...     geometry=geom, gates={"top": "V_B", "bottom": "V_A"},
    ... )

    **A bottom-gated device with the TMDC contacted** — a doping sweep.  One gate
    is one degree of freedom, so ``ef`` and ``v_top`` raise; sweep the gate itself:

    >>> scan = AttoCubeSpectralSweep(
    ...     "doping.csv", spectra_type="PL", sweep="bottom_voltage",
    ...     gates={"bottom": "V_A", "channel": "V_B"},
    ... )
    >>> scan.is_dual_gated
    False
    >>> scan.gate_mode
    'bottom-gate only'

    **The same sweep on a density axis**, which is what one gate controls:

    >>> bottom = DeviceGeometry.from_single("WSe2", d_hbn_bottom=46)
    >>> scan = AttoCubeSpectralSweep(
    ...     "doping.csv", spectra_type="PL", sweep="carrier_density",
    ...     geometry=bottom, gates={"bottom": "V_A", "channel": "V_B"},
    ... )
    >>> scan.sweep_axis_label
    '$\\Delta n$ (cm$^{-2}$)'

    **Any instrument parameter, and a raw row as the sweep axis:**

    >>> scan.parameter_labels                 # ['Excitation Power', 'Galvo_X', ...]
    >>> scan.get_parameter("Scanner X")       # (n_sweeps,) raw units
    >>> scan["Galvo_Y"]                       # sugar for get_parameter
    >>> AttoCubeSpectralSweep("s.csv", spectra_type="PL",
    ...                       sweep="Galvo_Y", sweep_unit="V")

    **Corrections, and a round trip through HDF5:**

    >>> scan = AttoCubeSpectralSweep(
    ...     "myscan.csv", spectra_type="PL", sweep="electric_field",
    ...     geometry=geom, gates={"top": "V_A", "bottom": "V_B"},
    ...     bg_region_eV=(1.28, 1.32), apply_jacobian=True,
    ... )
    >>> scan.to_hdf5("myscan.h5")
    >>> again = AttoCubeSpectralSweep("myscan.h5")   # type, sweep, geometry, gates restored

    See Also
    --------
    tmdc_optics_tools.hdf5 : the export/import format, and what it stores.
    """

    # One sweep point per 4-column block: [Par, Wavelength, ExpROI1, ExpROI2].
    # The block *shape* is read from the header (see _read_block_layout); this is
    # the layout this class is willing to accept.
    _LAYOUT_KIND = "spectral"
    _AXIS_ATTR   = "wavelength"
    _SIGNAL_ATTR = "spectra"
    _POINT_NOUN  = "pixels"

    def __init__(
        self,
        path           : str,
        *,
        spectra_type   : str   = None,
        sweep          : str   = None,
        sweep_label    : str   = None,
        sweep_unit     : str   = None,
        fast_sweep     : str   = None,
        slow_sweep     : str   = None,
        n_fast          : int   = None,
        n_slow          : int   = None,
        fast_group_by   : str   = None,
        slow_group_by   : str   = None,
        geometry        : DeviceGeometry = None,
        cosmic_rays     : dict  = None,
        bg_region_nm    : tuple = None,
        bg_region_eV    : tuple = None,
        bg_spectrum            = None,
        reference              = None,
        reference_scale : float = None,
        contrast        : str   = "contrast",
        apply_jacobian  : bool  = False,
        gates           : dict  = None,
        curated_labels  : dict  = None,
        curated_scales  : dict  = None,
        curated_units   : dict  = None,
        roi             : int   = None,
    ):
        # Both checks precede the read: an export is large enough that a mistyped
        # argument should not cost the decode before it is reported.
        if bg_region_nm is not None and bg_region_eV is not None:
            raise ValueError(
                "Provide at most one of bg_region_nm or bg_region_eV, not both."
            )
        if cosmic_rays is not None:
            unknown = set(cosmic_rays) - _COSMIC_RAY_KEYS
            if unknown:
                raise ValueError(
                    f"cosmic_rays received unknown key(s) {sorted(unknown)}. "
                    f"Accepted: {sorted(_COSMIC_RAY_KEYS)} — these are forwarded "
                    f"to processing.remove_cosmic_rays, whose 'spectra' and "
                    f"'axis' are set by the loader. Pass cosmic_rays={{}} to "
                    f"accept every default."
                )

        # --- Decode, and settle everything independent of the spectral axis ---
        payload = self._decode_and_describe(
            path, spectra_type=spectra_type, geometry=geometry, gates=gates,
            curated_labels=curated_labels, curated_scales=curated_scales,
            curated_units=curated_units,
        )
        meta = payload["metadata"]

        self.wavelength   = payload["wavelength"]      # nm, ascending
        self.spectra_roi1 = payload["roi1"]            # (n_pixels, n_sweeps)
        self.spectra_roi2 = payload["roi2"]
        self._validate_payload()

        roi = roi if roi is not None else meta.get("roi", 1)
        if roi not in (1, 2):
            raise ValueError(f"roi must be 1 or 2, got {roi!r}.")
        self._roi    = int(roi)
        self.spectra = self.spectra_roi1 if roi == 1 else self.spectra_roi2

        # ExpROI2 is only written for two-spot galvo scans (excitation spot plus
        # a remote, spatially-filtered spot).  A flat zero array is otherwise
        # indistinguishable from a valid dark measurement, so say so.
        if not self.spectra.any():
            hint = (
                "ExpROI2 is only populated for two-spot galvo scans (excitation "
                "spot plus a remote, spatially-filtered spot); every other "
                "measurement leaves it blank. Did you mean roi=1?"
                if self._roi == 2 else
                "ExpROI1 is the normal acquisition ROI, so an all-zero one "
                "suggests the spectrometer recorded nothing for this scan."
            )
            warnings.warn(
                f"roi={self._roi} (ExpROI{self._roi}) is identically zero in "
                f"'{self.path}'. {hint}",
                UserWarning, stacklevel=3,
            )

        self.apply_jacobian  = apply_jacobian
        self.contrast_mode   = contrast
        self.reference_scale = reference_scale

        # Auxiliary spectra, resolved onto this scan's own wavelength grid.
        self.bg_spectrum = self._resolve_aux_spectrum(bg_spectrum, "bg_spectrum")
        self.reference   = self._resolve_aux_spectrum(reference, "reference")
        if self.reference is not None and reference_scale is not None:
            self.reference = self.reference * float(reference_scale)

        # No curated row is mandatory: a file from a different instrument
        # configuration still loads, and each property raises only if accessed.
        # What *is* checked is the row the declared sweep needs — which is why
        # this comes after the signal array, since it validates against n_sweeps.
        self._bind_sweep_axis(sweep, sweep_label, sweep_unit)
        self._bind_nesting(fast_sweep, slow_sweep, n_fast=n_fast, n_slow=n_slow,
                           fast_group_by=fast_group_by,
                           slow_group_by=slow_group_by)

        # --- Resolve background window to nm (always work in wavelength space) ---
        if bg_region_eV is not None:
            # E and λ are inversely related: higher E → shorter λ, so the
            # nm interval is (λ(E_max), λ(E_min)) — order flips.
            wl_lo = HC_EV_NM / bg_region_eV[1]   # E_max → λ_min
            wl_hi = HC_EV_NM / bg_region_eV[0]   # E_min → λ_max
            self.bg_region_nm = (wl_lo, wl_hi)
        else:
            self.bg_region_nm = bg_region_nm      # may be None

        # The Jacobian multiplies by λ², so it turns a constant dark pedestal
        # into a curve rather than leaving it as an offset a fit can absorb.
        # Both background mechanisms run in wavelength space below, so either
        # one satisfies this; neither means the pedestal is already curved on
        # every energy-axis rung.
        if apply_jacobian and self.bg_region_nm is None and self.bg_spectrum is None:
            warnings.warn(
                "apply_jacobian=True with no background subtraction: pass "
                "bg_region_nm / bg_region_eV or bg_spectrum. The Jacobian "
                "multiplies by λ²/hc, so an un-subtracted dark pedestal B "
                "becomes B·λ²/hc — a baseline curving up towards the red "
                "rather than a flat offset, which inflates fitted amplitude "
                "and FWHM. energy_spectra_pre_jacobian holds the file's counts on "
                "the energy axis with the Jacobian left off.",
                UserWarning, stacklevel=3,
            )

        # --- Cosmic rays, ahead of every other correction ----------------------
        # First because both of the corrections below read the counts as if they
        # were signal: a spike inside the bg_region window pulls the pedestal
        # estimate up, and one in either array of a contrast biases the ratio
        # non-linearly.  In wavelength space because the 3-point Laplacian the
        # detection is built on assumes uniform sample spacing, which the detector
        # axis has and the energy axis does not.
        self.cosmic_rays = dict(cosmic_rays) if cosmic_rays is not None else None
        if cosmic_rays is None:
            self.spectra_cr      = None
            self.cosmic_ray_mask = None
        else:
            self.spectra_cr, self.cosmic_ray_mask = processing.remove_cosmic_rays(
                self.spectra, axis=0, **cosmic_rays
            )

        # What the corrections below read: the repaired counts where a repair was
        # asked for, the file's own otherwise.  `spectra` is never reassigned, so a
        # repair adds a rung to the ladder rather than replacing one.
        signal = self.spectra if self.spectra_cr is None else self.spectra_cr

        # --- Build energy axis and energy-space spectra ---
        self.energy       = HC_EV_NM / self.wavelength              # eV, descending at this point
        _sort_idx         = np.argsort(self.energy)                 # ascending energy sort index
        self.energy       = self.energy[_sort_idx]                  # eV, ascending

        # The first rung, on both axes: the file's own counts, uncorrected. Built
        # from `spectra` rather than `signal` so that each rung has one meaning —
        # a repair is `energy_spectra_cr` below, not a silent change of this one.
        self.energy_spectra = self._build_energy_spectra(
            self.spectra, self.wavelength, _sort_idx, apply_jacobian
        )

        # energy_spectra_pre_jacobian: the same rung with no Jacobian, whatever
        # apply_jacobian says.  Identical object when it is off; a separate array
        # when it is on so both representations are always available.  The other
        # rungs need no pre-Jacobian variant of their own: their wavelength-space
        # arrays hold exactly those values.
        if apply_jacobian:
            self.energy_spectra_pre_jacobian = self._build_energy_spectra(
                self.spectra, self.wavelength, _sort_idx, apply_jacobian=False
            )
        else:
            self.energy_spectra_pre_jacobian = self.energy_spectra

        # The repair rung on the energy axis, mirroring spectra_cr.
        if self.spectra_cr is None:
            self.energy_spectra_cr = None
        else:
            self.energy_spectra_cr = self._build_energy_spectra(
                self.spectra_cr, self.wavelength, _sort_idx, apply_jacobian
            )

        # --- Wavelength-space corrections, in the order the physics requires ---
        # 1. the bg_region window mean
        # 2. a measured background spectrum.
        # Both must precede any ratio: a pedestal in either array biases a contrast
        # non-linearly, and both must precede the Jacobian (a flat pedestal B
        # becomes B·λ²/hc — curved, not flat — in energy space).
        corrected = signal
        if self.bg_region_nm is not None:
            corrected = subtract_background(
                corrected,
                bg_region = self.bg_region_nm,
                x         = self.wavelength,
                axis      = 0,
            )
        if self.bg_spectrum is not None:
            corrected = processing.subtract_spectrum(
                corrected, self.bg_spectrum, axis=0)

        # The background rung, on both axes.  Compared against `signal`, not
        # `spectra`, so a cosmic-ray repair on its own does not masquerade as a
        # background subtraction; both stay None in that case and `best_*` falls
        # back to the repair rung.
        if corrected is not signal:
            self.spectra_bg        = corrected
            self.energy_spectra_bg = self._build_energy_spectra(
                corrected, self.wavelength, _sort_idx, apply_jacobian
            )
        else:
            self.spectra_bg        = None
            self.energy_spectra_bg = None

        # --- Contrast against a reference spectrum -------------------------
        if self.reference is None:
            self.contrast = None
            self.energy_contrast = None
            self.reference_guarded = None
        else:
            self.contrast, self.reference_guarded = processing.spectral_contrast(
                corrected, self.reference, mode=contrast, axis=0,
            )
            # The Jacobian is NOT applied here, whatever apply_jacobian says:
            # (S·λ²/hc)/(R·λ²/hc) = S/R, so it cancels identically in a ratio.
            # Applying it to the numerator alone would distort the contrast.
            self.energy_contrast = self._build_energy_spectra(
                self.contrast, self.wavelength, _sort_idx, apply_jacobian=False
            )

    # --- Private helpers ---------------------------------------------------

    @staticmethod
    def _build_energy_spectra(
        spectra        : np.ndarray,
        wavelength_nm  : np.ndarray,
        sort_idx       : np.ndarray,
        apply_jacobian : bool,
    ) -> np.ndarray:
        """
        Convert raw wavelength-space spectra to an energy-axis array.

        Parameters
        ----------
        spectra : np.ndarray, shape (n_pixels, n_sweeps)
            Spectra in wavelength space (may already have BG subtracted).
        wavelength_nm : np.ndarray, shape (n_pixels,)
            Wavelength axis in nm, matching ``spectra`` row order.
        sort_idx : np.ndarray
            Argsort indices that put the energy axis in ascending order.
        apply_jacobian : bool
            Whether to apply the ``λ²/hc`` density correction.

        Returns
        -------
        np.ndarray, shape (n_pixels, n_sweeps)
            Spectra on the ascending energy axis.
        """
        if apply_jacobian:
            out = jacobian_correction_wvl2E(spectra, wavelength_nm, axis=0)
        else:
            out = spectra.copy()
        return out[sort_idx, :]

    # --- Decoding: one payload contract, one builder, two formats ----------

    @classmethod
    def _decode_csv(cls, path) -> dict:
        """
        Decode a raw AttoCube spectral sweep export.

        Each sweep point occupies a ``[Par, Wavelength, ExpROI1, ExpROI2]``
        block, so every field is read as a stride-``block_width`` column slice.
        The layout comes from the header rather than from arithmetic on the
        column count — see :func:`_read_block_layout`.
        """
        blocks = _read_block_layout(path)
        if blocks["kind"] != cls._LAYOUT_KIND:
            raise ValueError(
                f"'{path}' is a {blocks['kind']} export "
                f"({', '.join(blocks['roles'])}), but "
                f"{cls.__name__} reads {cls._LAYOUT_KIND} exports. "
                f"Use {_CLASS_FOR_KIND[blocks['kind']]} instead."
            )

        width, n_declared = blocks["block_width"], blocks["n_blocks"]
        raw = pd.read_csv(path, header=0, index_col=0, low_memory=False)
        row_labels = list(raw.index)

        # The code reads the whole CSV first and then keeps only the columns that
        # belong to the declared sweep blocks. That choice is faster than asking 
        # pandas.read_csv(..., usecols=...) to pre-filter a very large number of columns, 
        # because dropping thousands of unused columns during parsing is more expensive 
        # than parsing them and discarding them afterward.

        # Remove trailing pad
        d = raw.to_numpy(dtype=float)[:, :n_declared * width]

        # One column index per sweep point, for each field of the block.
        cols = {role: np.arange(offset, d.shape[1], width)
                for offset, role in enumerate(blocks["roles"])}

        # Blocks the exporter declared but never wrote are zero-filled, not
        # empty, so they survive any NaN-based strip and must go before the
        # arrays are shaped.
        keep, n_declared, axis_block = _drop_unwritten_blocks(
            d[:, cols["Wavelength"]], path)

        # Labeled scalar rows are overlaid on the leading pixel rows: row i's
        # value for sweep j sits in that block's Par column.
        parameters = {
            str(label): d[i, cols["Par"]][keep]
            for i, label in enumerate(row_labels)
            if not pd.isna(label) and str(label).strip()
        }

        # The axis is repeated once per sweep point; take the first written one
        # and use its finite entries to select the real pixel rows.  With nothing
        # written at all the axis is zeros, and the caller gets the diagnostic
        # from _validate_payload rather than an IndexError.
        axis_raw = d[:, cols["Wavelength"][axis_block]]
        valid_px = np.isfinite(axis_raw)

        payload = {
            "wavelength" : axis_raw[valid_px],
            "parameters" : parameters,
            "metadata"   : {},          # a raw export records none
            "n_declared" : n_declared,
        }
        for role, out in (("ExpROI1", "roi1"), ("ExpROI2", "roi2")):
            payload[out] = d[valid_px][:, cols[role][keep]]
        return payload

    def _resolve_aux_spectrum(self, spec, name: str) -> np.ndarray:
        """
        Resolve an auxiliary spectrum onto this scan's wavelength grid.

        Accepts a path, anything exposing ``wavelength`` plus ``best_spectra``
        (a :class:`SingleSpectrum`, or another sweep's single column), or a bare
        ``(n_pixels,)`` array — the same duck-typing the image loaders use for
        ``laser_ref``.

        A grid **mismatch raises** rather than interpolating: resampling changes
        the numbers and smooths the data, so it is a correction and cannot be a
        default.  A bare array is accepted precisely so a caller who has aligned
        the two axes themselves has a route in, with no extra API.
        """
        if spec is None:
            return None

        if isinstance(spec, (str, Path)):
            spec = SingleSpectrum(str(spec))

        wavelength = getattr(spec, "wavelength", None)
        if wavelength is None:
            values = np.asarray(spec, dtype=float)
            if values.ndim != 1 or values.size != self.wavelength.size:
                raise ValueError(
                    f"{name} was given as a bare array of shape {values.shape}, "
                    f"which does not match this scan's {self.wavelength.size}-pixel "
                    f"wavelength axis. Pass a SingleSpectrum or a path to let the "
                    f"axes be checked."
                )
            return values

        values = np.asarray(getattr(spec, "best_spectra", spec.spectra), dtype=float)
        wavelength = np.asarray(wavelength, dtype=float)
        if wavelength.shape != self.wavelength.shape or not np.allclose(
                wavelength, self.wavelength, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"{name} is on a different wavelength axis from this scan "
                f"({wavelength.size} points spanning "
                f"{wavelength.min():.3f}–{wavelength.max():.3f} nm, against "
                f"{self.wavelength.size} points spanning "
                f"{self.wavelength.min():.3f}–{self.wavelength.max():.3f} nm). "
                f"They are not resampled automatically, because interpolating "
                f"changes the numbers and smooths the data. Resample it yourself "
                f"and pass the resulting array as {name}=."
            )
        return values

    def _validate_payload(self) -> None:
        """Check the decoded arrays are self-consistent before anything uses them."""
        # Check that the spectra is 2D and has at least one sweep point
        if self.spectra_roi1.ndim == 2 and self.spectra_roi1.shape[1] == 0:
            # Every block was zero-filled: the file carries a parameter table and
            # no measurement.  That is the metadata companion written alongside a
            # TRPL sweep, whose header is the spectral layout.
            raise ValueError(
                f"'{self.path}' declares {self.n_declared_sweeps} sweep point(s) "
                f"but contains no spectra — every block is zero-filled. This is "
                f"the metadata companion written alongside a TRPL sweep, not a "
                f"spectral export. Point AttoCubeTRPLSweep at the directory "
                f"instead; it reads this file for cross-checking."
            )
        self._validate_axis_and_signals(
            self.wavelength,
            {"ExpROI1": self.spectra_roi1, "ExpROI2": self.spectra_roi2},
        )

    @property
    def n_pixels(self) -> int:
        """Number of spectrometer pixels — :attr:`n_points` under its optical name."""
        return self.n_points

    @property
    def roi(self) -> int:
        """Which spectrometer ROI :attr:`spectra` points at (1 or 2)."""
        return self._roi

    # --- The sweep axis ----------------------------------------------------

    @property
    def gate_axis(self) -> np.ndarray:
        """Deprecated alias for :attr:`sweep_axis`."""
        return self.sweep_axis

    @property
    def gate_axis_label(self) -> str:
        """Deprecated alias for :attr:`sweep_axis_label`."""
        return self.sweep_axis_label

    # --- What the spectra are ----------------------------------------------

    @property
    def best_spectra(self) -> np.ndarray:
        """
        Most-corrected wavelength-axis spectra available — the counterpart of
        :attr:`best_energy_spectra`, returning the same rung of the ladder.

        Returns :attr:`spectra_bg` when a background was supplied at load time,
        :attr:`spectra_cr` when a cosmic-ray repair was declared without one, and
        :attr:`spectra` otherwise, so downstream code need not know which.

        Never returns the contrast, even when a *reference* was supplied: that is
        a different quantity rather than a better-corrected one, and it is
        negative-going, which peak fits and intensity colour bars both misread.
        Use :attr:`contrast`.
        """
        for rung in (self.spectra_bg, self.spectra_cr):
            if rung is not None:
                return rung
        return self.spectra

    @property
    def best_energy_spectra(self) -> np.ndarray:
        """
        Most-corrected energy-axis spectra available — the counterpart of
        :attr:`best_spectra`, returning the same rung of the ladder.

        Returns :attr:`energy_spectra_bg` when a background was supplied at load
        time, :attr:`energy_spectra_cr` when a cosmic-ray repair was declared
        without one, and :attr:`energy_spectra` otherwise, so downstream code need
        not know which.

        Never returns the contrast, even when a *reference* was supplied: that is
        a different quantity rather than a better-corrected one, and it is
        negative-going, which peak fits and intensity colour bars both misread.
        Use :attr:`energy_contrast`, or ``spectra_source="contrast"`` in
        :mod:`~tmdc_optics_tools.plotting`.
        """
        for rung in (self.energy_spectra_bg, self.energy_spectra_cr):
            if rung is not None:
                return rung
        return self.energy_spectra

    @property
    def contrast_label(self) -> str:
        """
        Y-axis label for :attr:`contrast` / :attr:`energy_contrast`.

        Independent of :attr:`signal_label`, because the contrast is a *derived*
        quantity: a scan of ``spectra_type="R"`` keeps reporting "Reflected
        intensity (counts)" for its raw spectra while its contrast is labelled
        ΔR/R₀.
        """
        if self.contrast_mode == "ratio":
            return r"$R/R_0$"
        name, unit = SIGNAL_LABELS["RC"]
        return f"{name} ({unit})" if unit else name

    # --- Picking a window out of the spectral axis ----------------------------

    def pixel_slice(self, x_range: tuple, *, x_axis: str = "energy") -> slice:
        """
        Positions of the spectrometer pixels lying inside a spectral window.

        Gives back *where* the window is rather than the data inside it, so that
        one slice cuts a spectrum and its axis together and the two cannot drift
        apart — which is what a fit over part of a spectrum needs:

        >>> px   = scan.pixel_slice((1.63, 1.72))            # doctest: +SKIP
        >>> x, y = scan.energy[px], scan.get_spectrum_at(value=50)[px]

        Parameters
        ----------
        x_range : tuple of (lo, hi)
            The window, in the units of *x_axis* — eV for ``"energy"``, nm for
            ``"wavelength"``.  Bounds are inclusive, and their order carries no
            information: ``(1.72, 1.63)`` is the same window as ``(1.63, 1.72)``.
        x_axis : {"energy", "wavelength"}
            Which spectral axis *x_range* is given on.  It also fixes what the
            result may index, since the two orderings are reversed with respect
            to each other: :attr:`energy` and every ``energy_*`` array take an
            ``"energy"`` slice, while :attr:`wavelength`, the ``spectra*`` rungs
            and :attr:`contrast` take a ``"wavelength"`` one.  A slice from the
            wrong axis returns a real but wrong window.

        Returns
        -------
        slice
            Indexes the pixel axis — axis 0 of the spectra arrays, and the whole
            of :attr:`energy` / :attr:`wavelength`.  A slice rather than a mask,
            so indexing with it gives a view rather than a copy.

        Raises
        ------
        ValueError
            If no pixel lies inside the window, the message giving the span of
            the axis.  An empty window is refused here rather than being left to
            surface as an empty spectrum inside a fit.
        ValueError
            If the axis is not monotonic, so that the window is not one
            consecutive run of pixels and cannot be expressed as a slice.
        TypeError
            If *x_range* is not a pair of numbers.

        Warns
        -----
        UserWarning
            When a bound lies beyond the end of the axis by more than half a
            pixel, so the window returned is narrower than the one asked for.

        See Also
        --------
        nearest_index : the same idea on the sweep axis, one point rather than a
            run of them.

        Examples
        --------
        >>> scan.pixel_slice((1.63, 1.72))                     # doctest: +SKIP
        slice(400, 730, None)
        >>> scan.pixel_slice((720, 760), x_axis="wavelength")  # doctest: +SKIP
        slice(210, 540, None)
        """
        _, unit = _x_axis_name_unit(x_axis, what="pixel_slice()")
        values  = self.energy if x_axis == "energy" else self.wavelength
        return processing._window_slice(values, x_range, axis=x_axis, unit=unit,
                                       what="pixel_slice()", stacklevel=3)

    # --- Picking spectra out of the sweep ------------------------------------

    def get_spectrum_at(self, value: float = None, *,
                        axis   : str   = "sweep",
                        fast   : float = None,
                        slow   : float = None,
                        source : str   = "best",
                        x_axis : str   = "energy") -> np.ndarray:
        """
        Spectra at the sweep coordinate closest to the value(s) given.

        **The rank of the result depends on how much you specify.**  Pinning
        every axis gives one spectrum, ``(n_pixels,)``; leaving one free gives
        the line of spectra along it, ``(n_pixels, n)`` with the swept dimension
        last, as everywhere else in the package.

        Parameters
        ----------
        value : float, optional
            Coordinate on the sweep axis.  For a flat sweep only; a nest is
            addressed with *fast* and *slow*.
        axis : str
            Which quantity *value* is read against, spelled as ``sweep=`` spells
            it: a registry key such as ``"top_voltage"`` or a raw row label such
            as ``"V_A"``.  The default searches the declared sweep axis.  Use it
            when a sweep is declared in one coordinate and you want a point in
            another — a field sweep driven by both gates at a fixed ratio can be
            addressed by ``axis="top_voltage"``.  Not combinable with *fast* or
            *slow*: to address a nest by another quantity, declare the nest in
            it.
        fast, slow : float, optional
            Coordinates on the nest axes.  Give both for a single spectrum, or
            one to hold that axis and take every point of the other.
        source : str
            Which correction state to read: ``"best"`` (the most-corrected state
            available), ``"raw"``, ``"cr"``, ``"bg"``, ``"contrast"``, or
            ``"pre_jacobian"`` (energy axis only).
        x_axis : {"energy", "wavelength"}
            Which spectral ordering *source* is served on.  The returned spectra
            run along :attr:`energy` or :attr:`wavelength` accordingly.

        Returns
        -------
        np.ndarray
            A **view** into the source array, never a copy, so the
            never-mutate rule reaches it.  Strided rather than contiguous, as
            any column selection out of a row-major array is; anything
            downstream that demands contiguity will copy it.

        Raises
        ------
        ValueError
            If *value* is given for a nested sweep, or *fast*/*slow* for a flat
            one, or if neither is given.  For the whole grid, use :meth:`as_grid`.
        ValueError
            If a coordinate names more than one sweep point, since returning one
            of them would drop the rest without saying so.  Every quantity of a
            nest repeats, so this is what an undeclared nest looks like from
            here; the message says what to declare.  :meth:`nearest_index` warns
            instead of raising, because a single index is all it can return.

        Warns
        -----
        UserWarning
            When a requested coordinate is further than half a step from any real
            point — see :meth:`nearest_index`.

        See Also
        --------
        get_spectrum_by_index : the same selection by integer position.
        nearest_index : the index alone, for composing against another array.
        as_grid : the whole sweep reshaped onto the nest.

        Examples
        --------
        >>> scan.get_spectrum_at(2.5).shape                  # doctest: +SKIP
        (1340,)
        >>> scan.get_spectrum_at(15.0, axis="top_voltage").shape  # doctest: +SKIP
        (1340,)
        >>> scan.get_spectrum_at(fast=3.0, slow=1.0).shape   # doctest: +SKIP
        (1340,)
        >>> scan.get_spectrum_at(fast=3.0).shape             # doctest: +SKIP
        (1340, 51)
        """
        selector = self._sweep_selector(
            value, axis=axis, fast=fast, slow=slow, by_value=True,
            what="get_spectrum_at()")
        return _resolve_spectra(self, source, x_axis)[:, selector]

    def get_spectrum_by_index(self, index: int = None, *,
                              fast   : int = None,
                              slow   : int = None,
                              source : str = "best",
                              x_axis : str = "energy") -> np.ndarray:
        """
        Spectra at integer sweep positions — :meth:`get_spectrum_at` by index.

        Same arguments, same rank rules and same return contract, except that
        the coordinates are positions rather than values, so nothing is searched
        for and nothing is warned about.  Negative positions count from the end,
        as elsewhere in Python.

        Parameters
        ----------
        index : int, optional
            Position on the sweep axis.  For a flat sweep only.
        fast, slow : int, optional
            Positions on the nest axes, ``0 <= i < n_fast`` / ``n_slow``.
        source, x_axis
            As :meth:`get_spectrum_at`.

        Returns
        -------
        np.ndarray
            ``(n_pixels,)`` with every axis pinned, else ``(n_pixels, n)``.
            A view, as :meth:`get_spectrum_at`.

        Raises
        ------
        IndexError
            If a position is out of range for its axis.
        """
        selector = self._sweep_selector(
            index, fast=fast, slow=slow, by_value=False,
            what="get_spectrum_by_index()")
        return _resolve_spectra(self, source, x_axis)[:, selector]

    # --- Dunder methods ----------------------------------------------------

    def _repr_axis_lines(self, w: int) -> list:
        return [
            f"  {'λ range':<{w}}: "
            f"{self.wavelength.min():.1f} – {self.wavelength.max():.1f} nm",
            f"  {'Energy range':<{w}}: "
            f"{self.energy.min():.3f} – {self.energy.max():.3f} eV",
        ]

    def _repr_extra_lines(self, w: int) -> list:
        lines = [f"  {'ROI':<{w}}: ExpROI{self._roi}"]
        if self.cosmic_ray_mask is not None:
            n_flagged = int(self.cosmic_ray_mask.sum())
            lines.append(
                f"  {'Cosmic rays':<{w}}: {n_flagged} pixel"
                f"{'' if n_flagged == 1 else 's'} replaced"
            )
        if self.bg_region_nm is not None:
            lines.append(
                f"  {'BG region':<{w}}: "
                f"{self.bg_region_nm[0]:.1f} – {self.bg_region_nm[1]:.1f} nm"
            )
        lines.append(
            f"  {'Jacobian':<{w}}: "
            f"{'applied' if self.apply_jacobian else 'not applied'}"
        )
        return lines


# ---------------------------------------------------------------------------
# AttoCubePLVabScan — deprecated pre-rename name
# ---------------------------------------------------------------------------

class AttoCubePLVabScan(AttoCubeSpectralSweep):
    """
    Deprecated alias for :class:`AttoCubeSpectralSweep`, PL gate sweeps only.

    Reproduces the pre-rename behaviour exactly: ``spectra_type="PL"``, and a
    sweep axis of displacement field when a :class:`DeviceGeometry` is supplied,
    top-gate voltage otherwise — which is what the old ``gate_axis`` did.  The
    old per-channel ``*_label`` / ``power_scale`` arguments are accepted and
    folded into the new *gates* / *curated_labels* / *curated_scales* arguments.

    Assumes ``V_A`` drove the top gate and ``V_B`` the bottom unless
    *top_gate_label* / *bot_gate_label* say otherwise. That assumption is what
    :class:`AttoCubeSpectralSweep` refuses to make; it survives here only so that
    existing scripts keep running unchanged.

    .. deprecated::
       Use :class:`AttoCubeSpectralSweep` with an explicit ``spectra_type=`` and
       ``sweep=``.  This subclass exists so that existing notebooks and scripts
       keep running; it adds no behaviour of its own.

    Notes
    -----
    The warning is a ``FutureWarning``, not a ``DeprecationWarning``: Python
    filters the latter out by default outside ``__main__``, so a library that
    raises one is warning nobody.
    """

    def __init__(
        self,
        path            : str,
        geometry        : DeviceGeometry = None,
        bg_region_nm    : tuple = None,
        bg_region_eV    : tuple = None,
        apply_jacobian  : bool  = False,
        top_gate_label  : str   = None,
        bot_gate_label  : str   = None,
        power_label     : str   = None,
        power_scale     : float = None,
        roi             : int   = 1,
    ):
        warnings.warn(
            "AttoCubePLVabScan is deprecated; use AttoCubeSpectralSweep with an "
            "explicit spectra_type= and sweep=, e.g. "
            "AttoCubeSpectralSweep(path, spectra_type='PL', "
            "sweep='electric_field', geometry=geom).",
            FutureWarning, stacklevel=2,
        )

        labels = {name: value for name, value in (
            ("power", power_label),
        ) if value is not None}

        # Both sweeps below resolve through a gate role, which the new API requires
        # be declared.  State the historical mapping here so scripts written against
        # this class keep producing the numbers they always did — its FutureWarning
        # is what asks callers to confirm the wiring rather than inherit it.
        gates = {
            "top":    top_gate_label if top_gate_label is not None else "V_A",
            "bottom": bot_gate_label if bot_gate_label is not None else "V_B",
        }

        super().__init__(
            path,
            spectra_type   = "PL",
            # The old gate_axis was ef when a geometry was given, else v_top.
            sweep          = "electric_field" if geometry is not None else "top_voltage",
            geometry       = geometry,
            gates          = gates,
            bg_region_nm   = bg_region_nm,
            bg_region_eV   = bg_region_eV,
            apply_jacobian = apply_jacobian,
            curated_labels = labels or None,
            # No curated_units to pass on: this class never had a unit argument,
            # and it is being retired rather than extended.  A power_scale= here
            # therefore raises the rescaled-without-a-unit warning through one
            # extra frame, so it points at this call rather than at the caller's
            # line.  The FutureWarning above is what asks them to move.
            curated_scales = {"power": power_scale} if power_scale is not None else None,
            roi            = roi,
        )


# ---------------------------------------------------------------------------
# AttoCubeTRPLSweep
# ---------------------------------------------------------------------------

# The TRPL export's time axis. 4 ps bins over ~12.8 ns, consistent with the
# Picoharp rows in the parameter list and a ~78 MHz repetition rate. Defined once
# so the unit is a single edit if it is ever found to be otherwise.
_TRPL_TIME_UNIT = "ns"


class AttoCubeTRPLSweep(_Sweep):
    """
    Time-resolved PL from the AttoCube cryogenic confocal.

    One TCSPC decay per sweep point.  A **single decay is one file**; a **sweep is
    a directory** of per-point files, one per iteration, written alongside a
    metadata companion — so this class accepts either.

    The export's block is ``[Par_i, Wavelength{i}, Exp_{i}]``, and the column
    *named* "Wavelength" actually holds **time**: an acquisition-software
    misnomer that this class reads as time and exposes as :attr:`time`.

    Why this is a separate class from :class:`AttoCubeSpectralSweep` rather than a
    mode of it: a single decay is simply ``n_sweeps == 1``, which the sweep
    machinery already handles, so that is not the dividing line.  The axis is.
    ``energy = hc/t`` is meaningless and divides by zero at ``t = 0``, so
    ``energy``, ``energy_spectra``, ``apply_jacobian`` and ``bg_region_eV`` do not
    exist here — and neither does ``spectra``, so handing a TRPL sweep to a
    spectral plot raises rather than drawing time as if it were wavelength.

    Parameters
    ----------
    path : str or Path
        A directory (a sweep) or a single ``.csv`` (one decay).  An ``.h5``
        written by :meth:`to_hdf5` also works.
    prefix : str, optional
        Filename filter applied within a directory, e.g. ``"TRPL_"``.  ``None``
        (default) considers every ``.csv`` and classifies by content.
    spectra_type : str
        Defaults to ``"TRPL"``.  Deliberately unlike
        :class:`AttoCubeSpectralSweep`, which requires an explicit type: the
        class name already declares the modality here, whereas a spectral sweep
        may be PL, R, RC, T or A and cannot know which.
    sweep, sweep_label, sweep_unit : str, optional
        As :class:`AttoCubeSpectralSweep`.  A gate sweep needs a *geometry* for
        ``"electric_field"``, and *gates* for any of the three.
    fast_sweep, slow_sweep, n_fast, n_slow, fast_group_by, slow_group_by : optional
        As :class:`AttoCubeSpectralSweep`: declare a 2-D nest, optionally assert its
        shape, and optionally group its levels from a different row than the one
        labelled.  :meth:`as_grid` reshapes :attr:`decays` onto it.  No nested TRPL
        sweep has been seen, so this is untested against a real file.
    geometry : DeviceGeometry, optional
    bg_region_ns : tuple of (t_min, t_max), optional
        **Pre-pulse** time window whose mean is subtracted from every decay —
        the TCSPC equivalent of a spectral background window, and the dark/
        afterpulse floor before the rise.  ``None`` (default) subtracts nothing.
    gates : dict, optional
        Which parameter row reached each electrode, e.g.
        ``{"top": "V_A", "bottom": "V_B"}``.  As
        :class:`AttoCubeSpectralSweep` — the roles present describe the device, and
        the declaration is required for :attr:`v_top`, :attr:`v_bot`,
        :attr:`v_channel`, :attr:`ef` and the gate sweep types.
    curated_labels, curated_scales, curated_units : dict, optional
        Override which row backs a curated attribute, its scale, and its unit.
        As :class:`AttoCubeSpectralSweep`, and keyed by the same curated attribute
        names — which here include the three Picoharp rows added by this class.
        Two of those carry a unit this package has not confirmed (``"Hz?"``,
        ``"s?"``); *curated_units* is where a confirmed one is stated, so the
        question mark does not reach a figure.
    time_rtol : float
        Tolerance for agreement between the per-file time axes.  They are *not*
        bit-identical across files — the example sweep's bin width varies in its
        seventh figure — so this is a closeness test, not an equality test.

    Attributes
    ----------
    time : np.ndarray, shape (n_bins,)
        Time axis in ns, ascending.
    decays : np.ndarray, shape (n_bins, n_sweeps)
        Raw photon counts.  Never modified after load.
    decays_bg : np.ndarray or None
        Pre-pulse-subtracted decays, or ``None`` when no *bg_region_ns* was given.
    best_decays : np.ndarray
        :attr:`decays_bg` when available, else :attr:`decays`.
    n_bins : int
        Number of time bins.
    files : list of Path
        The per-point data files, in sweep order.  A single-file load gives one.
    metadata_file : Path or None
        The companion parameter table, when the directory contained one.
    irf_files : list of str
        Names of candidate files excluded from the sweep because "irf"
        appears in their filename — never a data point or companion, load
        one on its own via a separate ``AttoCubeTRPLSweep(irf_path)``. Empty
        for a single-file load.
    declared_parameters : dict or None
        The companion's own parameter table, one value per declared sweep point.
        Provided for comparison against :attr:`parameters`; the two are
        independent read-backs and are not expected to agree on drifting
        channels — see :meth:`_cross_check_companion`.

    Notes
    -----
    Parameters come from **each data file's own snapshot**, not from the metadata
    companion.  Each snapshot is recorded at the moment its own decay was
    measured, and using them means a sweep still loads when the companion was
    never written or has been lost.  The companion is evidence: it supplies
    :attr:`n_declared_sweeps`, so a truncated sweep is visible, and its own table
    is exposed as :attr:`declared_parameters` for comparison.  Its values are not
    checked row by row — it is written after the last decay, so channels that
    drift genuinely disagree with the snapshots taken during acquisition.

    Examples
    --------
    >>> decay = AttoCubeTRPLSweep("TRPL_..._iter_0.csv")     # one decay
    >>> decay.n_sweeps
    1

    >>> geom  = DeviceGeometry.from_single("WSe2", d_hbn_top=53, d_hbn_bottom=46)
    >>> sweep = AttoCubeTRPLSweep(
    ...     "examples/data/TRPL/", sweep="electric_field", geometry=geom,
    ...     gates={"top": "V_A", "bottom": "V_B"}, bg_region_ns=(0.0, 1.0),
    ... )
    >>> sweep.time.max(), sweep.n_sweeps
    (12.817, 3)
    """

    _LAYOUT_KIND = "temporal"
    _AXIS_ATTR   = "time"
    _SIGNAL_ATTR = "decays"
    _POINT_NOUN  = "time bins"

    # Picoharp channels, relevant to normalising decay counts.  Units unconfirmed
    # — the same standing question as Scanner X/Y and power_scale.
    _CURATED = {
        **_Sweep._CURATED,
        "rep_rate":        ("Picoharp - RepRate",                 1.0, "Hz?"),
        "meas_time":       ("Picoharp - Actual Measurement Time", 1.0, "s?"),
        "picoharp_counts": ("Picoharp - Counts",                  1.0, "counts"),
    }

    def __init__(
        self,
        path,
        *,
        prefix         : str   = None,
        spectra_type   : str   = "TRPL",
        sweep          : str   = None,
        sweep_label    : str   = None,
        sweep_unit     : str   = None,
        fast_sweep     : str   = None,
        slow_sweep     : str   = None,
        n_fast         : int   = None,
        n_slow         : int   = None,
        fast_group_by  : str   = None,
        slow_group_by  : str   = None,
        geometry       : DeviceGeometry = None,
        bg_region_ns   : tuple = None,
        gates          : dict  = None,
        curated_labels : dict  = None,
        curated_scales : dict  = None,
        curated_units  : dict  = None,
        time_rtol      : float = 1e-4,
    ):
        self._prefix    = prefix
        self._time_rtol = time_rtol

        payload = self._decode_and_describe(
            path, spectra_type=spectra_type, geometry=geometry, gates=gates,
            curated_labels=curated_labels, curated_scales=curated_scales,
            curated_units=curated_units,
        )

        self.time   = payload["time"]                 # ns, ascending
        self.decays = payload["counts"]               # (n_bins, n_sweeps)
        self.files              = payload.get("files", [Path(self.path)])
        self.metadata_file      = payload.get("metadata_file")
        self.declared_parameters = payload.get("declared_parameters")
        self.irf_files          = payload.get("irf_files", [])
        self._validate_axis_and_signals(self.time, {"Exp": self.decays})

        self._bind_sweep_axis(sweep, sweep_label, sweep_unit)
        self._bind_nesting(fast_sweep, slow_sweep, n_fast=n_fast, n_slow=n_slow,
                           fast_group_by=fast_group_by,
                           slow_group_by=slow_group_by)

        # Pre-pulse baseline.  processing.subtract_background is generic in x, so
        # a time window needs no separate implementation from a spectral one.
        self.bg_region_ns = bg_region_ns
        if bg_region_ns is not None:
            self.decays_bg = subtract_background(
                self.decays, bg_region=bg_region_ns, x=self.time, axis=0,
            )
        else:
            self.decays_bg = None

    # --- Decoding ----------------------------------------------------------

    @classmethod
    def _decode_csv(cls, path) -> dict:
        """Decode one TRPL export: a single decay, so ``n_sweeps == 1``."""
        blocks = _read_block_layout(path)
        if blocks["kind"] != cls._LAYOUT_KIND:
            raise ValueError(
                f"'{path}' is a {blocks['kind']} export "
                f"({', '.join(blocks['roles'])}), but {cls.__name__} reads "
                f"{cls._LAYOUT_KIND} exports. Use "
                f"{_CLASS_FOR_KIND[blocks['kind']]} instead."
            )
        return cls._decode_temporal_file(path, blocks)

    @staticmethod
    def _decode_temporal_file(path, blocks: dict) -> dict:
        """Shared per-file decode, used for one decay and for each of a sweep."""
        width      = blocks["block_width"]
        n_declared = blocks["n_blocks"]
        raw = pd.read_csv(path, header=0, index_col=0, low_memory=False)
        row_labels = list(raw.index)
        d = raw.to_numpy(dtype=float)[:, :n_declared * width]

        cols = {role: np.arange(offset, d.shape[1], width)
                for offset, role in enumerate(blocks["roles"])}
        keep, n_declared, axis_block = _drop_unwritten_blocks(
            d[:, cols["Wavelength"]], path)

        parameters = {
            str(label): d[i, cols["Par"]][keep]
            for i, label in enumerate(row_labels)
            if not pd.isna(label) and str(label).strip()
        }

        # "Wavelength" holds time here — the exporter's misnomer, not ours.
        axis_raw  = d[:, cols["Wavelength"][axis_block]]
        valid_bin = np.isfinite(axis_raw)

        return {
            "time"       : axis_raw[valid_bin],
            "counts"     : d[valid_bin][:, cols["Exp"][keep]],
            "parameters" : parameters,
            "metadata"   : {},
            "n_declared" : n_declared,
        }

    def _decode_dir(self, path) -> dict:
        """
        Assemble a sweep from a directory of per-point files.

        Files are classified by **header alone**, so the 11 MB metadata companion
        is never parsed just to discover what it is. An instrument-response
        reference is the one exception content cannot settle: an IRF file has
        the identical ``[Par, Wavelength, Exp]`` shape as a real decay, so only
        its **name** says it is not a sweep point. Any candidate with "irf" in
        its filename (case-insensitive) is therefore excluded here regardless
        of *prefix* — it is never a data point or a companion, only ever an
        IRF, loaded on its own via a separate ``AttoCubeTRPLSweep(irf_path)``.
        """
        pattern    = f"{self._prefix or ''}*.csv"
        candidates = sorted(Path(path).glob(pattern))
        data_files, companions, skipped, irf_files = [], [], [], []
        for f in candidates:
            if "irf" in f.stem.lower():
                irf_files.append(f.name)
                continue
            try:
                layout = _read_block_layout(f)
            except ValueError:
                skipped.append(f.name)
                continue
            if layout["kind"] == self._LAYOUT_KIND:
                data_files.append((f, layout))
            elif layout["kind"] == "spectral":
                # A spectral header in a TRPL directory is the parameter-table
                # companion; a real spectral sweep would not live here.
                companions.append((f, layout))
            else:
                skipped.append(f.name)

        if not data_files:
            raise ValueError(
                f"No TRPL data files found in '{path}' matching '{pattern}'. "
                f"Looked at {len(candidates)} .csv file(s): "
                f"{len(companions)} metadata companion(s), "
                f"{len(irf_files)} IRF reference(s) (excluded by name), "
                f"{len(skipped)} unrecognised ({', '.join(skipped[:5])}). "
                f"A TRPL export has a [Par, Wavelength, Exp] block layout."
            )

        # _order_by_iter speaks paths; the shared image-sequence loader has no
        # layouts. dict is Path -> layout, so they re-attach by lookup after the
        # sort. stacklevel 4 matches this call chain's existing depth.
        layouts    = dict(data_files)
        data_files = [(f, layouts[f]) for f in _order_by_iter(
            [f for f, _ in data_files], path, stacklevel=4)]
        payload    = self._assemble(data_files, path)
        payload["irf_files"] = irf_files

        if companions:
            self._cross_check_companion(payload, companions, path)
        return payload

    def _assemble(self, data_files: list, path) -> dict:
        """
        Stack one decay per file into ``(n_bins, n_sweeps)``.

        The time axis is taken from the first file and the rest are checked
        against it; parameters come from each file's own ``Par_0`` snapshot,
        contemporaneous with the decay it belongs to.
        """
        per_file = [self._decode_temporal_file(f, layout)
                    for f, layout in data_files]

        time = per_file[0]["time"]
        rtol = self._time_rtol
        for (f, _), one in zip(data_files[1:], per_file[1:]):
            if one["time"].size != time.size:
                raise ValueError(
                    f"'{f.name}' has {one['time'].size} time bins but "
                    f"'{data_files[0][0].name}' has {time.size}. Every file in a "
                    f"sweep must share one time axis."
                )
            if not np.allclose(one["time"], time, rtol=rtol, atol=0.0):
                worst = int(np.argmax(np.abs(one["time"] - time)))
                raise ValueError(
                    f"'{f.name}' has a time axis that differs from "
                    f"'{data_files[0][0].name}' by more than rtol={rtol:g} "
                    f"(largest disagreement at bin {worst}: "
                    f"{one['time'][worst]:.9g} vs {time[worst]:.9g} "
                    f"{_TRPL_TIME_UNIT}). Raise time_rtol if this is only "
                    f"acquisition jitter."
                )

        # One column per file: each file holds a single decay, so its (n_bins, 1)
        # counts block becomes column i of the assembled sweep.
        counts = np.hstack([one["counts"] for one in per_file])

        # Each file carries its own full snapshot; keep only rows every file has,
        # so a parameter array can never be short of a sweep point.
        shared = set(per_file[0]["parameters"])
        for one in per_file[1:]:
            shared &= set(one["parameters"])
        parameters = {
            label: np.concatenate([one["parameters"][label] for one in per_file])
            for label in shared
        }

        return {
            "time"       : time,
            "counts"     : counts,
            "parameters" : parameters,
            "metadata"   : {},
            "files"      : [f for f, _ in data_files],
        }

    def _cross_check_companion(self, payload: dict, companions: list, path) -> None:
        """
        Use the metadata companion as evidence, never as the source.

        It supplies the declared sweep length, so a **truncated or aborted sweep
        is visible** — that is the check with real signal, and the only one made
        here.

        Its parameter *values* are exposed as :attr:`declared_parameters` but are
        deliberately **not** compared row by row.  The companion is written seconds
        after the last decay, so it and the per-file snapshots are independent
        read-backs of a moving instrument: on the example sweep the leakage
        currents (~1e-11 A) and ``Fianium_Select_A6`` (160/140/130 against
        190/190/140) differ genuinely, while the swept gates agree to seven
        figures.  Nothing in the file says which channels are stable, so a
        value check cannot separate "wrong companion" from "channel drifted", and
        would fire on every real sweep — which is how warnings get ignored.
        Both tables are returned; the comparison is the caller's to make.
        """
        if len(companions) > 1:
            warnings.warn(
                f"'{path}' holds {len(companions)} metadata companions "
                f"({', '.join(f.name for f, _ in companions)}); using "
                f"'{companions[0][0].name}'.",
                UserWarning, stacklevel=5,
            )
        companion, layout = companions[0]
        payload["metadata_file"]   = companion
        payload["n_declared"]      = layout["n_blocks"]

        raw = pd.read_csv(companion, header=0, index_col=0, low_memory=False)
        width = layout["block_width"]
        d = raw.to_numpy(dtype=float)[:, :layout["n_blocks"] * width]
        par_cols = np.arange(0, d.shape[1], width)
        declared = {
            str(label): d[i, par_cols]
            for i, label in enumerate(raw.index)
            if not pd.isna(label) and str(label).strip()
        }
        payload["declared_parameters"] = declared

        n_found = payload["counts"].shape[1]
        if layout["n_blocks"] != n_found:
            warnings.warn(
                f"'{companion.name}' declares {layout['n_blocks']} sweep point(s) "
                f"but {n_found} data file(s) were found in '{path}'. The sweep may "
                f"have been aborted, or files may be missing. The files present "
                f"are loaded; compare declared_parameters against parameters to "
                f"see which points they are.",
                UserWarning, stacklevel=5,
            )

    # --- Convenience -------------------------------------------------------

    @property
    def n_bins(self) -> int:
        """Number of time bins — :attr:`n_points` under its temporal name."""
        return self.n_points

    @property
    def axis_label(self) -> str:
        """Label for the measured axis, e.g. ``"Time (ns)"``."""
        return f"Time ({_TRPL_TIME_UNIT})"

    @property
    def best_decays(self) -> np.ndarray:
        """
        Pre-pulse-subtracted decays when available, else the raw counts.

        The temporal counterpart of
        :attr:`AttoCubeSpectralSweep.best_energy_spectra`.
        """
        return self.decays_bg if self.decays_bg is not None else self.decays

    def _repr_axis_lines(self, w: int) -> list:
        return [
            f"  {'Time range':<{w}}: {self.time.min():.4g} – "
            f"{self.time.max():.4g} {_TRPL_TIME_UNIT} "
            f"({(self.time[1] - self.time[0]) * 1e3:.3g} ps bins)"
            if self.n_bins > 1 else
            f"  {'Time':<{w}}: {self.time[0]:.4g} {_TRPL_TIME_UNIT}",
        ]

    def _repr_extra_lines(self, w: int) -> list:
        lines = []
        if len(self.files) > 1:
            lines.append(f"  {'Files':<{w}}: {len(self.files)} per-point files")
        if self.metadata_file is not None:
            lines.append(f"  {'Companion':<{w}}: {self.metadata_file.name}")
        lines.append(f"  {'Peak counts':<{w}}: {int(self.decays.max())}")
        if self.bg_region_ns is not None:
            lines.append(
                f"  {'Pre-pulse':<{w}}: {self.bg_region_ns[0]:.4g} – "
                f"{self.bg_region_ns[1]:.4g} {_TRPL_TIME_UNIT} subtracted"
            )
        return lines


# ---------------------------------------------------------------------------
# SingleSpectrum
# ---------------------------------------------------------------------------

class SingleSpectrum:
    """
    Single PL spectrum loaded from a 2-row CSV.

    The file must contain exactly two comma-separated rows:

    * **Row 0** : wavelength axis in nm (ascending).
    * **Row 1** : counts (PL intensity).

    Attribute names mirror :class:`AttoCubeSpectralSweep` so the same plotting
    helpers (e.g. :func:`~tmdc_optics_tools.plotting.plot_single_spectrum`,
    :func:`~tmdc_optics_tools.plotting._resolve_x_axis`) work unchanged.

    Parameters
    ----------
    path : str or Path
        Path to the ``.csv`` file.
    apply_jacobian : bool
        If ``True``, apply the ``dλ/dE = λ²/hc`` density correction when
        building :attr:`energy_spectra`, conserving integrated intensity
        under the wavelength → energy change of variables. Default ``False``.
        Warns when ``True`` and no background region was supplied, because the
        λ² factor scales a dark pedestal into a curved baseline instead of
        leaving it a flat offset.
    bg_region_nm : tuple of (wl_min, wl_max), optional
        Wavelength range in **nm** used to estimate the background level.
        The mean counts in this window are subtracted in wavelength space
        *before* any Jacobian correction is applied (the correct order of
        operations). Mutually exclusive with *bg_region_eV*.
    bg_region_eV : tuple of (E_min, E_max), optional
        Same as *bg_region_nm* but specified in **eV** (internally converted
        to a wavelength range, with the order flipped). Mutually exclusive
        with *bg_region_nm*.

    Attributes
    ----------
    wavelength : np.ndarray, shape (n_pixels,)
        Wavelength axis in nm (ascending).
    spectra : np.ndarray, shape (n_pixels,)
        Raw counts in wavelength space. Never modified after loading.
    spectra_bg : np.ndarray or None, shape (n_pixels,)
        Background-subtracted counts in wavelength space, or ``None`` when no
        background region was supplied.
    energy : np.ndarray, shape (n_pixels,)
        Photon energy axis in eV (ascending).
    energy_spectra : np.ndarray, shape (n_pixels,)
        Raw counts remapped to the energy axis, Jacobian-corrected if
        *apply_jacobian* is ``True``. No background subtraction.
    energy_spectra_bg : np.ndarray or None, shape (n_pixels,)
        Background-subtracted version of :attr:`energy_spectra` (background
        removed in wavelength space before the Jacobian). ``None`` when no
        background region was supplied.
    bg_region_nm : tuple or None
        The background window actually used, always in nm.
    apply_jacobian : bool
    path : str

    Notes
    -----
    Use :attr:`best_spectra` / :attr:`best_energy_spectra` in downstream code
    to automatically pick the background-corrected array when one is available
    and fall back to the raw array otherwise.
    """

    def __init__(
        self,
        path           : str,
        apply_jacobian : bool  = False,
        bg_region_nm   : tuple = None,
        bg_region_eV   : tuple = None,
    ):
        if bg_region_nm is not None and bg_region_eV is not None:
            raise ValueError(
                "Provide at most one of bg_region_nm or bg_region_eV, not both."
            )

        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError(
                f"Expected a 2-row CSV [wavelength; counts], got array of "
                f"shape {arr.shape} from '{path}'."
            )

        self.path           = str(path)
        self.apply_jacobian = apply_jacobian

        # Resolve background window to nm (always work in wavelength space).
        # E and λ are inversely related, so the nm interval order flips.
        if bg_region_eV is not None:
            self.bg_region_nm = (HC_EV_NM / bg_region_eV[1],   # E_max → λ_min
                                 HC_EV_NM / bg_region_eV[0])   # E_min → λ_max
        else:
            self.bg_region_nm = bg_region_nm                   # may be None

        # λ² scaling turns a constant dark pedestal into a curve, so the
        # subtraction below has to happen for energy_spectra to mean anything.
        if apply_jacobian and self.bg_region_nm is None:
            warnings.warn(
                "apply_jacobian=True with no background subtraction: pass "
                "bg_region_nm or bg_region_eV. The Jacobian multiplies by "
                "λ²/hc, so an un-subtracted dark pedestal B becomes B·λ²/hc — "
                "a baseline curving up towards the red rather than a flat "
                "offset, which inflates fitted amplitude and FWHM.",
                UserWarning, stacklevel=2,
            )

        self.wavelength = arr[0]                      # nm, ascending
        self.spectra    = arr[1].astype(float)        # raw counts, wavelength space

        # Energy axis (ascending) and matching energy-space spectra.
        energy        = HC_EV_NM / self.wavelength    # descending
        self._sort_idx = np.argsort(energy)
        self.energy   = energy[self._sort_idx]

        self.energy_spectra = self._to_energy(self.spectra)

        # Background subtracted in wavelength space first, then to energy.
        if self.bg_region_nm is not None:
            self.spectra_bg = subtract_background(
                self.spectra, bg_region=self.bg_region_nm,
                x=self.wavelength, axis=0,
            )
            self.energy_spectra_bg = self._to_energy(self.spectra_bg)
        else:
            self.spectra_bg        = None
            self.energy_spectra_bg = None

    # --- Private helpers ---------------------------------------------------

    def _to_energy(self, spectra: np.ndarray) -> np.ndarray:
        """Apply the Jacobian (if enabled) and reorder onto the energy axis."""
        y = (jacobian_correction_wvl2E(spectra, self.wavelength, axis=0)
             if self.apply_jacobian else spectra)
        return y[self._sort_idx]

    # --- Convenience properties --------------------------------------------

    @property
    def n_pixels(self) -> int:
        """Number of spectrometer pixels."""
        return self.spectra.shape[0]

    @property
    def best_spectra(self) -> np.ndarray:
        """Background-subtracted wavelength-space counts if available, else raw."""
        return self.spectra_bg if self.spectra_bg is not None else self.spectra

    @property
    def best_energy_spectra(self) -> np.ndarray:
        """Background-subtracted energy-space counts if available, else raw."""
        return (self.energy_spectra_bg
                if self.energy_spectra_bg is not None
                else self.energy_spectra)

    def __repr__(self) -> str:
        bg_str = (
            f"  BG region: {self.bg_region_nm[0]:.1f} – {self.bg_region_nm[1]:.1f} nm\n"
            if self.bg_region_nm is not None else ""
        )
        return (
            f"SingleSpectrum — {self.n_pixels} pixels\n"
            f"  File     : {self.path}\n"
            f"  λ range  : {self.wavelength.min():.1f} – {self.wavelength.max():.1f} nm"
            f"  ({self.energy.min():.3f} – {self.energy.max():.3f} eV)\n"
            f"{bg_str}"
            f"  Jacobian : {'applied' if self.apply_jacobian else 'not applied'}\n"
        )


# ---------------------------------------------------------------------------
# RamanSpectrum
# ---------------------------------------------------------------------------

class RamanSpectrum:
    """
    Single Raman spectrum loaded from a LabRAM-style ``.txt`` export.

    The file is a block of ``#``-prefixed metadata lines (acquisition
    settings, instrument, stage position, ...) of no fixed length, followed
    by two whitespace-separated data columns:

    * **Column 0** : Raman shift, in cm⁻¹ (ascending).
    * **Column 1** : counts (intensity).

    The header is skipped by content — any line starting with ``#`` — not by
    a fixed row count: it varies between files (e.g. extra ``#Peaks:Edit=``
    lines appear only on acquisitions someone has annotated in the vendor
    software), so counting rows would silently misalign the data on a file
    with a differently-sized header.

    Parameters
    ----------
    path : str or Path
        Path to the ``.txt`` file.

    Attributes
    ----------
    shift : np.ndarray, shape (n_points,)
        Raman shift axis in cm⁻¹, in whatever order the file stores it
        (ascending in every file seen so far).
    counts : np.ndarray, shape (n_points,)
        Raw intensity. Never modified after loading.
    path : str

    Notes
    -----
    Header fields (acquisition time, laser, grating, stage X/Y/Z, ...) are
    not parsed — nothing downstream uses them yet. Open the file directly if
    you need one.

    Reads a single spectrum only. A spatial Raman map (one file, many
    (X, Y, spectrum) rows) is a different export shape and needs
    :class:`RamanMap`; this class raises rather than misreading one as a
    lone spectrum.
    """

    def __init__(self, path: str):
        self.path = str(path)

        arr = np.loadtxt(path, comments="#", encoding="latin-1")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected 2 columns (Raman shift, counts), got array of "
                f"shape {arr.shape} from '{path}'. A wider array is likely a "
                f"spatial-map export (X, Y, then one counts column per "
                f"shift) -- use RamanMap for that shape."
            )

        self.shift  = arr[:, 0]
        self.counts = arr[:, 1]

    def __repr__(self) -> str:
        return (
            f"RamanSpectrum — {self.shift.size} points\n"
            f"  File  : {self.path}\n"
            f"  Shift : {self.shift.min():.1f} – {self.shift.max():.1f} cm⁻¹\n"
        )


# ---------------------------------------------------------------------------
# RamanMap
# ---------------------------------------------------------------------------

class RamanMap:
    """
    2-D spatial Raman map loaded from a LabRAM-style ``.txt`` export.

    Same ``#``-prefixed header block as :class:`RamanSpectrum`, but followed
    by a different body:

    * **Row 0** : two empty leading fields, then the Raman shift axis in
      cm⁻¹ (ascending) — shared by every spectrum in the map.
    * **Row 1, 2, ...** : one row per scan position,
      ``[X, Y, counts_0, counts_1, ..., counts_{n-1}]``, X/Y in µm (per the
      file's own ``#AxisUnit[2]``/``[3]`` fields).

    ``Y`` is the fast (inner) axis and ``X`` the slow (outer) one in the
    reference export (a full Y sweep at each X before X steps) — read off
    the row order and each row's own (X, Y), not assumed from a fixed
    convention, so a differently-ordered file with the same shape still
    loads correctly.

    Parameters
    ----------
    path : str or Path
        Path to the ``.txt`` file.

    Attributes
    ----------
    shift : np.ndarray, shape (n_shift,)
        Raman shift axis in cm⁻¹, ascending.
    x : np.ndarray, shape (n_x,)
        Scan X positions in µm, ascending.
    y : np.ndarray, shape (n_y,)
        Scan Y positions in µm, ascending.
    counts : np.ndarray, shape (n_y, n_x, n_shift)
        Raw intensity, reshaped onto the (Y, X) grid. Never modified after
        loading.
    path : str

    Raises
    ------
    ValueError
        If the number of spectra rows does not equal ``len(x) * len(y)``
        exactly — an aborted or ragged scan, which this class does not try
        to reshape partially.
    """

    def __init__(self, path: str):
        self.path = str(path)

        with open(path, encoding="latin-1") as fh:
            lines = fh.readlines()
        data_lines = [line for line in lines if not line.startswith("#") and line.strip()]
        if len(data_lines) < 2:
            raise ValueError(
                f"'{path}' has no data rows after its header — expected a "
                f"shift-axis row followed by at least one (X, Y, counts...) row."
            )

        shift_fields = data_lines[0].rstrip("\n").split("\t")
        self.shift = np.array([float(v) for v in shift_fields[2:] if v.strip() != ""])
        n_shift = self.shift.size

        rows = [line.rstrip("\n").split("\t") for line in data_lines[1:]]
        xy     = np.array([[float(r[0]), float(r[1])] for r in rows])
        counts_flat = np.array([[float(v) for v in r[2:2 + n_shift]] for r in rows])

        self.x = np.unique(xy[:, 0])
        self.y = np.unique(xy[:, 1])
        n_x, n_y = self.x.size, self.y.size
        if n_x * n_y != len(rows):
            raise ValueError(
                f"'{path}' has {len(rows)} spectra row(s), but {n_x} unique X "
                f"x {n_y} unique Y = {n_x * n_y} -- not a complete grid. An "
                f"aborted or partly copied scan is not reshaped partially; "
                f"read the file's own rows directly if you need them."
            )

        # Index each row by its own (X, Y) rather than assuming file order,
        # so a differently-ordered file with the same shape still loads
        # correctly.
        ix = np.searchsorted(self.x, xy[:, 0])
        iy = np.searchsorted(self.y, xy[:, 1])
        self.counts = np.empty((n_y, n_x, n_shift))
        self.counts[iy, ix, :] = counts_flat

    @property
    def n_x(self) -> int:
        """Number of X positions."""
        return self.x.size

    @property
    def n_y(self) -> int:
        """Number of Y positions."""
        return self.y.size

    def spectrum_at(self, ix: int, iy: int) -> np.ndarray:
        """Counts at grid position ``(ix, iy)``, shape ``(n_shift,)``."""
        return self.counts[iy, ix, :]

    def nearest_index(self, x: float, y: float) -> tuple:
        """
        Grid indices ``(ix, iy)`` of the scan position nearest ``(x, y)``, in µm.

        For picking a point off a plotted map (whose axes are in µm) rather
        than an exact grid coordinate — snaps to the nearest actual scan
        position instead of raising when the two don't match exactly.

        Returns
        -------
        ix, iy : int
        """
        ix = int(np.argmin(np.abs(self.x - x)))
        iy = int(np.argmin(np.abs(self.y - y)))
        return ix, iy

    def __repr__(self) -> str:
        return (
            f"RamanMap — {self.n_x} x {self.n_y} = {self.n_x * self.n_y} points\n"
            f"  File  : {self.path}\n"
            f"  X     : {self.x.min():.2f} – {self.x.max():.2f} µm\n"
            f"  Y     : {self.y.min():.2f} – {self.y.max():.2f} µm\n"
            f"  Shift : {self.shift.min():.1f} – {self.shift.max():.1f} cm⁻¹\n"
        )


# ---------------------------------------------------------------------------
# AttoCubePLScanRealSpace
# ---------------------------------------------------------------------------

# Rows a bare numeric grid needs before it counts as a frame.  Two rows is a
# single spectrum (row 0 wavelength, row 1 counts), which is numeric on its first
# line exactly like an image and so cannot be told apart by parsing that line.
_IMAGE_MIN_ROWS = 3

# What each _classify_csv kind is, and where it should go instead — shared by the
# "nothing usable here" error and the "these were skipped" warning so one file
# kind is described one way.  The two _BLOCK_LAYOUTS kinds take their class name
# from _CLASS_FOR_KIND rather than repeating it.
_CSV_KIND_REASON = {
    "spectrum"     : "a two-row single spectrum (row 0 wavelength in nm, row 1 "
                     "counts) — load it with SingleSpectrum",
    "spectral"     : f"a spectral export — load it with {_CLASS_FOR_KIND['spectral']}",
    "temporal"     : f"a temporal export — load it with {_CLASS_FOR_KIND['temporal']}",
    "too_short"    : "a single row of numbers",
    "unrecognised" : "headed, but by no known AttoCube block layout",
    "unreadable"   : "empty or unreadable",
}

# Kinds that belong in an image directory and are therefore dropped in silence: an
# acquisition writes its parameter export alongside the frames every time.  Every
# other kind is named in a warning, having no routine reason to be there.
_EXPECTED_NON_IMAGE_KINDS = frozenset(_CLASS_FOR_KIND)


def _classify_csv(path: Path) -> str:
    """
    Name what kind of CSV *path* is, from its opening lines alone.

    Cheap enough to run over a whole directory: at most three lines are read,
    so a multi-hundred-MB export costs the same as a small one.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str
        ``"image"`` for a numeric grid of at least three rows;
        ``"spectrum"`` for one of exactly two (a wavelength axis and its
        counts); ``"too_short"`` for one of a single row; ``"spectral"`` or
        ``"temporal"`` for an AttoCube export, named as in its header;
        ``"unrecognised"`` for a header matching no known block layout; and
        ``"unreadable"`` for an empty or unopenable file.

    See Also
    --------
    _read_block_layout : names the two export layouts, and is what this
        defers to once a text header is seen.
    """
    try:
        with open(path, "r") as fh:
            first_line = fh.readline()
            if not first_line.strip():
                return "unreadable"
            try:
                # Only the first field, and split no further: a frame's row is
                # wide, and one field is all that separates a numeric grid from
                # a header whose leading column reads "Parameters Labels".
                float(first_line.split(",", 1)[0].strip())
            except ValueError:
                header = True
            else:
                header = False
                # The handle sits on row 1, so two more lines settle whether a
                # third exists.  Nothing beyond _IMAGE_MIN_ROWS is read, so this
                # is a threshold test and not a row count.
                n_rows = 1 + _n_rows_upto(fh, _IMAGE_MIN_ROWS - 1)
    except OSError:
        return "unreadable"

    if not header:
        if n_rows >= _IMAGE_MIN_ROWS:
            return "image"
        return "spectrum" if n_rows == 2 else "too_short"

    # Headed files are classified where every other loader classifies them,
    # so "spectral" and "temporal" keep meaning the block layouts they name.
    try:
        return _read_block_layout(path)["kind"]
    except ValueError:
        return "unrecognised"


class AttoCubePLScanRealSpace:
    """
    Loader for a gate-dependent sequence of real-space PL images from the
    AttoCube cryogenic confocal.

    Frames are the pure numeric CSVs (no header row) matching ``{prefix}*.csv``
    in *path*, and must have **at least three rows** — a two-row grid is a single
    spectrum rather than an image, and is excluded along with anything carrying a
    text header.  An excluded file is named in a warning unless it is an AttoCube
    export, which an acquisition routinely writes beside its frames.

    Frames are ordered by the integer in their ``_iter_N`` filename suffix, so
    ``iter_10`` follows ``iter_2`` rather than preceding it.  A file with no such
    suffix, and a gap in the sequence, are both warned about; a gap is never
    closed up, so ``load_frame(i)`` is not necessarily iteration ``i``.

    Parameters
    ----------
    path : str or Path
        Directory containing the ``.csv`` files.
    prefix : str
        Common filename prefix, e.g. ``"PLdualgatesweep_iter_"``.
    geometry : DeviceGeometry, optional
        Device geometry. Stored for reference but not currently used to
        compute a field axis (gate voltages are not embedded in these files).
    laser_ref : AttoCubeLaserReferenceImage, optional
        Laser spot reference, used for annotation in animations.
    bg_region : tuple of (row_slice, col_slice), optional
        Pixel-space patch of signal-free detector used to estimate a dark
        pedestal, e.g. ``(slice(0, 40), slice(0, 40))`` for a corner.  Supplying
        it makes :meth:`load_frame_bg` available; :meth:`load_frame` is
        unaffected and always returns the file's own counts.
    bg_stat : {``"median"``, ``"mean"``}
        Statistic used to reduce *bg_region* to the pedestal.  ``"median"``
        survives a cosmic ray inside the patch; ``"mean"`` does not.
    """

    def __init__(
        self,
        path      : str,
        prefix    : str,
        geometry  : DeviceGeometry = None,
        laser_ref : "AttoCubeLaserReferenceImage" = None,
        bg_region : tuple = None,
        bg_stat   : str   = "median",
    ):
        if bg_stat not in _BG_STATS:
            # _apply_bg_region falls through to the mean for anything that is not
            # "median", so an unchecked typo silently changes the estimator.
            raise ValueError(
                f"bg_stat {bg_stat!r} is not recognised. "
                f"Choose from: {list(_BG_STATS)}."
            )

        self.path      = str(path)
        self.geometry  = geometry
        self.laser_ref = laser_ref
        self.bg_region = bg_region      # (row_slice, col_slice) in pixel space
        self.bg_stat   = bg_stat

        candidates = sorted(Path(path).glob(f"{prefix}*.csv"))
        kinds      = {f: _classify_csv(f) for f in candidates}
        images     = [f for f, kind in kinds.items() if kind == "image"]
        skipped    = {f: kind for f, kind in kinds.items() if kind != "image"}
        if not images:
            raise ValueError(
                f"No real-space image CSV files found with prefix '{prefix}' in '{path}'. "
                f"Found {len(candidates)} candidate(s), none of them an image of at "
                f"least {_IMAGE_MIN_ROWS} rows:\n"
                + self._describe_skipped(skipped)
            )
        # An export beside the frames is the expected case and says nothing; a
        # spectrum or an unreadable file among them means the directory is not
        # what the caller thinks it is, and silently dropping it is how a
        # half-loaded sequence reaches a diffusion fit.
        unexpected = {f: kind for f, kind in skipped.items()
                      if kind not in _EXPECTED_NON_IMAGE_KINDS}
        if unexpected:
            warnings.warn(
                f"Skipped {len(unexpected)} file(s) in '{path}' that are not "
                f"real-space images:\n" + self._describe_skipped(unexpected),
                stacklevel=2,
            )
        # Acquisition order, read from `_iter_N` — not the glob's lexicographic
        # order, which puts iter_10 before iter_2 on any export whose padding is
        # absent or narrower than the frame count. Every frame would then carry
        # the wrong index into animations and into
        # analyse_diffusion_sequence(var_array=…).
        # stacklevel 3: the helper, this __init__, the caller's line.
        self.files = _order_by_iter(images, path, stacklevel=3)

    @staticmethod
    def _describe_skipped(skipped: dict) -> str:
        """One indented ``name — why`` line per entry of a ``{Path: kind}`` map."""
        return "\n".join(
            f"  {f.name} — {_CSV_KIND_REASON.get(kind, kind)}"
            for f, kind in skipped.items()
        )

    def load_frame(self, idx: int) -> np.ndarray:
        """
        Load frame *idx* as a 2-D array of the file's own counts.

        Never background-corrected, whatever was passed as *bg_region* — see
        :meth:`load_frame_bg` for that.

        Returns
        -------
        np.ndarray
            Shape ``(ny, nx)``.
        """
        return np.loadtxt(self.files[idx], delimiter=",")

    def load_frame_bg(self, idx: int) -> np.ndarray:
        """
        Load frame *idx* with the *bg_region* pedestal subtracted.

        Returns
        -------
        np.ndarray
            Shape ``(ny, nx)``.  A new array; :meth:`load_frame` is unaffected.

        Raises
        ------
        ValueError
            If no *bg_region* was supplied at load time.  There is no pedestal
            to estimate, and returning the raw frame would silently answer a
            different question.
        """
        if self.bg_region is None:
            raise ValueError(
                f"No background-corrected frames on this scan: '{self.path}' was "
                f"loaded without a bg_region. Pass bg_region=(row_slice, col_slice) "
                f"to the constructor, or use load_frame() for the raw counts."
            )
        return processing._apply_bg_region(
            self.load_frame(idx), self.bg_region, self.bg_stat
        )

    @property
    def n_frames(self) -> int:
        """Number of frames loaded."""
        return len(self.files)

    def preview_image(self, idx: int, cmap = "gray", frame_source: str = "best") -> tuple:
        """
        Plot a single frame and return ``(fig, ax)``.

        Parameters
        ----------
        idx : int
            Frame index.
        cmap : str or Colormap
        frame_source : {``"best"``, ``"raw"``, ``"bg"``}
            Which frame to draw.  ``"best"`` is background-corrected when a
            *bg_region* was supplied at load time, and raw otherwise.
        """
        fig, ax = plt.subplots()
        ax.imshow(_resolve_frame(self, idx, frame_source), cmap)
        ax.axis("off")
        return fig, ax


# ---------------------------------------------------------------------------
# Shared base for single-image classes
# ---------------------------------------------------------------------------

class _AttoCubeImage:
    """
    Base class for single grayscale images loaded from a CSV.

    Provides :meth:`load_image`, :meth:`show_image`, and the shared laser
    circle annotation logic so subclasses do not duplicate it.
    """

    def __init__(self, path: str, laser_ref: "AttoCubeLaserReferenceImage" = None, bg_region : tuple = None, bg_stat : str = "median"):
        self.path      = str(path)
        self.laser_ref = laser_ref
        self.bg_region = bg_region
        self.bg_stat   = bg_stat
        self.img_raw       = np.loadtxt(self.path, delimiter=",")
        self.img = (
            processing._apply_bg_region(self.img_raw, bg_region, bg_stat)
            if bg_region is not None else self.img_raw
        )

    # --- Internal helpers --------------------------------------------------

    @staticmethod
    def _add_laser_circle(
        ax,
        laser_ref : "AttoCubeLaserReferenceImage",
        linewidth : float = 1,
    ) -> patches.Circle:
        """
        Draw the 1/e² laser boundary circle on *ax*, and return the Circle artist.

        Draws only. The legend is the caller's, because an axes can carry more than
        one labelled artist and a legend built here could only ever list this one.
        """
        circle = patches.Circle(
            (laser_ref.center_x, laser_ref.center_y),
            radius    = laser_ref.radius,
            edgecolor = "red",
            facecolor = "none",
            linewidth = linewidth,
            linestyle = "--",
            label     = f"$1/e^2$ Radius ({laser_ref.radius:.1f} px)",
        )
        ax.add_patch(circle)
        return circle

    def __array__(self, dtype=None):
        return np.asarray(self.img, dtype=dtype)


    # --- Public interface --------------------------------------------------

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        """
        Return the image data as a NumPy array.

        Parameters
        ----------
        copy : bool
            If True, return a copy of the image data. If False, return the
            underlying array directly.
        """
        return self.img.copy() if copy else self.img

    def show_image(
        self,
        img              = None,
        laser_annotation : bool = False,
        legend           : bool = False,
        normalise        : bool = False,
        show_bg_region   : bool = False,
        bg_region_color  : str  = "orange",
    ) -> tuple:
        """
        Display the image and return (fig, ax).

        Parameters
        ----------
        img : np.ndarray, optional
            Image to display. Uses ``self.img`` if ``None``.
        laser_annotation : bool
            Overlay the 1/e² laser spot boundary if a ``laser_ref`` is set.
        legend : bool
            Label whatever was drawn on top of the image — the laser circle, the
            background-region box, or both. Nothing else switches it on: asking
            for an annotation does not imply a legend.
        normalise : bool
            Rescale intensity to [0, 1] before display.
        show_bg_region : bool
            Outline the region whose statistic was subtracted at construction.
            Warns and draws nothing if this image was built without a
            *bg_region* — some classes take none at all, e.g.
            :class:`AttoCubeSampleImage`.
        bg_region_color : str
            Edge colour of that outline.

        Returns
        -------
        fig, ax

        Warns
        -----
        UserWarning
            If *show_bg_region* is asked for and there is no region to draw.
        """
        display = self.img if img is None else img
        if normalise:
            display = rescale_intensity(display, in_range="image", out_range=(0, 1))

        fig, ax = plt.subplots()
        ax.imshow(display, cmap="gray")
        ax.axis("off")

        # One legend, built here from what was actually drawn.  Neither drawer can
        # build it: `ax.legend(handles=[...])` takes an explicit list, so a legend
        # raised by either one would silently omit the other's artist.
        handles = []
        if show_bg_region:
            if self.bg_region is None:
                warnings.warn(
                    f"show_bg_region=True, but this {type(self).__name__} was "
                    f"constructed without a bg_region, so no box was drawn.",
                    UserWarning, stacklevel=2,
                )
            box = processing._draw_region_box(ax, self.bg_region, bg_region_color,
                                             label="bg region")
            if box is not None:
                handles.append(box)
        if laser_annotation and self.laser_ref is not None:
            handles.append(self._add_laser_circle(ax, self.laser_ref))
        if legend and handles:
            ax.legend(handles=handles, loc="upper right")

        return fig, ax


# ---------------------------------------------------------------------------
# AttoCubeSampleImage
# ---------------------------------------------------------------------------

class AttoCubeSampleImage(_AttoCubeImage):
    """
    White-light reference image of the sample taken on the AttoCube confocal.

    Use in conjunction with :class:`AttoCubeLaserReferenceImage` to locate
    the laser spot on the sample.

    Parameters
    ----------
    path : str or Path
        Path to the CSV image file.
    laser_ref : AttoCubeLaserReferenceImage, optional
        Laser spot reference for annotation.

    Notes
    -----
    Takes **no background region**, unlike :class:`AttoCubePLImage`, and
    :attr:`bg_region` is therefore always ``None`` on an instance of this
    class.  Both reasons are about what a white-light frame is rather than
    about this class:

    * A patch of a reflection image is substrate, and substrate reflects.  Its
      median estimates the substrate's own reflectance, not a detector
      pedestal, so subtracting it gives counts measured from the substrate —
      which reads like a contrast without being one, a reflectance contrast
      being a dimensionless ratio.
    * The correction such an image wants is a **ratio** against a reference
      frame, not a constant taken off it.  A single scalar is neither the dark
      frame nor the reference.

    Where a white-light background genuinely has to go, it is removed with an
    estimator shaped like it: :class:`AttoCubeLaserReferenceImage` runs a white
    top-hat before fitting the spot, because the background there is flake
    contrast and reflectivity gradients rather than a constant.
    """

    # Load-bearing despite forwarding nothing: deleting this override would
    # inherit the base's bg_region/bg_stat parameters, which is what the Notes
    # above refuse.  laser_ref is the only extra this class accepts.
    def __init__(self, path: str, laser_ref: "AttoCubeLaserReferenceImage" = None):
        super().__init__(path, laser_ref)


# ---------------------------------------------------------------------------
# AttoCubeLaserReferenceImage
# ---------------------------------------------------------------------------

class AttoCubeLaserReferenceImage(_AttoCubeImage):
    """
    Laser-spot reference image taken on the AttoCube cryogenic confocal.

    On construction the laser-spot centre and 1/e² radius are extracted with a
    pipeline that is robust to white-light illumination (where the spot sits on
    a structured background of flake contrast and reflectivity gradients):

    1. Median filter to despeckle hot pixels.
    2. **White top-hat** background suppression (optional, on by default): the
       laser PSF is a compact bright blob while white light is large-scale
       background.  A structuring element larger than the spot keeps the laser
       and removes the broad illumination/flake structure.
    3. Robust threshold + connected components → pick the brightest *compact,
       round* region (rejecting elongated flake edges) → intensity-weighted
       centroid as the seed.
    4. **2-D Gaussian fit with a tilted-plane baseline** on a local window
       around the seed → precise centre and σ (radius = 2σ).
    5. If the fit fails or returns an implausible σ, fall back to the weighted
       centroid and a second-moment radius estimate.

    Set ``white_light=False`` to skip the top-hat step for clean dark-background
    images.

    Parameters
    ----------
    path : str or Path
        Path to the CSV image file.
    expected_radius_px : float
        Approximate 1/e² laser radius in pixels.  Seeds the fit and sizes the
        top-hat structuring element and fit window.  Default ``10.0``.
    white_light : bool
        Apply white top-hat background suppression before fitting.  Default
        ``True``; set ``False`` for clean dark-background reference images.
    tophat_radius : float, optional
        Override the top-hat structuring-element radius in pixels.  ``None``
        (default) uses ``2.5 * expected_radius_px`` (must exceed the spot size).

    Attributes
    ----------
    center_x, center_y : float
        Fitted laser-spot centre (column, row) in pixels.
    radius : float
        Fitted 1/e² radius in pixels (``2σ``).
    """

    def __init__(
        self,
        path               : str,
        expected_radius_px : float = 10.0,
        white_light        : bool  = True,
        tophat_radius      : float = None,
    ):
        super().__init__(path, laser_ref=None)   # no external ref needed
        self.expected_radius_px = float(expected_radius_px)
        self.white_light        = bool(white_light)
        self.tophat_radius      = tophat_radius
        self.center_x, self.center_y, self.radius = self._fit_laser_spot()

    # --- Preprocessing -----------------------------------------------------

    def _preprocess(self) -> np.ndarray:
        """
        Despeckle and (optionally) remove the white-light background.

        Returns a non-negative image in which the laser spot dominates.
        """
        img = ndi.median_filter(np.nan_to_num(self.img.astype(float)), size=3)
        if self.white_light:
            r = self.tophat_radius
            if r is None:
                r = max(3, int(round(2.5 * self.expected_radius_px)))
            img = white_tophat(img, footprint=disk(int(r)))
        return img - img.min()           # ensure non-negative for weighting

    # --- Seed detection ----------------------------------------------------

    def _seed(self, proc: np.ndarray) -> tuple:
        """
        Robust (x0, y0) seed for the laser centre.

        Thresholds the preprocessed image, labels connected components, and
        picks the brightest *compact, round* region (rejecting elongated flake
        edges), returning its intensity-weighted centroid.  Falls back to the
        global intensity centroid if no region qualifies.
        """
        ny, nx = proc.shape
        if not np.any(proc > 0):
            return nx / 2.0, ny / 2.0
        try:
            thr = threshold_otsu(proc)
        except Exception:                       # noqa: BLE001 - degenerate image
            thr = float(np.median(proc) + 3 * np.std(proc))

        mask = proc > thr
        if not mask.any():
            cy, cx = ndi.center_of_mass(proc)
            return float(cx), float(cy)

        regions = regionprops(label(mask), intensity_image=proc)
        exp_area = np.pi * self.expected_radius_px ** 2
        good = [
            r for r in regions
            if r.eccentricity < 0.85 and 0.1 * exp_area <= r.area <= 10 * exp_area
        ]
        pool = good or regions
        best = max(pool, key=lambda r: r.intensity_mean * r.area)  # total intensity
        cy, cx = best.centroid_weighted                            # (row, col)
        return float(cx), float(cy)

    # --- Gaussian fitting --------------------------------------------------

    @staticmethod
    def _gaussian2d_plane(coords, A, x0, y0, sigma, c0, cx, cy):
        """Isotropic 2-D Gaussian plus a tilted-plane baseline."""
        x, y = coords
        g = A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))
        return (g + c0 + cx * x + cy * y).ravel()

    def _window(self, proc: np.ndarray, x0: float, y0: float) -> tuple:
        """Return (sub, X, Y) for a local window of ±3·expected_radius."""
        ny, nx = proc.shape
        half = int(round(3 * self.expected_radius_px)) + 1
        x_lo, x_hi = max(0, int(x0) - half), min(nx, int(x0) + half + 1)
        y_lo, y_hi = max(0, int(y0) - half), min(ny, int(y0) + half + 1)
        sub = proc[y_lo:y_hi, x_lo:x_hi]
        X, Y = np.meshgrid(np.arange(x_lo, x_hi), np.arange(y_lo, y_hi))
        return sub, X, Y

    def _fit_2d(self, proc: np.ndarray, x0: float, y0: float) -> tuple:
        """2-D Gaussian + plane fit on a local window. Returns (cx, cy, sigma)."""
        sub, X, Y = self._window(proc, x0, y0)
        base = float(np.median(sub))
        p0 = [float(sub.max() - base), x0, y0, self.expected_radius_px,
              base, 0.0, 0.0]
        lo = [0.0, X.min(), Y.min(), 0.5, -np.inf, -np.inf, -np.inf]
        hi = [np.inf, X.max(), Y.max(),
              max(2 * self.expected_radius_px, (X.max() - X.min()) + 1),
              np.inf, np.inf, np.inf]
        popt, _ = curve_fit(
            self._gaussian2d_plane, (X.ravel(), Y.ravel()), sub.ravel(),
            p0=p0, bounds=(lo, hi), maxfev=10000,
        )
        return float(popt[1]), float(popt[2]), abs(float(popt[3]))

    def _fallback(self, proc: np.ndarray, x0: float, y0: float) -> tuple:
        """Intensity-weighted centroid + second-moment radius on a window."""
        sub, X, Y = self._window(proc, x0, y0)
        sub = np.clip(sub - np.median(sub), 0, None)
        total = sub.sum()
        if total <= 0:
            return x0, y0, 2.0 * self.expected_radius_px
        cx = float((sub * X).sum() / total)
        cy = float((sub * Y).sum() / total)
        var = float((sub * ((X - cx) ** 2 + (Y - cy) ** 2)).sum() / total)
        sigma = np.sqrt(var / 2.0)          # <r²> = 2σ² for a 2-D Gaussian
        return cx, cy, 2.0 * sigma

    def _fit_laser_spot(self) -> tuple:
        """Return (center_x, center_y, 1/e² radius) using the robust pipeline."""
        proc = self._preprocess()
        x0, y0 = self._seed(proc)
        ny, nx = proc.shape
        try:
            cx, cy, sigma = self._fit_2d(proc, x0, y0)
            plausible = (0 <= cx < nx and 0 <= cy < ny
                         and 0.5 <= sigma <= 0.5 * max(ny, nx))
            if not plausible:
                raise RuntimeError("implausible fit result")
            return cx, cy, 2.0 * sigma
        except Exception:                       # noqa: BLE001 - graceful fallback
            return self._fallback(proc, x0, y0)

    # --- Display -----------------------------------------------------------

    def show_image(
        self,
        laser_annotation : bool = False,
        legend           : bool = False,
        normalise        : bool = False,
    ) -> tuple:
        """
        Display the laser reference image and return (fig, ax).

        Parameters
        ----------
        laser_annotation : bool
            Overlay the fitted 1/e² boundary circle.
        legend : bool
            Show a legend for the circle.
        normalise : bool
            Rescale intensity to [0, 1] before display.

        Returns
        -------
        fig, ax
        """
        # Provide self as own laser_ref so the base helper can draw the circle
        self.laser_ref = self
        fig, ax = super().show_image(
            laser_annotation=laser_annotation,
            legend=legend,
            normalise=normalise,
        )
        self.laser_ref = None   # reset — this class has no external reference
        return fig, ax

    # --- Dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AttoCubeLaserReferenceImage\n"
            f"  File                  : {self.path}\n"
            f"  Center                : ({self.center_x:.1f}, {self.center_y:.1f}) px\n"
            f"  Estimated 1/e² Radius : {self.radius:.1f} px\n"
            f"  Estimated 1/e² Diameter: {2 * self.radius:.1f} px"
        )


# ---------------------------------------------------------------------------
# SingleImage
# ---------------------------------------------------------------------------

class SingleImage(_AttoCubeImage):
    """
    Generic single 2-D image loaded from a numeric CSV grid.

    Exposes the image as :attr:`img`, so it can be displayed in grayscale via
    the inherited :meth:`show_image` or with a colormap (and optional colorbar)
    via :func:`~tmdc_optics_tools.plotting.plot_image`.

    Parameters
    ----------
    path : str or Path
        Path to the CSV image file (numeric grid, comma-delimited).
    """

    def __init__(self, path: str):
        super().__init__(path, laser_ref=None)

    def __repr__(self) -> str:
        return (
            f"SingleImage — {self.img.shape[0]} × {self.img.shape[1]} px\n"
            f"  File : {self.path}\n"
        )

class AttoCubePLImage(_AttoCubeImage):
    """A single real-space PL frame — same handling as AttoCubePLScanRealSpace,
    for spot-checking one file without building a full sequence."""
    def __init__(self, path, laser_ref=None, bg_region=None, bg_stat="median"):
        super().__init__(path, laser_ref=laser_ref, bg_region=bg_region, bg_stat=bg_stat)
