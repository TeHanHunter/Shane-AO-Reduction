"""
Flat-frame classification helper.

Centralizes the "is this FITS frame a flat?" test so that the Shane AO
reduction pipeline recognizes sky flats, dome flats, and lamp flats via
a single definition.

The original notebook (Image_Reduction_Plots.ipynb / Veto.ipynb) checked
only `'flat' in OBJECT.lower()`. That already catches common forms such as
"flat", "sky flat", "dome flat", and "lamp flat" -- any label containing
the substring "flat". This helper keeps that behavior as a superset and
additionally matches the few non-"flat" variants observed in Shane AO
headers:

    dome flats  OBJECT = "dome flat", "dome_flat", "dflat", "dome"
    sky flats   OBJECT = "flat", "sky flat", "skyflat"
    lamp flats  OBJECT = "lamps on", "lamps_off", "lamp flat"

Backwards compatible: any string that passed the old test still passes.
"""
from __future__ import annotations


FLAT_OBJECT_KEYWORDS = ("flat", "dflat", "dome", "lamp")


def is_flat_object(obj) -> bool:
    """True if the FITS OBJECT string looks like a flat-field frame.

    Accepts sky flats, dome flats (lamps on or off), and lamp flats.
    """
    if obj is None:
        return False
    s = str(obj).lower()
    return any(k in s for k in FLAT_OBJECT_KEYWORDS)
