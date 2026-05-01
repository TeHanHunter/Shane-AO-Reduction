"""Low-quality salvage reduction for ShaneAO targets where the dither
sequence is too sparse for proper sky subtraction.

Same pipeline as run_night.py but with sky subtraction replaced by a
zero-array master sky. The resulting stack retains the sky background
(median-stacking partially averages it out, but residual gradients
remain) and is therefore not science-grade. It is good enough to
answer "is this a binary?" for targets where we'd otherwise have no
imaging at all.

The output FITS has `QUALITY = LOW_NO_SKY_SUB` and `SALVAGE = True`
header cards so downstream consumers can flag it.

Usage:
    python src/run_night_no_sky.py --night 20241217 \
        --data-dir Raw_Data/data--web2-2024-12-17-AO-Paul.Robertson \
        --obs-log Raw_Data/2024B_obs_log.csv \
        --outdir Raw_Data/2024B_Results/20241217 \
        --only TIC_60764070
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

# Reuse run_night.py's CLI + reduce_target logic, but swap make_master_sky
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sky_subtraction  # noqa: E402

_orig_make_master_sky = sky_subtraction.make_master_sky


def _zero_master_sky(object_list, flat_darkcor_sigmacut_data, datadir):
    """Return a zero master sky per ITIME group; bypasses dither check."""
    print("  [salvage] using zero master sky (no sky subtraction)")
    exposuretimes = []
    for obj in object_list:
        header = fits.getheader(datadir + obj)
        t = float(header["ITIME"]) / 1000
        if t not in exposuretimes:
            exposuretimes.append(t)
    exp_dict = {}
    for time in exposuretimes:
        newlist = [
            o for o in object_list
            if float(fits.getheader(datadir + o)["ITIME"]) / 1000 == time
        ]
        exp_dict[time] = newlist
    # Pick the first frame's shape as the sky shape
    first_im = object_list[0]
    shape = flat_darkcor_sigmacut_data[first_im].shape
    master_sky_dict = {t: np.zeros(shape, dtype=np.float64)
                       for t in exposuretimes}
    # `center` is the LIST of center-dither frame names (image_shift
    # expects names, not coords). Without a real dither classification
    # we just hand it every frame so SNR-pick works.
    center_frames = list(object_list)
    return master_sky_dict, exp_dict, center_frames


sky_subtraction.make_master_sky = _zero_master_sky

# Now drive run_night.py's main with the swapped function
import run_night  # noqa: E402

# Wrap reduce_target to tag output FITS with QUALITY=LOW_NO_SKY_SUB
_orig_run_single = run_night._run_single_filter


def _tagged_run_single(*args, **kwargs):
    out_fits = _orig_run_single(*args, **kwargs)
    try:
        with fits.open(out_fits, mode="update") as hdul:
            hdul[0].header["QUALITY"] = ("LOW_NO_SKY_SUB",
                                         "Sky subtraction bypassed")
            hdul[0].header["SALVAGE"] = (True,
                                         "Salvage reduction, "
                                         "not science-grade")
            hdul.flush()
    except Exception as e:
        print(f"  [salvage] could not tag header on {out_fits}: {e}")
    return out_fits


run_night._run_single_filter = _tagged_run_single

if __name__ == "__main__":
    run_night.main()
