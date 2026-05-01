"""
Headless reduction driver for one ShaneAO night.

Ports the per-target logic from Image_Reduction_Plots.ipynb into a CLI.
Per target:
    1. Load sNNNN.fits headers, partition into object / flat / dark.
    2. Apply veto_list.txt + obs-log manual veto.
    3. Dark-correct object+flat frames (per exposure time).
    4. Build per-filter master flat; flat-field object frames.
    5. Sigma-clip on a fixed 600x600 central crop.
    6. Build master sky per exposure time (5-dither aware); sky-subtract.
    7. Shift-and-add stack via FFT cross-correlation.
    8. Write Results/<target>_final.fits (with WCS cutout header) +
       Results/<target>_filter_<FILT>.png + _linear.png.

Usage:
    python src/run_night.py --night 20241016 \
        --data-dir Raw_Data/data-2024-10-16-AO-Paul.Robertson \
        --obs-log Raw_Data/2024B_obs_log.csv \
        [--only TIC_352409590] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

# Allow running as `python src/run_night.py` from repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Neutralize the interactive plt.show() in weighted_centroid before import.
_orig_show = plt.show
plt.show = lambda *a, **k: None  # noqa: E731

from dark_correct import generate_master_darks, dark_correct  # noqa: E402
from flat_field import make_master_flats, flat_field  # noqa: E402
from sky_subtraction import make_master_sky, sky_subtract  # noqa: E402
from utils import sigma_clip, image_shift  # noqa: E402
from flat_utils import is_flat_object  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SKIP_KEYWORDS = ("dark", "flat", "alignment", "junk", "seeing", "optimization", "test")


def sanitize_target_name(obj: str) -> str:
    """Replace '-' with '_' and '+' with 't' (matches notebook cell 3)."""
    return obj.replace("-", "_").replace("+", "t")


def scan_headers(datadir: Path) -> dict[str, dict]:
    """Return {filename: {OBJECT, FILT1NAM, ITIME_s}} for every sNNNN.fits in datadir."""
    out = {}
    for fp in sorted(datadir.glob("s[0-9][0-9][0-9][0-9].fits")):
        try:
            h = fits.getheader(fp)
        except Exception:
            continue
        obj = str(h.get("OBJECT", "")).strip()
        try:
            itime_s = float(h["ITIME"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            itime_s = float("nan")
        out[fp.name] = {
            "OBJECT": obj,
            "FILT1NAM": str(h.get("FILT1NAM", "")).strip(),
            "ITIME_s": itime_s,
            "path": fp,
        }
    return out


def pick_target_frames(header_map: dict[str, dict], target_sanitized: str) -> list[str]:
    """Frames whose sanitized OBJECT name equals `target_sanitized`."""
    out = []
    for name, meta in header_map.items():
        if sanitize_target_name(meta["OBJECT"]) == target_sanitized:
            out.append(name)
    return sorted(out)


def pick_flat_dark(header_map: dict[str, dict]):
    # is_flat_object handles sky/dome/lamp flat variants (see flat_utils).
    flats = [n for n, m in header_map.items() if is_flat_object(m["OBJECT"])]
    darks = [n for n, m in header_map.items() if "dark" in m["OBJECT"].lower()]
    drkhdr = []
    for d in darks:
        drkhdr.append(int(header_map[d]["ITIME_s"]))
    return sorted(flats), sorted(darks), drkhdr


def load_raw(datadir: Path, names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for n in names:
        try:
            out[n] = fits.getdata(datadir / n)
        except Exception as e:
            print(f"  skip {n}: {e}")
    return out


def write_cutout_fits(datadir: Path, ref_frame: str, final_image: np.ndarray,
                      out_path: Path) -> None:
    """Write final_image with a WCS cutout header derived from `ref_frame`.

    Matches the WCS logic at the bottom of Image_Reduction_Plots.ipynb cell 14.
    Falls back to a bare primary HDU if any WCS key is missing — better to
    save the science than to abort.
    """
    try:
        with fits.open(datadir / ref_frame) as hdul:
            hdr = hdul[0].header.copy()
            wcs_keys = [
                "EQUINOX", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
                "CD1_1", "CD1_2", "CD2_1", "CD2_2",
                "CRDER1", "CRDER2", "CSYER1", "CSYER2",
                "CRPIX1C", "CRPIX2C", "CRVAL1C", "CRVAL2C",
                "CD1_1C", "CD1_2C", "CD2_1C", "CD2_2C",
                "CRDER1C", "CRDER2C", "CSYER1C", "CSYER2C",
            ]
            for k in wcs_keys:
                if k in hdr:
                    try:
                        hdr[k] = float(hdr[k])
                    except (TypeError, ValueError):
                        pass
            wcs = WCS(hdr)
            cutout = Cutout2D(hdul[0].data, (700, 1350), (600, 600),
                              wcs=wcs, mode="partial")
            cutout_hdr = cutout.wcs.to_header()
        fits.writeto(out_path, final_image, cutout_hdr, overwrite=True)
    except Exception as e:
        print(f"  WCS cutout failed ({e}), writing bare FITS")
        fits.writeto(out_path, final_image, overwrite=True)


def save_previews(final_image: np.ndarray, target: str, filt: str,
                  outdir: Path) -> None:
    """Two PNGs matching the reference-archive filenames."""
    # Log-stretched
    safe = np.where(final_image > 0, final_image, np.nan)
    fig = plt.figure(figsize=(15, 15))
    plt.title(f"{target}: Aligned and Stacked")
    plt.imshow(np.log10(safe), origin="lower", cmap="gray")
    fig.savefig(outdir / f"{target}_filter_{filt}.png", dpi=150)
    plt.close(fig)
    # Linear
    fig = plt.figure(figsize=(15, 15))
    plt.title(f"{target}: Aligned and Stacked")
    plt.imshow(final_image, origin="lower", cmap="gray")
    fig.savefig(outdir / f"{target}_filter_{filt}_linear.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core per-target reduction
# ---------------------------------------------------------------------------

def reduce_target(target_sanitized: str,
                  datadir: Path,
                  outdir: Path,
                  veto_set: set[str],
                  header_map: dict[str, dict],
                  min_frames: int = 5) -> dict:
    """Run the full pipeline for one target. Returns a status dict."""
    # 1. partition
    all_target_frames = pick_target_frames(header_map, target_sanitized)
    flat_list, dark_list, drkhdr_list = pick_flat_dark(header_map)

    # 2. remove vetoed frames from target list and flats
    object_list = [f for f in all_target_frames if f not in veto_set]
    flat_list = [f for f in flat_list if f not in veto_set]

    status = {
        "target": target_sanitized,
        "n_total": len(all_target_frames),
        "n_after_veto": len(object_list),
        "n_flats": len(flat_list),
        "n_darks": len(dark_list),
        "status": None,
        "filters": [],
        "outputs": [],
        "error": None,
    }

    if len(object_list) < min_frames:
        status["status"] = "skipped_too_few"
        status["error"] = f"only {len(object_list)} frames after veto"
        return status

    if len(flat_list) == 0:
        status["status"] = "skipped_no_flats"
        status["error"] = "no flat frames in night"
        return status

    # 3. bin by filter, process each filter separately
    # (The notebook processed object_filter_list2[0] only. We preserve that
    # behavior but loop over each filter the target was observed in.)
    filters_used = {header_map[f]["FILT1NAM"] for f in object_list}
    status["filters"] = sorted(filters_used)

    for filt in sorted(filters_used):
        obj_in_filt = [f for f in object_list if header_map[f]["FILT1NAM"] == filt]
        if len(obj_in_filt) < min_frames:
            print(f"  [{target_sanitized} / {filt}] skip: only {len(obj_in_filt)} frames")
            continue
        try:
            out_fits = _run_single_filter(
                target_sanitized, filt, obj_in_filt, flat_list, dark_list,
                drkhdr_list, datadir, outdir, len(filters_used),
            )
            status["outputs"].append(str(out_fits))
        except Exception as e:
            print(f"  [{target_sanitized} / {filt}] FAILED: {e}")
            traceback.print_exc()
            status["error"] = f"{filt}: {e}"

    status["status"] = "ok" if status["outputs"] else "failed"
    return status


def _run_single_filter(target: str, filt: str, object_list: list[str],
                       flat_list: list[str], dark_list: list[str],
                       drkhdr_list: list[int], datadir: Path, outdir: Path,
                       n_filters_total: int) -> Path:
    print(f"  [{target} / {filt}] {len(object_list)} object frames, "
          f"{len(flat_list)} flats, {len(dark_list)} darks")

    # Load raw data for everything we need
    needed = list(dict.fromkeys(flat_list + object_list + dark_list))
    raw = load_raw(datadir, needed)
    missing = [n for n in needed if n not in raw]
    if missing:
        print(f"    WARNING: {len(missing)} frames failed to load")

    # 4. Dark correct
    darkcor_in = [f for f in (object_list + flat_list) if f in raw]
    master_dark_dict = generate_master_darks(dark_list, drkhdr_list, str(datadir) + "/")
    darkcor = {}
    for im in darkcor_in:
        try:
            darkcor[im] = dark_correct(im, raw, master_dark_dict, str(datadir) + "/")
        except Exception as e:
            print(f"    dark_correct {im} failed: {e}")

    # 5. Master flat (per filter) + flat-field object frames
    master_flat = make_master_flats(flat_list, filt, darkcor, str(datadir) + "/")
    obj_have = [f for f in object_list if f in darkcor]
    flat_darkcor = flat_field(obj_have, master_flat, darkcor)

    # 6. Sigma clip (600x600 crop baked into helper)
    flat_darkcor_sc = sigma_clip(obj_have, flat_darkcor)

    # 6b. Drop sparse exposure-time groups. make_master_sky requires
    # >= 3 of 5 dither positions populated within each exposure-time group;
    # when only 1-4 frames exist at a given ITIME the notebook hits an
    # undefined-variable path ("exposuretimes[i]"/"explist[i]" / IndexError).
    # Filter those groups out before sky subtraction.
    from collections import Counter as _Counter
    itime_counts = _Counter()
    for n in obj_have:
        try:
            itime_counts[round(float(fits.getheader(datadir / n)["ITIME"]) / 1000.0, 3)] += 1
        except Exception:
            pass
    MIN_FRAMES_PER_ITIME = 5
    keep_itimes = {t for t, c in itime_counts.items() if c >= MIN_FRAMES_PER_ITIME}
    dropped_itimes = {t: c for t, c in itime_counts.items() if c < MIN_FRAMES_PER_ITIME}
    if dropped_itimes:
        print(f"    dropping sparse exposure groups (< {MIN_FRAMES_PER_ITIME} frames): "
              f"{dropped_itimes}")
    def _itime(name):
        try:
            return round(float(fits.getheader(datadir / name)["ITIME"]) / 1000.0, 3)
        except Exception:
            return None
    obj_have_full = obj_have
    obj_have = [n for n in obj_have if _itime(n) in keep_itimes]
    if len(obj_have) < 5:
        raise RuntimeError(
            f"after sparse-itime filter, only {len(obj_have)} frames remain "
            f"(counts={dict(itime_counts)})")
    flat_darkcor_sc = {n: flat_darkcor_sc[n] for n in obj_have if n in flat_darkcor_sc}

    # 7. Sky subtraction (dither aware, per exposure time).
    # make_master_sky may drop ITIME groups with <3 distinct dither
    # positions, so refresh obj_have to only include frames from the
    # surviving groups before passing to sky_subtract / image_shift.
    master_sky, exp_dict, center = make_master_sky(
        obj_have, flat_darkcor_sc, str(datadir) + "/")
    surviving_frames = set()
    for _t, _frames in exp_dict.items():
        surviving_frames.update(_frames)
    dropped = [n for n in obj_have if n not in surviving_frames]
    if dropped:
        print(f"    dropping {len(dropped)} frame(s) from sparse-dither ITIME "
              f"groups before stack")
    obj_have = [n for n in obj_have if n in surviving_frames]
    sky_flat_darkcor = sky_subtract(
        obj_have, list(master_sky.keys()), flat_darkcor_sc, exp_dict, master_sky)

    # 8. Shift-and-add stack
    final_image = image_shift(obj_have, center, sky_flat_darkcor, str(datadir) + "/")

    # 9. Save outputs
    outdir.mkdir(parents=True, exist_ok=True)
    ref_frame = obj_have[0]
    if n_filters_total > 1:
        out_fits = outdir / f"{target}_final_filter_{filt}.fits"
    else:
        out_fits = outdir / f"{target}_final.fits"
    write_cutout_fits(datadir, ref_frame, final_image, out_fits)
    save_previews(final_image, target, filt, outdir)
    print(f"    -> {out_fits.name}")
    return out_fits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_veto_file(datadir: Path) -> set[str]:
    vf = datadir / "veto_list.txt"
    if not vf.exists():
        return set()
    return {line.strip() for line in vf.read_text().splitlines() if line.strip()}


def obs_log_targets(obs_log: Path, night: str, only: str | None) -> list[str]:
    """Sanitized, deduplicated list of non-flat targets for a given night."""
    df = pd.read_csv(obs_log)
    # night may come in as '20241016' string or int
    night_int = int(str(night).replace("-", ""))
    df = df[df["night"] == night_int]
    df = df[~df["is_flat"].fillna(False).astype(bool)]
    targets = []
    seen = set()
    for t in df["target"]:
        if pd.isna(t):
            continue
        tn = sanitize_target_name(str(t))
        if tn in seen:
            continue
        seen.add(tn)
        targets.append(tn)
    if only:
        only_s = sanitize_target_name(only)
        targets = [t for t in targets if t == only_s]
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", required=True, help="e.g. 20241016")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--obs-log", required=True, type=Path)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output Results/ dir (default: <data-dir>/Results)")
    ap.add_argument("--only", default=None, help="Sanitized target name filter, "
                    "e.g. TIC_352409590 (optional)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List planned targets and exit without reducing")
    ap.add_argument("--min-frames", type=int, default=5,
                    help="Skip targets with fewer than this many surviving frames")
    args = ap.parse_args()

    datadir: Path = args.data_dir
    if not datadir.exists():
        raise SystemExit(f"data-dir does not exist: {datadir}")
    outdir = args.outdir or (datadir / "Results")

    targets = obs_log_targets(args.obs_log, args.night, args.only)
    print(f"Night {args.night}: {len(targets)} target(s) from obs log: {targets}")

    if args.dry_run:
        # Still load headers so we can report frame counts
        header_map = scan_headers(datadir) if datadir.exists() else {}
        veto_set = load_veto_file(datadir)
        for t in targets:
            frames = pick_target_frames(header_map, t) if header_map else []
            after = [f for f in frames if f not in veto_set]
            print(f"  {t}: {len(frames)} frames, {len(after)} after veto")
        return

    header_map = scan_headers(datadir)
    veto_set = load_veto_file(datadir)
    print(f"Veto list: {len(veto_set)} frames")

    summary = []
    for t in targets:
        print(f"\n=== {t} ===")
        try:
            res = reduce_target(t, datadir, outdir, veto_set, header_map,
                                min_frames=args.min_frames)
        except Exception as e:
            traceback.print_exc()
            res = {"target": t, "status": "exception", "error": str(e)}
        summary.append(res)

    print("\n=== Summary ===")
    for r in summary:
        print(f"  {r.get('target')}: {r.get('status')} "
              f"outputs={len(r.get('outputs', []))} error={r.get('error')}")


if __name__ == "__main__":
    main()
