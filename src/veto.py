"""
Port of Veto.ipynb to a CLI.

Scans a night directory's sNNNN.fits frames, classifies non-flat/non-dark
frames by mean pixel counts in a central window, and writes veto_list.txt
listing frames outside the [min, max] counts window.

Also accepts a manual-veto list (space- or comma-separated integer frame IDs)
pulled from the obs log, and appends those to the veto list.
"""
from __future__ import annotations

import argparse
import math
import os
from glob import glob
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

# Shane AO flats come in sky / dome / lamp flavors. Use the centralized
# detector so veto.py matches run_night.py and the notebook behavior.
import sys as _sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from flat_utils import is_flat_object  # noqa: E402


def guess_gaussian_parameters(d: np.ndarray):
    """Return (x0, y0, sigma, A) for a rough Gaussian centroid + width.

    Ported verbatim from Veto.ipynb cell 1, minus the commented-out median
    filter step (noisy but faster — original author's choice).
    """
    window = int(np.sqrt(np.sqrt(d.shape[0] * d.shape[1])))
    if window % 2 == 0:
        window += 1
    d_g = gaussian_filter(d, sigma=window)
    y0, x0 = np.where(d_g == np.max(d_g))
    y0 = int(y0[0]); x0 = int(x0[0])
    x_cut = d[:, x0]
    sigma_x = (np.sum(x_cut * np.abs(np.arange(len(x_cut)) - y0)) / np.sum(x_cut)) / 3.0
    y_cut = d[y0, :]
    sigma_y = (np.sum(y_cut * np.abs(np.arange(len(y_cut)) - x0)) / np.sum(y_cut)) / 3.0
    sigma = np.sqrt(sigma_x * sigma_y)
    A = np.sum(d - np.median(d)) / (2.0 * np.pi * sigma ** 2)
    return x0, y0, sigma, 2.0 * A


def frame_counts(fits_path: str, center=(1100, 730), half_window=200) -> float:
    """Return the mean counts in a 3x3 box about the peak (flats: mean of whole window)."""
    header = fits.getheader(fits_path)
    obj = str(header.get("OBJECT", ""))
    data = fits.getdata(fits_path)
    cx, cy = center
    win = data[cy - half_window:cy + half_window, cx - half_window:cx + half_window]
    if is_flat_object(obj):
        return float(np.mean(win))
    try:
        x0, y0, _sigma, _A = guess_gaussian_parameters(win)
    except Exception:
        return float("nan")
    cent = win[max(y0 - 1, 0):y0 + 1, max(x0 - 1, 0):x0 + 1]
    return float(np.mean(cent)) if cent.size else float("nan")


def collect_frames(datadir: Path) -> list[Path]:
    return sorted(Path(datadir).glob("s[0-9][0-9][0-9][0-9].fits"))


def build_veto_list(datadir: Path, min_counts: float, max_counts: float,
                    manual_ids: set[int], verbose: bool) -> list[str]:
    veto = []
    for fp in collect_frames(datadir):
        try:
            hdr = fits.getheader(fp)
        except Exception:
            continue
        obj = str(hdr.get("OBJECT", ""))
        if "dark" in obj.lower():
            continue  # never veto darks
        try:
            c = frame_counts(str(fp))
        except Exception as e:
            if verbose: print(f"  {fp.name}: error {e} -> veto")
            veto.append(fp.name)
            continue
        if math.isnan(c) or c < min_counts or c > max_counts:
            if verbose: print(f"  {fp.name} obj={obj} counts={c:.0f} -> veto")
            veto.append(fp.name)
    # Manual vetoes from obs log
    for fid in sorted(manual_ids):
        name = f"s{fid:04d}.fits"
        if name not in veto:
            veto.append(name)
    return veto


def parse_manual_ids(spec: str | None) -> set[int]:
    if not spec:
        return set()
    out: set[int] = set()
    for tok in spec.replace(",", " ").split():
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--min-counts", type=float, default=500.0,
                    help="Reject frames with peak counts below this (default 500).")
    ap.add_argument("--max-counts", type=float, default=25000.0,
                    help="Reject frames with peak counts above this (default 25000).")
    ap.add_argument("--manual-ids", default="",
                    help="Space/comma-separated frame IDs and ranges to veto, e.g. '163 176-179'.")
    ap.add_argument("--out", default=None,
                    help="Output file path (default: <data-dir>/veto_list.txt).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    manual = parse_manual_ids(args.manual_ids)
    veto = build_veto_list(args.data_dir, args.min_counts, args.max_counts,
                           manual, args.verbose)

    out = Path(args.out) if args.out else args.data_dir / "veto_list.txt"
    with open(out, "w") as f:
        for v in veto:
            f.write(v + "\n")
    print(f"Wrote {out}: {len(veto)} vetoed frames")


if __name__ == "__main__":
    main()
