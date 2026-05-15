"""
pytag_puncta_quantification_refactored.py

Minimal puncta quantification for pYtag lightsheet stacks with a labeled branch mask.

Assumptions (matching the user's workflow):
- You have a 3D fluorescence stack `img` with puncta on top of diffuse/background signal.
- You have a 3D labeled mask `mask` with 0 = outside branches, and 1..N = branch IDs.
- You care about puncta, not diffuse/background within the branch.

Workflow implemented here
-------------------------
1) Global robust normalization per image (uses the *entire* image):
      norm = (img - P25(img)) / IQR(img)
   where IQR = P75 - P25.

2) Global puncta threshold per image (computed on all *branch pixels* combined):
   - "median_k_iqr": pixels > median(all_branches) + k * IQR(all_branches)
   - "percentile":   pixels > Pp(all_branches)  (e.g. p=99)
   The resulting single threshold is then applied within each branch.

3) Per-branch puncta summary:
   - n_peaks: number of distinct puncta, counted as local maxima within puncta pixels
              with a user-defined minimum separation (via an ellipsoid footprint).
   - puncta_mean_intensity: mean normalized intensity of puncta pixels.

This module is intentionally short and opinionated (no heavy CLI / plugin system).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import measure, morphology


Array = np.ndarray
ThrMode = Literal["median_k_iqr", "percentile"]


# ----------------------------
# I/O
# ----------------------------

def load_tiff_stack(path: str) -> Array:
    """Load a TIFF stack into a numpy array."""
    return np.asarray(tifffile.imread(path))


# ----------------------------
# Branch-mask filtering (CSV lookup)
# ----------------------------

def _parse_embryo_num(embryo: Union[str, int]) -> int:
    """Accepts 'Embryo2' or '2' -> 2."""
    if isinstance(embryo, int):
        return embryo
    m = re.search(r"(\d+)", str(embryo))
    if not m:
        raise ValueError(f"Could not parse embryo number from {embryo!r}")
    return int(m.group(1))


def filter_mask_to_metameres(mask: Array, csv_path: str, embryo: Union[str, int], half: str) -> Array:
    """
    Keep only the label IDs listed in `csv_path` for the specified embryo/half.

    Expected CSV columns (as used in your notebook):
      - Embryo   (int)
      - Half     (string, e.g. 'Anterior' / 'Posterior')
      - Label1, Label2, ... (ints; empty cells allowed)

    Returns a new mask where all labels not in the row are set to 0.
    """
    df = pd.read_csv(csv_path)
    embryo_int = _parse_embryo_num(embryo)

    sub = df[(df["Embryo"] == embryo_int) & (df["Half"].str.lower() == half.strip().lower())]
    if sub.empty:
        raise ValueError(f"No CSV row found for Embryo={embryo_int}, Half={half!r} in {csv_path}")

    row = sub.iloc[0]
    label_cols = [c for c in df.columns if c.lower().startswith("label")]

    allowed: List[int] = [0]
    for c in label_cols:
        v = row.get(c, np.nan)
        if pd.notna(v):
            try:
                allowed.append(int(v))
            except Exception:
                pass

    allowed_arr = np.array(sorted(set(allowed)), dtype=mask.dtype)
    return np.where(np.isin(mask, allowed_arr), mask, 0)


# ----------------------------
# Normalization
# ----------------------------

def normalize_global_iqr(img: Array, p_low: float = 25.0, p_high: float = 75.0) -> Tuple[Array, Dict]:
    """
    Robust per-image normalization using the entire image.

    norm = (img - P_low) / (P_high - P_low)
    """
    img_f = img.astype(np.float32, copy=False)
    lo, hi = np.percentile(img_f, [p_low, p_high])
    iqr = float(hi - lo)
    if iqr <= 0:
        iqr = 1.0
    norm = (img_f - float(lo)) / iqr
    meta = {"p_low": float(p_low), "p_high": float(p_high), "p25": float(lo), "p75": float(hi), "iqr": float(iqr)}
    return norm, meta


# ----------------------------
# Puncta detection / counting
# ----------------------------

def ellipsoid_footprint(rz: int, rxy: int) -> Array:
    """Boolean ellipsoid footprint for local-max filtering in (Z,Y,X)."""
    rz = int(max(rz, 0))
    rxy = int(max(rxy, 0))
    z = np.arange(-rz, rz + 1)
    y = np.arange(-rxy, rxy + 1)
    x = np.arange(-rxy, rxy + 1)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dz = (zz / max(rz, 1)) ** 2
    dy = (yy / max(rxy, 1)) ** 2
    dx = (xx / max(rxy, 1)) ** 2
    return (dz + dy + dx) <= 1.0


def branch_threshold(vals: Array, mode: ThrMode, k: float = 4.0, percentile: float = 99.0) -> float:
    """Compute a per-branch threshold from 1D values."""
    vals = np.asarray(vals)
    if vals.size == 0:
        return float("nan")

    if mode == "percentile":
        return float(np.percentile(vals, percentile))

    # mode == "median_k_iqr"
    q25, q75 = np.percentile(vals, [25, 75])
    iqr = float(q75 - q25)
    return float(np.median(vals) + k * iqr)


def local_maxima_within_mask(
    img: Array,
    binary_mask: Array,
    footprint: Array,
) -> Tuple[pd.DataFrame, Array]:
    """
    Return one peak per connected local-max component inside `binary_mask`.

    Output peaks_df columns: z, y, x, peak_intensity
    """
    if not np.any(binary_mask):
        peaks_df = pd.DataFrame(columns=["z", "y", "x", "peak_intensity"])
        return peaks_df, np.zeros_like(binary_mask, dtype=np.int32)

    masked = np.where(binary_mask, img, -np.inf)
    local_max = ndi.maximum_filter(masked, footprint=footprint) == masked
    local_max &= binary_mask

    peak_labels = measure.label(local_max, connectivity=3)
    rows = []
    for lab in range(1, int(peak_labels.max()) + 1):
        coords = np.argwhere(peak_labels == lab)
        if coords.size == 0:
            continue
        intens = img[tuple(coords.T)]
        best = coords[int(np.argmax(intens))]
        z, y, x = map(int, best)
        rows.append((z, y, x, float(img[z, y, x])))

    peaks_df = pd.DataFrame(rows, columns=["z", "y", "x", "peak_intensity"])
    return peaks_df, peak_labels


@dataclass
class QuantParams:
    # Normalization
    norm_p_low: float = 25.0
    norm_p_high: float = 75.0

    # Puncta threshold within each branch
    thr_mode: ThrMode = "median_k_iqr"   # or "percentile"
    thr_k: float = 4.0                  # used for median_k_iqr
    thr_percentile: float = 99.0        # used for percentile

    # Binary cleanup
    min_puncta_voxels: int = 3          # remove tiny CCs; set to 1 to disable

    # Peak counting / separation
    peak_rxy: int = 2                   # radius in XY (pixels)
    peak_rz: int = 1                    # radius in Z (pixels)


def quantify_puncta_per_branch(
    img: Array,
    branch_mask: Array,
    params: QuantParams = QuantParams(),
    return_intermediates: bool = True,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Quantify puncta per branch.

    Returns
    -------
    df : one row per branch_id
    meta : dict with normalization + params
    inter : dict with intermediate arrays for QC (optional)
            keys: norm_img, puncta_mask, peak_coords_df
    """
    norm_img, norm_meta = normalize_global_iqr(img, params.norm_p_low, params.norm_p_high)

    footprint = ellipsoid_footprint(params.peak_rz, params.peak_rxy)

    puncta_union = np.zeros_like(branch_mask, dtype=bool)
    peak_rows = []
    out_rows = []

    branch_ids = np.unique(branch_mask)
    branch_ids = branch_ids[branch_ids != 0]

    # Compute ONE threshold for the whole image (using all branch pixels).
    all_branch_vals = norm_img[branch_mask > -1]
    thr_global = branch_threshold(
        all_branch_vals,
        mode=params.thr_mode,
        k=params.thr_k,
        percentile=params.thr_percentile,
    )

    for bid in branch_ids:
        m = branch_mask == bid

        puncta = (norm_img > thr_global) & m
        if params.min_puncta_voxels > 1:
            puncta = morphology.remove_small_objects(puncta, min_size=int(params.min_puncta_voxels), connectivity=3)

        puncta_union |= puncta

        puncta_mean = float(np.mean(norm_img[puncta])) if np.any(puncta) else float("nan")
        puncta_vox = int(np.sum(puncta))
        branch_vox = int(np.sum(m))

        peaks_df, _peak_labels = local_maxima_within_mask(norm_img, puncta, footprint)
        n_peaks = int(len(peaks_df))

        if n_peaks:
            peaks_df = peaks_df.copy()
            peaks_df["branch_id"] = int(bid)
            peak_rows.append(peaks_df)

        out_rows.append(
            {
                "branch_id": int(bid),
                "branch_voxels": branch_vox,
                "puncta_voxels": puncta_vox,
                "puncta_fraction": puncta_vox / branch_vox if branch_vox else float("nan"),
                "puncta_mean_intensity": puncta_mean,
                "n_puncta": n_peaks,
                "thr": float(thr_global),
            }
        )

    df = pd.DataFrame(out_rows).sort_values("branch_id").reset_index(drop=True)

    meta = {
        "normalization": norm_meta,
        "params": asdict(params),
        "thr_global": float(thr_global),
    }

    inter: Dict = {}
    if return_intermediates:
        peaks_all = pd.concat(peak_rows, ignore_index=True) if peak_rows else pd.DataFrame(
            columns=["z", "y", "x", "peak_intensity", "branch_id"]
        )
        inter = {"norm_img": norm_img, "puncta_mask": puncta_union, "peaks_df": peaks_all, "thr_global": float(thr_global)}

    return df, meta, inter
