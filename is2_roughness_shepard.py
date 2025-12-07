#!/usr/bin/env python
"""
ICESat-2 → topographic roughness following Shepard et al. (2001):

For each beam (all ATL03 files merged):

1. AOI + projection + conf>=2 + robust detrend + MAD filter (same as z0m script).
2. 1 m along-track kriged profile z(s).
3. Sliding 200 m segments every 50 m:

   - Detrend segment with best-fit line.
   - RMS height σ_L (std of detrended profile over 200 m).
   - Deviogram: RMS deviation v(Δx) for lags Δx in [1, 2, 4, 8, 16] m using full interleaving.
   - Fit log10 v vs log10 Δx over 1–20 m → Hurst exponent H and intercept log10(v0).
   - RMS slope at smallest step (Δx=1 m): S_rms = v(1 m) / 1 m.

Outputs:
- CSV with one row per 200 m segment:
  beam, s_center, lat, lon, sigma_L, H, v0, s_rms_1m, v_lag_1m, v_lag_2m, ...

- Optional diagnostic plots per beam:
  - Example deviogram (log-log) with fitted H.
  - Histograms of H, σ_L.

Requires: numpy, pandas, h5py, pyproj, matplotlib.
"""

from pathlib import Path
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

# ================== USER SETTINGS ==================

H5_FOLDER = Path(r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\FINAL-\ATL03 - DATA\ATL03_007-20251206_125637")

# AOI (UTM20N, meters)
E_MIN, E_MAX = 359000.0, 368500.0
N_MIN, N_MAX = 8480000.0, 8489000.0

# projection: WGS84 -> UTM 20N
SRC_CRS = "EPSG:4326"
TGT_CRS = "EPSG:32620"
TRANS = Transformer.from_crs(SRC_CRS, TGT_CRS, always_xy=True)

CONF_MIN = 2                 # land-ice confidence threshold

# MAD filter
MAD_WINDOW_M = 50.0
Q_LOW, Q_HIGH = 1.0, 2.0

# Kriging grid & variogram
GRID_STEP = 1.0
RANGE_M = 15.0
MAX_NEIGHBORS = 100
MAX_DIST = 15.0

# Shepard-style segments and lags
SEG_LEN = 200.0       # segment length (m)
SEG_STEP = 50.0       # step between segment centers (m)
LAGS = np.array([1.0, 2.0, 4.0, 8.0, 16.0])  # lags (m) for deviogram
FIT_LAG_MIN = 1.0     # min lag for H fit (m)
FIT_LAG_MAX = 20.0    # max lag for H fit (m)

OUT_CSV = H5_FOLDER / "is2_roughness_shepard.csv"

PLOT_DIAGNOSTICS = True
PLOT_DIR = H5_FOLDER / "plots_shepard"
PLOT_DIR.mkdir(exist_ok=True)

# ================== HELPERS (same core as z0m script) ==================


def get_landice_conf(hgrp, npts):
    if "signal_conf_ph" not in hgrp:
        return np.full(npts, 3, dtype=np.int16)
    sc = np.asarray(hgrp["signal_conf_ph"])
    if sc.ndim == 2 and 5 in sc.shape:
        if sc.shape[0] == 5:
            lic = sc[3, :]
        else:
            lic = sc[:, 3]
    else:
        lic = np.full(npts, 3, dtype=np.int16)
    out = np.full(npts, 3, dtype=np.int16)
    m = min(npts, lic.size)
    out[:m] = lic[:m]
    return out


def alongtrack_distance(E, N):
    d = np.hypot(np.diff(E), np.diff(N))
    return np.concatenate(([0.0], np.cumsum(d)))


def robust_line(x, y, seed=0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 5:
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a), float(b)

    rng = np.random.default_rng(seed)
    m = min(20000, x.size)
    i = rng.integers(0, x.size, size=m)
    j = rng.integers(0, x.size, size=m)
    mask = i != j
    i, j = i[mask], j[mask]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    a = np.nanmedian(slopes)
    b = np.nanmedian(y - a * x)
    return float(a), float(b)


def mad_gate(s, res, W, qL, qH):
    s = np.asarray(s, float)
    res = np.asarray(res, float)
    idx = np.argsort(s)
    s_sorted, r_sorted = s[idx], res[idx]
    keep_sorted = np.zeros_like(r_sorted, bool)

    nb = max(1, int(np.ceil((s_sorted.max() - s_sorted.min()) / W)))
    edges = np.linspace(s_sorted.min(), s_sorted.max(), nb + 1)
    bi = np.clip(np.searchsorted(edges, s_sorted, side="right") - 1, 0, nb - 1)

    for b in range(nb):
        m = (bi == b)
        if m.sum() < 15:
            continue
        r = r_sorted[m]
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad == 0:
            mad = np.median(np.abs(r - med)) + 1e-9
        lo = med - (qL / 0.6745) * mad
        hi = med + (qH / 0.6745) * mad
        keep_sorted[m] = (r >= lo) & (r <= hi)

    keep = np.zeros_like(keep_sorted)
    keep[idx] = keep_sorted
    return keep


# ========== Simple 1D OK (same as in z0m script) ==========


def gaussian_cov(h, rng=RANGE_M, sill=1.0):
    h = np.abs(h)
    return sill * np.exp(-(h ** 2) / (rng ** 2))


def ok1d_simple(s_data, z_data, s_grid,
                R=RANGE_M, max_neigh=MAX_NEIGHBORS, max_dist=MAX_DIST):
    s_data = np.asarray(s_data, float)
    z_data = np.asarray(z_data, float)
    s_grid = np.asarray(s_grid, float)

    z_grid = np.full_like(s_grid, np.nan, dtype=float)
    if len(s_data) < 3:
        return z_grid

    for ig, sg in enumerate(s_grid):
        dist = np.abs(s_data - sg)
        mask = dist <= max_dist
        idx = np.where(mask)[0]
        if len(idx) < 3:
            continue
        idx = idx[np.argsort(dist[idx])][:max_neigh]
        sd = s_data[idx]
        zd = z_data[idx]
        nd = len(sd)

        C = gaussian_cov(sd.reshape(-1, 1) - sd.reshape(1, -1), rng=R)
        sill = np.var(zd) if nd > 1 else 1.0
        C.flat[::nd + 1] += 0.03 * sill

        A = np.block([[C, np.ones((nd, 1))],
                      [np.ones((1, nd)), np.zeros((1, 1))]])
        c = np.r_[gaussian_cov(sd - sg, rng=R), 1.0]
        try:
            wlam = np.linalg.solve(A, c)
        except np.linalg.LinAlgError:
            continue
        w = wlam[:nd]
        z_grid[ig] = float(w @ zd)

    return z_grid


# ================== Shepard-style deviogram ==================


def rms_height(z):
    """RMS height (std) of profile (assumes detrended)."""
    z = np.asarray(z, float)
    return float(np.nanstd(z))


def rms_deviation(z, lag_steps):
    """
    RMS deviation v(Δx) for integer lag measured in grid steps (Δx = lag_steps * GRID_STEP).
    Uses full interleaving as recommended by Shepard et al. (2001). 
    """
    z = np.asarray(z, float)
    n = z.size
    if lag_steps <= 0 or lag_steps >= n:
        return np.nan
    diffs = z[:-lag_steps] - z[lag_steps:]
    return float(np.nanstd(diffs))


def compute_deviogram(z_seg, dx=GRID_STEP, lags=LAGS):
    """
    Detrend segment, then compute:
      - σ_L: RMS height
      - v(Δx) for each Δx in lags
      - fit Hurst exponent H from log v vs log Δx over selected range
    """
    z_seg = np.asarray(z_seg, float)
    n = z_seg.size
    x = np.arange(n) * dx

    # linear detrend
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, z_seg, rcond=None)[0]
    z_detr = z_seg - (a * x + b)

    sigma_L = rms_height(z_detr)

    # RMS deviation for each lag
    v_vals = []
    for L in lags:
        lag_steps = int(round(L / dx))
        v_vals.append(rms_deviation(z_detr, lag_steps))
    v_vals = np.array(v_vals)

    # fit H the way Shepard suggests: over usable lags within [FIT_LAG_MIN, FIT_LAG_MAX]
    lag_mask = (lags >= FIT_LAG_MIN) & (lags <= FIT_LAG_MAX) & np.isfinite(v_vals) & (v_vals > 0)
    H = np.nan
    v0 = np.nan
    if lag_mask.sum() >= 2:
        logL = np.log10(lags[lag_mask])
        logv = np.log10(v_vals[lag_mask])
        # linear regression: logv = logv0 + H * logL
        Afit = np.vstack([logL, np.ones_like(logL)]).T
        H, logv0 = np.linalg.lstsq(Afit, logv, rcond=None)[0]
        v0 = 10.0 ** logv0

    return sigma_L, v_vals, H, v0


# ================== MAIN BEAM PIPELINE ==================


def process_beam_all_files(h5_files, gtx):
    print(f"\n=== Beam {gtx} (all files merged) ===")

    E_list, N_list, Z_list, lat_list, lon_list = [], [], [], [], []
    total = 0
    after_conf = 0
    after_aoi = 0

    for file in h5_files:
        try:
            with h5py.File(file, "r") as f:
                if gtx not in f:
                    continue
                base = f[f"{gtx}/heights"]
                lat = np.asarray(base["lat_ph"])
                lon = np.asarray(base["lon_ph"])
                z = np.asarray(base["h_ph"])
                n = z.size

                conf = get_landice_conf(base, n)
                total += n
                after_conf += np.sum(conf >= CONF_MIN)

                E, N = TRANS.transform(lon, lat)
                E = np.asarray(E)
                N = np.asarray(N)

                ok_aoi = (E >= E_MIN) & (E <= E_MAX) & (N >= N_MIN) & (N <= N_MAX)
                ok = ok_aoi & (conf >= CONF_MIN) & np.isfinite(E) & np.isfinite(N) & np.isfinite(z)
                after_aoi += np.sum(ok)

                if np.any(ok):
                    E_list.append(E[ok])
                    N_list.append(N[ok])
                    Z_list.append(z[ok])
                    lat_list.append(lat[ok])
                    lon_list.append(lon[ok])

        except Exception as e:
            print(f"  [WARN] {file.name}: {e}")

    print(f"  total photons: {total:,}")
    print(f"  after conf>={CONF_MIN}: {after_conf:,}")
    print(f"  after AOI+conf: {after_aoi:,}")

    if not Z_list:
        print("  [skip] No photons in AOI after merging all files.")
        return []

    E = np.concatenate(E_list)
    N = np.concatenate(N_list)
    Z = np.concatenate(Z_list)
    lat_all = np.concatenate(lat_list)
    lon_all = np.concatenate(lon_list)

    if Z.size < 50:
        print("  [skip] <50 photons in AOI – too small for roughness.")
        return []

    # along-track sort
    order = np.argsort(E)
    Eo, No, Zo = E[order], N[order], Z[order]
    s = alongtrack_distance(Eo, No)
    a, b = robust_line(s, Zo)
    trend = a * s + b
    res = Zo - trend

    keep = mad_gate(s, res, MAD_WINDOW_M, Q_LOW, Q_HIGH)
    E_keep = Eo[keep]
    N_keep = No[keep]
    Z_keep = Zo[keep]
    s_keep = s[keep]

    print(f"  kept after MAD: {E_keep.size:,} photons")

    if E_keep.size < 50:
        print("  [skip] too few after MAD")
        return []

    lat_all = lat_all[order]
    lon_all = lon_all[order]
    lat_keep = lat_all[keep]
    lon_keep = lon_all[keep]

    # 1 m grid + kriging
    s_min, s_max = s_keep.min(), s_keep.max()
    s_grid = np.arange(s_min, s_max + GRID_STEP, GRID_STEP)
    z_grid = ok1d_simple(s_keep, Z_keep, s_grid)

    valid_grid = np.isfinite(z_grid)
    if valid_grid.sum() < 100:
        print("  [skip] too few kriged points")
        return []

    s_grid = s_grid[valid_grid]
    z_grid = z_grid[valid_grid]
    lat_grid = np.interp(s_grid, s_keep, lat_keep)
    lon_grid = np.interp(s_grid, s_keep, lon_keep)

    # sliding segments
    results = []
    half_L = SEG_LEN / 2.0
    s_center_min = s_grid[0] + half_L
    s_center_max = s_grid[-1] - half_L
    if s_center_max <= s_center_min:
        print("  [skip] track too short for 200 m segments")
        return []

    s_centers = np.arange(s_center_min, s_center_max + SEG_STEP, SEG_STEP)

    for sc in s_centers:
        seg_mask = (s_grid >= sc - half_L) & (s_grid <= sc + half_L)
        if seg_mask.sum() < 0.75 * (SEG_LEN / GRID_STEP):
            continue

        s_seg = s_grid[seg_mask]
        z_seg = z_grid[seg_mask]

        sigma_L, v_vals, H, v0 = compute_deviogram(z_seg, dx=GRID_STEP, lags=LAGS)
        if not np.isfinite(sigma_L):
            continue

        lat_c = np.interp(sc, s_grid, lat_grid)
        lon_c = np.interp(sc, s_grid, lon_grid)

        # RMS slope at 1 m (if available)
        v1 = v_vals[0] if len(v_vals) > 0 else np.nan
        s_rms_1m = v1 / 1.0 if np.isfinite(v1) else np.nan

        row = {
            "beam": gtx,
            "s_center": sc,
            "lat": lat_c,
            "lon": lon_c,
            "sigma_L": sigma_L,
            "H": H,
            "v0": v0,
            "s_rms_1m": s_rms_1m,
        }
        # add v_lag_x columns
        for L, vL in zip(LAGS, v_vals):
            row[f"v_lag_{int(L)}m"] = vL

        results.append(row)

    print(f"  segments computed: {len(results)}")

    # diagnostics: deviogram for the first segment
    if PLOT_DIAGNOSTICS and results:
        dfb = pd.DataFrame(results)
        # example deviogram from first segment with valid H
        first_idx = dfb["H"].first_valid_index()
        if first_idx is not None:
            rec = dfb.loc[first_idx]
            v_example = np.array([rec.get(f"v_lag_{int(L)}m", np.nan) for L in LAGS])
            mask = np.isfinite(v_example) & (v_example > 0)
            if mask.sum() >= 2 and np.isfinite(rec["H"]):
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.loglog(LAGS[mask], v_example[mask], "o-", label="data")
                # fitted line v = v0 * L^H
                H_fit = rec["H"]
                v0_fit = rec["v0"]
                if np.isfinite(H_fit) and np.isfinite(v0_fit):
                    L_fit = np.linspace(LAGS[mask].min(), LAGS[mask].max(), 100)
                    v_fit = v0_fit * L_fit ** H_fit
                    ax.loglog(L_fit, v_fit, "--", label=f"fit H={H_fit:.2f}")
                ax.set_xlabel("lag Δx (m)")
                ax.set_ylabel("RMS deviation v(Δx) (m)")
                ax.set_title(f"{gtx}: example deviogram")
                ax.legend()
                fig.tight_layout()
                fig.savefig(PLOT_DIR / f"{gtx}_example_deviogram.png", dpi=200)
                plt.close(fig)

        # histograms of sigma_L and H
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(dfb["sigma_L"].dropna(), bins=20)
        ax[0].set_xlabel("σ_L (m)")
        ax[0].set_ylabel("count")
        ax[0].set_title(f"{gtx}: RMS height over {SEG_LEN:.0f} m")

        ax[1].hist(dfb["H"].dropna(), bins=20)
        ax[1].set_xlabel("Hurst exponent H")
        ax[1].set_ylabel("count")
        ax[1].set_title(f"{gtx}: H distribution")

        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{gtx}_shepard_summary.png", dpi=200)
        plt.close(fig)

    return results


# ================== MAIN ==================


def main():
    h5_files = sorted(H5_FOLDER.glob("*.h5"))
    if not h5_files:
        print(f"No ATL03 .h5 files found in {H5_FOLDER}")
        return

    print(f"Found {len(h5_files)} files in {H5_FOLDER}")

    all_results = []
    for gtx in ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]:
        res = process_beam_all_files(h5_files, gtx)
        all_results.extend(res)

    if not all_results:
        print("No segments produced.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} segments to {OUT_CSV}")


if __name__ == "__main__":
    main()
