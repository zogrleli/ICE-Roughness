#!/usr/bin/env python
"""
Debug / diagnostic script for Van Tiggelen method:
- Same preprocessing as main z0m script (AOI + conf>=2 + detrend + MAD)
- 1 m kriged profile along track
- Plots:
    1) MAD-kept photons + kriged profile z(s)
    2) Experimental semivariogram γ(h) of the kriged profile

This script does NOT compute segments, H, λ, z0m.
It is only for fast visualisation/checks.
"""

from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pyproj import Transformer

# ================== USER SETTINGS ==================

H5_FOLDER = Path(
    r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\FINAL-\ATL03 - DATA\ATL03_007-20251206_125637"
)

# choose ONE beam to inspect
BEAM_ID = "gt3r"   # change to "gt1l", "gt1r", etc.

# AOI (UTM20N, meters) – same as main script
E_MIN, E_MAX = 359000.0, 368500.0
N_MIN, N_MAX = 8480000.0, 8489000.0

# projection: WGS84 -> UTM 20N
SRC_CRS = "EPSG:4326"
TGT_CRS = "EPSG:32620"
TRANS = Transformer.from_crs(SRC_CRS, TGT_CRS, always_xy=True)

CONF_MIN = 2

# MAD filter params
MAD_WINDOW_M = 50.0
Q_LOW, Q_HIGH = 1.0, 2.0

# Kriging params
GRID_STEP = 1.0
RANGE_M = 15.0
MAX_NEIGHBORS = 100
MAX_DIST = 15.0

# Where to save plots
PLOT_DIR = H5_FOLDER / "plots_kriging_debug"
PLOT_DIR.mkdir(exist_ok=True)

# ================== HELPERS (copied from main script) ==================


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


def experimental_semivariogram(z, dx=GRID_STEP, lags_m=None):
    """
    Simple 1D experimental semivariogram γ(h) on a regular profile.
    """
    z = np.asarray(z, float)
    n = z.size
    if lags_m is None:
        lags_m = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
    lags_m = np.asarray(lags_m, float)

    semivars = []
    lags_used = []
    for h in lags_m:
        k = int(round(h / dx))
        if k <= 0 or k >= n:
            continue
        diffs = z[k:] - z[:-k]
        gamma_h = 0.5 * np.nanmean(diffs ** 2)
        semivars.append(gamma_h)
        lags_used.append(h)

    return np.array(lags_used), np.array(semivars)


# ================== MAIN ==================


def main():
    h5_files = sorted(H5_FOLDER.glob("*.h5"))
    if not h5_files:
        print(f"No ATL03 .h5 files found in {H5_FOLDER}")
        return

    print(f"Found {len(h5_files)} files in {H5_FOLDER}")
    print(f"Processing only beam {BEAM_ID}")

    E_list, N_list, Z_list = [], [], []
    total = 0
    after_conf = 0
    after_aoi = 0

    for file in h5_files:
        try:
            with h5py.File(file, "r") as f:
                if BEAM_ID not in f:
                    continue
                base = f[f"{BEAM_ID}/heights"]
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

        except Exception as e:
            print(f"  [WARN] {file.name}: {e}")

    print(f"  total photons: {total:,}")
    print(f"  after conf>={CONF_MIN}: {after_conf:,}")
    print(f"  after AOI+conf: {after_aoi:,}")

    if not Z_list:
        print("No photons in AOI for this beam.")
        return

    E = np.concatenate(E_list)
    N = np.concatenate(N_list)
    Z = np.concatenate(Z_list)

    if Z.size < 50:
        print("Too few photons after AOI to do anything.")
        return

    # along-track + detrend + MAD
    order = np.argsort(E)
    Eo, No, Zo = E[order], N[order], Z[order]
    s = alongtrack_distance(Eo, No)
    a, b = robust_line(s, Zo)
    trend = a * s + b
    res = Zo - trend

    keep = mad_gate(s, res, MAD_WINDOW_M, Q_LOW, Q_HIGH)
    s_keep = s[keep]
    Z_keep = Zo[keep]

    print(f"  kept after MAD: {s_keep.size:,} photons")

    if s_keep.size < 50:
        print("Too few photons after MAD.")
        return

    # 1 m grid + kriging
    s_min, s_max = s_keep.min(), s_keep.max()
    s_grid = np.arange(s_min, s_max + GRID_STEP, GRID_STEP)
    z_grid = ok1d_simple(s_keep, Z_keep, s_grid)

    valid = np.isfinite(z_grid)
    s_grid = s_grid[valid]
    z_grid = z_grid[valid]

    if s_grid.size < 100:
        print("Too few kriged points.")
        return

    # ---- Plot 1: photons + kriged profile ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(s_keep, Z_keep, s=2, alpha=0.4, label="MAD-kept photons")
    ax.plot(s_grid, z_grid, "r-", lw=1.0, label="1 m kriged profile")
    ax.set_xlabel("Along-track distance s (m)")
    ax.set_ylabel("Surface height (m)")
    ax.set_title(f"{BEAM_ID}: photons + kriged profile")
    ax.legend()
    fig.tight_layout()
    out1 = PLOT_DIR / f"{BEAM_ID}_kriging_profile.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    print(f"Saved kriging profile plot to: {out1}")

    # ---- Plot 2: experimental semivariogram of kriged profile ----
    lags_m, semivars = experimental_semivariogram(z_grid, dx=GRID_STEP)
    if lags_m.size >= 2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(lags_m, semivars, "o-")
        ax2.set_xlabel("lag h (m)")
        ax2.set_ylabel("semivariance γ(h) (m²)")
        ax2.set_title(f"{BEAM_ID}: semivariogram (kriged profile)")
        ax2.grid(True, linestyle=":", alpha=0.5)
        fig2.tight_layout()
        out2 = PLOT_DIR / f"{BEAM_ID}_semivariogram.png"
        fig2.savefig(out2, dpi=200)
        plt.close(fig2)
        print(f"Saved semivariogram plot to: {out2}")
    else:
        print("Not enough lags for semivariogram.")


if __name__ == "__main__":
    main()
