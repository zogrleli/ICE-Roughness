#!/usr/bin/env python
"""
ICESat-2 → aerodynamic roughness (z0m) following Van Tiggelen et al. (2021)
and the Raupach (1992) drag-partition model adapted to 1D profiles.

Pipeline (per beam, all ATL03 files merged):

1. Read ATL03 photons (lat, lon, h_ph) in AOI, land-ice conf >= CONF_MIN.
2. Project to UTM20N, compute along-track distance s.
3. Robust linear detrend vs s, compute residuals.
4. MAD gate on residuals (window in meters), keep "good" photons.
5. Build 1 m along-track grid in s, ordinary kriging (Gaussian) to get z_grid(s).
6. For each overlapping 200 m segment (step 50 m):
   a. High-pass filter profile (cutoff ~35 m) to remove large-scale domes.
   b. Define obstacles as runs of positive heights; compute:
        - H = 2 * std(z_hp)
        - f = number of obstacles in segment
        - λ = f * H / L  (L = 200 m; width w ≈ footprint ≈ 15 m)
   c. Compute unresolved sub-grid σ_sub from photon residuals.
   d. Compute Hcorr from resolved + unresolved variance.
   e. Use Raupach (1992) bulk drag model (R92) to get z0m(H, λ) and z0m_corr(Hcorr, λ).

Outputs:
- CSV with one row per 200 m segment:
  beam, s_center, lat, lon, H, Hcorr, f, lambda, z0m, z0m_corr
- Simple diagnostic plots per beam (optional).

Requires:
  numpy, pandas, h5py, pyproj, matplotlib
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

# MAD filter (on residuals)
MAD_WINDOW_M = 50.0
Q_LOW, Q_HIGH = 1.0, 2.0     # more aggressive below the mean

# Kriging (simple OK for roughness)
GRID_STEP = 1.0
RANGE_M = 15.0               # Gaussian range
MAX_NEIGHBORS = 100
MAX_DIST = 15.0              # search radius (m)

# Roughness windows (Van Tiggelen et al. 2021)
SEG_LEN = 200.0         # m
SEG_STEP = 50.0         # m between centres
HPF_CUTOFF = 35.0       # m, cutoff wavelength for high-pass

SIGMA_I = 0.13          # ICESat-2 instrumental precision (m)
OUT_CSV = H5_FOLDER / "is2_roughness_z0m_vantiggelen.csv"

PLOT_DIAGNOSTICS = True
PLOT_DIR = H5_FOLDER / "plots_vantiggelen"
PLOT_DIR.mkdir(exist_ok=True)

# ================== HELPERS ==================


def get_landice_conf(hgrp, npts):
    """
    Read land_ice_conf from signal_conf_ph.
    Follows the same logic as in your kriging script.

    Returns:
        conf array (int16) length npts.
    """
    if "signal_conf_ph" not in hgrp:
        return np.full(npts, 3, dtype=np.int16)

    sc = np.asarray(hgrp["signal_conf_ph"])
    if sc.ndim == 2 and 5 in sc.shape:
        # dimension with length 5 is "surface type" index; take index 3 = land-ice
        if sc.shape[0] == 5:
            lic = sc[3, :]
        else:
            lic = sc[:, 3]
    else:
        # fallback if weird shapes
        lic = np.full(npts, 3, dtype=np.int16)

    out = np.full(npts, 3, dtype=np.int16)
    m = min(npts, lic.size)
    out[:m] = lic[:m]
    return out


def alongtrack_distance(E, N):
    """Compute along-track distance (cumulative) in meters."""
    d = np.hypot(np.diff(E), np.diff(N))
    return np.concatenate(([0.0], np.cumsum(d)))


def robust_line(x, y, seed=0):
    """
    Robust line fit (Theil–Sen style) for detrending.
    Returns slope a and intercept b (y ≈ a*x + b).
    """
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
    """
    MAD gate on residuals:
    - Bin along s with window W (m)
    - In each bin, compute median and MAD of residuals
    - Keep points within scaled MAD (asymmetric qL, qH)
    """
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


# ================== SIMPLE 1D KRIGING (Gaussian) ==================


def gaussian_cov(h, rng=RANGE_M, sill=1.0):
    h = np.abs(h)
    return sill * np.exp(-(h ** 2) / (rng ** 2))


def ok1d_simple(s_data, z_data, s_grid,
                R=RANGE_M, max_neigh=MAX_NEIGHBORS, max_dist=MAX_DIST):
    """
    Basic 1D ordinary kriging with Gaussian covariance.
    Only for getting a 1 m profile for roughness statistics (not precision altimetry).
    """
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

        # covariance matrix + small nugget
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


# ================== HIGHPASS + H, λ, Hcorr, z0m ==================


def highpass_profile_fft(z_seg, dx=1.0, cutoff=HPF_CUTOFF):
    """
    High-pass filter by:
    - detrending with least-squares line
    - FFT on mirrored profile, zero low frequencies, inverse FFT
    Returns high-pass profile z_hp.
    """
    z_seg = np.asarray(z_seg, float)
    x = np.arange(len(z_seg)) * dx

    # linear detrend
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, z_seg, rcond=None)[0]
    trend = m * x + b
    z_detr = z_seg - trend

    # mirror to reduce edge artefacts
    z_ext = np.concatenate([z_detr, z_detr[::-1]])
    n_ext = len(z_ext)

    Z = np.fft.rfft(z_ext)
    freqs = np.fft.rfftfreq(n_ext, d=dx)
    f_min = 1.0 / cutoff
    hp_mask = np.abs(freqs) >= f_min
    Z_hp = Z * hp_mask
    z_ext_hp = np.fft.irfft(Z_hp, n_ext)
    return z_ext_hp[:len(z_seg)]


def obstacle_stats(z_hp, L=SEG_LEN):
    """
    Compute H, f, λ = frontal area index in profile direction.

    - H = 2 * std(z_hp)
    - f = number of "obstacles" (runs of positive z_hp)
    - λ = f * H / L (assuming profile width ≈ obstacle width)
    """
    z_hp = np.asarray(z_hp, float)
    valid = np.isfinite(z_hp)
    if valid.sum() < 10:
        return np.nan, np.nan, np.nan

    sigma_ez = np.std(z_hp[valid])
    H = 2.0 * sigma_ez

    positive = (z_hp > 0) & valid
    if positive.sum() == 0:
        return np.nan, np.nan, np.nan

    # count contiguous positive runs
    # transitions: false -> true
    f = np.sum((~positive[:-1]) & positive[1:])
    if positive[0]:
        f += 1

    if H <= 0 or L <= 0 or f == 0:
        return np.nan, np.nan, np.nan

    lam = f * H / L
    return H, f, lam


def unresolved_sigma(z_hp, s_seg, s_ph, z_ph, z_grid_seg,
                     sigma_i=SIGMA_I):
    """
    Estimate unresolved roughness σ_sub from photon residuals around segment.

    - z_hp: high-pass grid profile in segment (for context)
    - s_seg, z_grid_seg: 1 m grid in segment
    - s_ph, z_ph: photons (after MAD) within segment
    """
    z_hp = np.asarray(z_hp, float)
    valid_hp = np.isfinite(z_hp)
    if valid_hp.sum() < 10:
        return np.nan

    # resolved high-pass variance (grid)
    sigma_g_res = np.std(z_hp[valid_hp])

    s_ph = np.asarray(s_ph, float)
    z_ph = np.asarray(z_ph, float)
    if len(s_ph) < 5:
        return np.nan

    valid_grid = np.isfinite(z_grid_seg)
    if valid_grid.sum() < 3:
        return np.nan

    s_grid_valid = s_seg[valid_grid]
    z_grid_valid = z_grid_seg[valid_grid]

    # interpolate grid back to photon positions
    z_interp_ph = np.interp(s_ph, s_grid_valid, z_grid_valid)
    residuals = z_ph - z_interp_ph
    sigma_ph_res = np.std(residuals)

    val = sigma_ph_res ** 2 - sigma_i ** 2
    if val <= 0:
        return np.nan

    sigma_sub = np.sqrt(val) / 2.0
    sigma_tot = np.sqrt(sigma_g_res ** 2 + sigma_sub ** 2)
    return 2.0 * sigma_tot  # Hcorr


# ===== Raupach (1992) & Garbrecht Cd(H) parametrisation (Van Tiggelen) =====


def cd_garbrecht(H):
    """
    Drag coefficient Cd(H) from Garbrecht-based parametrisation
    used in Van Tiggelen et al. (2021), their Eq. (3) approximations. 
    H in meters (obstacle height).
    """
    H = np.asarray(H, float)
    Cd = np.empty_like(H)
    small = H <= 2.5
    Cd[small] = 0.5 * (0.185 + 0.147 * H[small])
    big = ~small
    Cd[big] = 0.5 * (0.22 * np.log(H[big] / 0.2))
    return Cd


def r92_z0m(H, lam):
    """
    Raupach (1992) drag partition model, solved as in Van Tiggelen et al. (2021),
    Sect. 2.2–2.3 (moderate λ case). Returns z0m (m). 
    """
    H = float(H)
    lam = float(lam)
    if not np.isfinite(H) or not np.isfinite(lam) or H <= 0 or lam <= 0:
        return np.nan

    kappa = 0.4
    Cs10 = 1.2071e-3   # neutral drag coeff at 10 m over smooth ice in their setup
    c = 0.25
    cd1 = 7.5
    psi_hat_H = 0.193

    sqrt_cd1lam = np.sqrt(cd1 * lam)
    if sqrt_cd1lam == 0:
        return np.nan

    # displacement height fraction
    d = H * (1.0 - (1.0 - np.exp(-sqrt_cd1lam)) / sqrt_cd1lam)

    # bulk drag coefficient at height H
    denom = Cs10 ** (-0.5) - (1.0 / kappa) * np.log((10.0 - d) / (H - d))
    Cs_H = denom ** (-2)

    # obstacle drag coefficient Cd(H)
    Cd_H = float(cd_garbrecht(H))

    # dimensionless parameter a from R92 / Van Tiggelen
    a = (c * lam / 2.0) * (Cs_H + lam * Cd_H) ** (-0.5)

    # solve X * exp(-X) = a for X by Newton iterations
    def f(X):
        return X * np.exp(-X) - a

    def fprime(X):
        return np.exp(-X) * (1.0 - X)

    X = max(a, 1e-4)
    for _ in range(30):
        fX = f(X)
        dX = fX / fprime(X)
        X_new = X - dX
        if abs(dX) < 1e-6:
            X = X_new
            break
        X = X_new

    if X <= 0:
        return np.nan

    uH_over_u_star = 2.0 * X / (c * lam)
    z0m = (H - d) * np.exp(-kappa * uH_over_u_star + psi_hat_H)
    return z0m if z0m > 0 and np.isfinite(z0m) else np.nan


# ================== MAIN BEAM PIPELINE ==================


def process_beam_all_files(h5_files, gtx):
    print(f"\n=== Beam {gtx} (all files merged) ===")

    E_list, N_list, H_list, lat_list, lon_list = [], [], [], [], []
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
                h = np.asarray(base["h_ph"])
                n = h.size

                conf = get_landice_conf(base, n)

                total += n
                after_conf += np.sum(conf >= CONF_MIN)

                E, N = TRANS.transform(lon, lat)
                E = np.asarray(E)
                N = np.asarray(N)

                ok_aoi = (E >= E_MIN) & (E <= E_MAX) & (N >= N_MIN) & (N <= N_MAX)
                ok = ok_aoi & (conf >= CONF_MIN) & np.isfinite(E) & np.isfinite(N) & np.isfinite(h)

                after_aoi += np.sum(ok)

                if np.any(ok):
                    E_list.append(E[ok])
                    N_list.append(N[ok])
                    H_list.append(h[ok])
                    lat_list.append(lat[ok])
                    lon_list.append(lon[ok])

        except Exception as e:
            print(f"  [WARN] {file.name}: {e}")

    print(f"  total photons: {total:,}")
    print(f"  after conf>={CONF_MIN}: {after_conf:,}")
    print(f"  after AOI+conf: {after_aoi:,}")

    if not H_list:
        print("  [skip] No photons in AOI after merging all files.")
        return []

    E = np.concatenate(E_list)
    N = np.concatenate(N_list)
    Z = np.concatenate(H_list)
    lat_all = np.concatenate(lat_list)
    lon_all = np.concatenate(lon_list)

    if Z.size < 50:
        print("  [skip] <50 photons in AOI – too small for roughness.")
        return []

    # sort by along-track distance
    order = np.argsort(E)  # simple proxy for along-track ordering
    Eo, No, Zo = E[order], N[order], Z[order]
    s = alongtrack_distance(Eo, No)
    a, b = robust_line(s, Zo)
    trend = a * s + b
    res = Zo - trend

    # MAD gate on residuals
    keep = mad_gate(s, res, MAD_WINDOW_M, Q_LOW, Q_HIGH)
    E_keep = Eo[keep]
    N_keep = No[keep]
    Z_keep = Zo[keep]
    s_keep = s[keep]

    print(f"  kept after MAD: {E_keep.size:,} photons")

    if E_keep.size < 50:
        print("  [skip] too few after MAD")
        return []

    # interpolate lat/lon on s using all AOI photons (for later segment centres)
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

        # high-pass filter
        z_hp = highpass_profile_fft(z_seg, dx=GRID_STEP, cutoff=HPF_CUTOFF)
        H_val, f, lam = obstacle_stats(z_hp, L=SEG_LEN)

        if not np.isfinite(H_val) or not np.isfinite(lam) or lam <= 0:
            continue

        # photon subset for unresolved part
        ph_mask = (s_keep >= sc - half_L) & (s_keep <= sc + half_L)
        Hcorr = np.nan
        if ph_mask.sum() > 10:
            Hcorr = unresolved_sigma(
                z_hp,
                s_seg,
                s_keep[ph_mask],
                Z_keep[ph_mask],
                z_seg,
                sigma_i=SIGMA_I,
            )

        z0m = r92_z0m(H_val, lam)
        z0m_corr = np.nan
        if np.isfinite(Hcorr):
            z0m_corr = r92_z0m(Hcorr, lam)

        lat_c = np.interp(sc, s_grid, lat_grid)
        lon_c = np.interp(sc, s_grid, lon_grid)

        results.append(
            dict(
                beam=gtx,
                s_center=sc,
                lat=lat_c,
                lon=lon_c,
                H=H_val,
                Hcorr=Hcorr,
                f=f,
                lambda_val=lam,
                z0m=z0m,
                z0m_corr=z0m_corr,
            )
        )

    print(f"  segments computed: {len(results)}")

    if PLOT_DIAGNOSTICS and results:
        # simple diagnostic plots
        dfb = pd.DataFrame(results)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sc1 = ax[0].scatter(dfb["H"], dfb["lambda_val"], c=np.log10(dfb["z0m"]), s=20, cmap="viridis")
        ax[0].set_xlabel("H (m)")
        ax[0].set_ylabel("lambda")
        ax[0].set_title(f"{gtx}: H vs λ, coloured by log10(z0m)")
        cb = fig.colorbar(sc1, ax=ax[0])
        cb.set_label("log10(z0m)")

        ax[1].hist(np.log10(dfb["z0m"].dropna()), bins=20)
        ax[1].set_xlabel("log10(z0m) (m)")
        ax[1].set_ylabel("count")
        ax[1].set_title(f"{gtx}: z0m distribution")

        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{gtx}_z0m_summary.png", dpi=200)
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
