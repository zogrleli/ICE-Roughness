#!/usr/bin/env python
# PHOTON_PLOT_COLORBAR_ALLFILES.py
#
# - reads all ATL03 .h5 in one folder
# - For each beam (gt1l..gt3r) merges photons from all files
# - Filters photons by AOI(UTM20N) + land-ice confidence >= CONF_MIN
# - Creates scatter plot:
#       x = Easting (m, UTM Zone 20N)
#       y = Height (m)
#       color = land-ice photon confidence (1–4)
# - Produces one PNG per beam with all photons from all files

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Transformer

# ==================================================

H5_FOLDER = Path(
    r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\FINAL-\ATL03 - DATA\ATL03_007-20251206_125637"
)

# AOI (UTM20N, m)
E_MIN, E_MAX = 359000.0, 368500.0
N_MIN, N_MAX = 8480000.0, 8489000.0

# min land-ice confidence - to keep photons
# (1–4 = increasing confidence, 0 = no info)
CONF_MIN = 1

# folder plots
OUTDIR = H5_FOLDER / "PHOTON_PLOTS_COLORBAR_ALLFILES"
OUTDIR.mkdir(exist_ok=True, parents=True)

# WGS84 -> UTM20N
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32620", always_xy=True)

BEAMS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

# =============================================================


def get_coords_and_height(hgrp):
    if "h_ph" not in hgrp:
        return None

    H = np.array(hgrp["h_ph"])

    if "easting" in hgrp and "northing" in hgrp:
        E = np.array(hgrp["easting"])
        N = np.array(hgrp["northing"])
    else:
        lon = np.array(hgrp["lon_ph"])
        lat = np.array(hgrp["lat_ph"])
        E, N = to_utm.transform(lon, lat)
        E, N = np.asarray(E), np.asarray(N)

    ok = np.isfinite(E) & np.isfinite(N) & np.isfinite(H)
    if not np.any(ok):
        return None

    return E[ok], N[ok], H[ok]


def get_landice_conf(hgrp, npts):
    if "signal_conf_ph" not in hgrp:
        # αν δεν υπάρχει, δώσε 1 (χαμηλή εμπιστοσύνη αλλά land-ice)
        return np.full(npts, 1, dtype=np.int16)

    sc = np.array(hgrp["signal_conf_ph"])  # shape (5,N) ή (N,5)
    if sc.ndim != 2 or 5 not in sc.shape:
        return np.full(npts, 1, dtype=np.int16)

    if sc.shape[0] == 5:
        li = sc[3, :]
    else:
        li = sc[:, 3]

    li = np.asarray(li).astype(np.int16)

    # safe copy
    if li.size != npts:
        out = np.full(npts, 1, dtype=np.int16)
        m = min(li.size, npts)
        out[:m] = li[:m]
        return out

    return li


# =============================================================
# MAIN
# =============================================================


def main():
    h5_files = sorted(H5_FOLDER.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {H5_FOLDER}")
        return

    print(f"Found {len(h5_files)} ATL03 files.")

    E_all = {b: [] for b in BEAMS}
    N_all = {b: [] for b in BEAMS}
    H_all = {b: [] for b in BEAMS}
    C_all = {b: [] for b in BEAMS}


    for h5_path in h5_files:
        print(f"\n=== Reading {h5_path.name} ===")
        with h5py.File(h5_path, "r") as f:
            file_beams = [k for k in f.keys() if k.startswith("gt")]
            if not file_beams:
                print("  [warn] no gt* groups in file")
                continue

            for gtx in BEAMS:
                if gtx not in file_beams:
                    continue

                hgrp = f.get(f"{gtx}/heights")
                if hgrp is None:
                    print(f"  [skip] {gtx}: no /heights group")
                    continue

                coords = get_coords_and_height(hgrp)
                if coords is None:
                    print(f"  [skip] {gtx}: missing/invalid coords or heights")
                    continue

                E, N, H = coords
                npts = H.size

                conf = get_landice_conf(hgrp, npts)

                # AOI + confidence
                ok = (
                    (E >= E_MIN)
                    & (E <= E_MAX)
                    & (N >= N_MIN)
                    & (N <= N_MAX)
                    & (conf >= CONF_MIN)
                )

                if ok.sum() == 0:
                    print(f"  {gtx}: 0 photons in AOI+conf>= {CONF_MIN}")
                    continue

                E_all[gtx].append(E[ok])
                N_all[gtx].append(N[ok])
                H_all[gtx].append(H[ok])
                C_all[gtx].append(conf[ok])

                print(f"  {gtx}: added {ok.sum()} photons")

    # --------  for each beam - merge & plot --------
    for gtx in BEAMS:
        if not E_all[gtx]:
            print(f"\n[beam {gtx}] no photons collected from any file, skipping.")
            continue

        E = np.concatenate(E_all[gtx])
        N = np.concatenate(N_all[gtx])
        H = np.concatenate(H_all[gtx])
        C = np.concatenate(C_all[gtx])

        print(f"\n[beam {gtx}] total photons after merge: {E.size}")

        if E.size < 10:
            print(f"[beam {gtx}] too few photons, skipping plot.")
            continue

        # ---------------- PLOT ----------------
        plt.figure(figsize=(12, 4))

        sc = plt.scatter(
            E,
            H,
            c=C,
            cmap="viridis",
            s=5,
            vmin=1,
            vmax=4,
            marker=".",
            linewidths=0,
        )

         
        

        # --------Title --------
        num_photons = E.size

        plt.title(
        f"ICESat-2 ATL03 - {gtx.upper()}\n"
        f"{num_photons:,} photons (all files)",
        fontsize=14,
        fontweight="bold",
        loc="center"
)



        plt.xlabel("Easting  (m, UTM Zone 20N)", fontsize=12)
        plt.ylabel("Surface Height  (m)", fontsize=12)

        plt.xlim(E_MIN, E_MAX)

        cbar = plt.colorbar(sc)
        cbar.set_label("Land-Ice Photon Confidence", fontsize=12)


        plt.tight_layout()

        out_png = OUTDIR / f"{gtx}_PHOTONS_COLORBAR_ALLFILES.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[OK] saved {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
