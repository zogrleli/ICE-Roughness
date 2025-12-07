# 02_plot_AOI_allfiles.py  (robust: supports AOI from multiple ATL03 files)
from pathlib import Path
import h5py, numpy as np, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pyproj import Transformer

# === USER SETTINGS ============================================================
H5_FOLDER = Path(r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\FINAL-\ATL03 - DATA\ATL03_007-20251206_125637")
MODE = "lines"     # "photons" or "lines"
LINES_STEP_M = 5.0   # spacing for the line representation (ignored if MODE="photons")

E_MIN, E_MAX = 359000, 368500
N_MIN, N_MAX = 8480000, 8489000

promice_wgs84 = [
    ("THU_L", -68.2677449, 76.3998511),
    ("THU_U", -68.1109372, 76.3900982),
]
# ============================================================================

to_utm = Transformer.from_crs("EPSG:4326","EPSG:32620", always_xy=True)

def fast_resample_line(E, N, step=5.0):
    if E.size < 2:
        return E, N
    d = np.hypot(np.diff(E), np.diff(N))
    s = np.concatenate(([0.0], np.cumsum(d)))
    tgt = np.arange(0.0, s[-1], step)
    idx = np.minimum(np.searchsorted(s, tgt), s.size-1)
    idx = np.unique(idx)
    return E[idx], N[idx]

# PROMICE stations → UTM
E_prom, N_prom, labels = [], [], []
for name, lon, lat in promice_wgs84:
    e, n = to_utm.transform(lon, lat)
    E_prom.append(float(e)); N_prom.append(float(n)); labels.append(name)

utm20 = ccrs.UTM(20)
fig = plt.figure(figsize=(7.2, 7.8))
ax = plt.axes(projection=utm20)
ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeee")
ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dfe9f2")
ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)

# === NEW: loop over ALL .h5 files in the folder ==============================
h5_files = sorted(H5_FOLDER.glob("*.h5"))
if not h5_files:
    raise SystemExit(f"No .h5 files found in {H5_FOLDER}")

print(f"Found {len(h5_files)} ATL03 files:")
for f in h5_files:
    print("  ", f.name)

for h5_path in h5_files:
    with h5py.File(h5_path, "r") as f:
        beams = [k for k in f.keys() if k.startswith("gt")]
        if not beams:
            continue

        for gtx in beams:
            base = f.get(f"{gtx}/heights")
            if base is None:
                continue

            # Prefer UTM if present; otherwise transform lon/lat -> UTM20N
            if ("easting" in base) and ("northing" in base):
                E = base["easting"][...]
                N = base["northing"][...]
            elif ("lon_ph" in base) and ("lat_ph" in base):
                lon = base["lon_ph"][...]
                lat = base["lat_ph"][...]
                ok = np.isfinite(lon) & np.isfinite(lat)
                lon, lat = lon[ok], lat[ok]
                if lon.size == 0:
                    continue
                E, N = to_utm.transform(lon, lat)
                E, N = np.asarray(E), np.asarray(N)
            else:
                continue  # nothing we can plot

            ok = np.isfinite(E) & np.isfinite(N)
            E, N = E[ok], N[ok]
            if E.size == 0:
                continue

            # Plot
            label = f"{h5_path.stem} {gtx}"
            if MODE.lower() == "photons":
                ax.scatter(E, N, s=0.4, alpha=0.5)
            else:
                Eb, Nb = fast_resample_line(E, N, step=LINES_STEP_M)
                ax.plot(Eb, Nb, lw=1.0, label=label)

# PROMICE stations
ax.scatter(E_prom, N_prom, s=30, edgecolor="black", facecolor="yellow",
           zorder=5, label="PROMICE")
for x, y, lbl in zip(E_prom, N_prom, labels):
    ax.text(x+250, y+250, lbl, fontsize=9, weight="bold")

# AOI frame + extent + ticks
ax.plot([E_MIN,E_MAX,E_MAX,E_MIN,E_MIN],
        [N_MIN,N_MIN,N_MAX,N_MAX,N_MIN],
        linestyle="--", linewidth=1.0, color="black")

ax.set_xlim(E_MIN-50, E_MAX+50)
ax.set_ylim(N_MIN-50, N_MAX+50)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Easting (m) — UTM20N WGS84")
ax.set_ylabel("Northing (m) — UTM20N WGS84")
ax.set_title(f"ICESat-2 ATL03 in AOI — {MODE}")
ax.set_xticks(np.arange(360000, 369000, 2000))
ax.set_yticks(np.arange(8480000, 8489000, 2000))
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
out_png = H5_FOLDER / f"ATL03_AOI_{MODE}.png"
plt.savefig(out_png, dpi=220, bbox_inches="tight")
plt.show()
print("Saved:", out_png)
