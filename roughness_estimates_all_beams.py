import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
shep = pd.read_csv(r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\FINAL-\ATL03 - DATA\ATL03_007-20251206_125637\is2_roughness_shepard.csv")
z0   = pd.read_csv(r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\FINAL-\ATL03 - DATA\ATL03_007-20251206_125637\is2_roughness_z0m_vantiggelen.csv")


SAVE_DIR = r"C:\Users\kater\Desktop\ESPE\3RD_SEMESTER\PROJECT IN ESPE\FINAL-\PLOTS\Comparison"
os.makedirs(SAVE_DIR, exist_ok=True)

# Merge tables on matching columns
df = pd.merge(shep, z0, on=["beam", "s_center", "lat", "lon"], how="inner")

# List of all beams
beams = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

# Loop through each beam and plot
for beam_id in beams:

    dfb = df[df["beam"] == beam_id].sort_values("s_center")

    if dfb.empty:
        print(f"No data for {beam_id}, skipping.")
        continue

    x = dfb["s_center"].values
    r_rmsd  = dfb["sigma_L"].values
    r_quant = dfb["Hcorr"].values

    # Clip extreme outliers for nicer axis scaling
    p99_rmsd  = np.nanpercentile(r_rmsd, 99)
    p99_quant = np.nanpercentile(r_quant[~np.isnan(r_quant)], 99)
    r_rmsd  = np.clip(r_rmsd,  0, p99_rmsd)
    r_quant = np.clip(r_quant, 0, p99_quant)

       # --- Plot 1: Beautiful Along-Track Roughness Profiles ---
    plt.figure(figsize=(10,5))

    plt.plot(
        x, r_rmsd,
        color="#1f77b4", linewidth=2, alpha=0.85,
        label="Shepard σₗ (RMS roughness)"
    )

    plt.plot(
        x, r_quant,
        color="#d62728", linewidth=2, alpha=0.85,
        label="Tiggelen Hcorr (corrected roughness)"
    )

    plt.title(f"Along-Track Roughness Profiles – {beam_id.upper()}",
              fontsize=15, weight="bold")
    plt.xlabel("Along-track distance (m)", fontsize=12)
    plt.ylabel("Roughness (m)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.35)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()

    # Save figure
    fname1 = os.path.join(SAVE_DIR, f"{beam_id}_roughness_profiles.png")
    plt.savefig(fname1, dpi=200)
    plt.show()

    print(f"Saved: {fname1}")



              # --- Plot 2: Beautiful Scatter Plot σ_L vs Hcorr ---
    plt.figure(figsize=(7,6))

    mask = np.isfinite(r_rmsd) & np.isfinite(r_quant)
    x_scatter = r_rmsd[mask]
    y_scatter = r_quant[mask]

    plt.scatter(
        x_scatter, y_scatter,
        s=40, color="#6A5ACD", edgecolor="white",
        linewidth=0.4, alpha=0.75
    )

    # 1:1 line
    upper = max(np.nanmax(x_scatter), np.nanmax(y_scatter))
    plt.plot([0, upper], [0, upper], "k--", alpha=0.4, label="1:1 line")

    plt.xlabel("Shepard roughness σₗ (m)", fontsize=12)
    plt.ylabel("Van Tiggelen corrected roughness Hcorr (m)", fontsize=12)
    plt.title(f"Scatter Plot: σₗ vs Hcorr – {beam_id.upper()}",
              fontsize=14, weight="bold")

    plt.grid(True, linestyle=":", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()

    # Save figure
    fname2 = os.path.join(SAVE_DIR, f"{beam_id}_scatter_sigmaL_vs_Hcorr.png")
    plt.savefig(fname2, dpi=200)
    plt.show()

    print(f"Saved: {fname2}")


