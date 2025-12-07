# ICE-Roughness
Python tool for loading, filtering, and visualizing ICESat-2 ATL03 data within a chosen area of interest. Supports multiple files, UTM conversion, photon or line-mode plotting, and PROMICE station markers. Outputs high-resolution maps for analysis of ground tracks and surface structure

# ICESat-2 Surface Roughness over the Greenland Ice Sheet

This repository contains a set of Python scripts to process ICESat-2 ATL03 photon data and derive **topographic** and **aerodynamic** roughness over a selected area of the Greenland Ice Sheet. The workflow:

1. Inspect ATL03 ground tracks in a user-defined AOI.
2. Visualise photon clouds and land-ice confidence.
3. Krige along-track elevation profiles.
4. Compute roughness metrics following **Shepard et al. (2001)** and **Van Tiggelen et al. (2021)**.
5. Compare the different roughness estimates for all beams.

---

## Repository contents

- **`PLOT_PHOTONS_OR_LINES.py`**  
  Plots ATL03 ground tracks inside a UTM20N AOI, either as individual photons or as simplified lines. Uses Cartopy and can overlay PROMICE stations. Saves a map PNG.

- **`icephotons_plots.py`**  
  Reads all ATL03 files in a folder, merges photons per beam, filters by AOI and land-ice confidence, and creates scatter plots of Easting vs height, coloured by confidence class (1–4). One PNG per beam.

- **`is2_roughness_shepard.py`**  
  Main script for **Shepard-style topographic roughness**.  
  For each beam (all ATL03 files merged):  
  AOI + projection → detrending → MAD filtering → 1 m kriged profile → sliding 200 m segments →  
  - RMS height σₗ  
  - deviogram v(Δx)  
  - Hurst exponent H  
  - RMS slope Sᵣₘₛ  
  Outputs a CSV with segment-wise roughness metrics.

- **`is2_roughness_vantiggelen.py`**  
  Main script for **aerodynamic roughness z₀m**, following Van Tiggelen et al. (2021) and a 1D adaptation of the Raupach drag-partition model. Uses the same preprocessing as the Shepard script, then identifies obstacles along the profile and computes z₀m and related length scales. Outputs a CSV per beam.

- **`vantiggelen_krigging_plots.py`**  
  Diagnostic script: shows MAD-filtered photons, the 1 m kriged profile z(s), and its experimental semivariogram. Useful for checking the variogram parameters and kriging behaviour.

- **`roughness_estimates_all_beams.py`**  
  Reads the CSV outputs from the Shepard and Van Tiggelen scripts, merges them by beam and segment centre, and produces comparison plots (e.g. σₗ vs z₀m, along-track profiles of both metrics).

---

## Data

The scripts are designed for **ICESat-2 ATL03 Global Geolocated Photon Data (collection 007)** obtained from:

**National Snow and Ice Data Center (NSIDC) DAAC**  
via the **NASA Earthdata Search** interface, using:
- a spatial filter defined by a bounding box over north-west Greenland (around the THU_L PROMICE station), drawn interactively on the map, and  
- a temporal filter of **2020-01-01 to 2020-03-01**.

Each selected ATL03 `*.h5` file is downloaded locally into a folder such as:

```text
ATL03_007-YYYYMMDD_...
