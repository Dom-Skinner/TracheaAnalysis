# TracheaAnalysis

Code accompanying **"A quantitative model of a phenotypically variable structural defect"** (Simpkins, Skinner et al., 2026).

The code here implements two complementary analyses:

1. **Statistical modeling** of larval endpoint data (terminal cell counts per metamere) using a liability-threshold framework fit by Bayesian inference.
2. **Live imaging shape analysis** of dorsal branch morphology and DSRF::sfGFP fluorescence dynamics.

---

## Repository layout

```
LarvaStats/      Julia scripts and data for the statistical analysis
ShapeAnalysis/   Julia scripts (+ MATLAB helpers) for shape and fluorescence analysis
```

---

## LarvaStats

### Dependencies

Julia packages are managed via `LarvaStats/Project.toml`. To install:

```julia
using Pkg; Pkg.activate("LarvaStats"); Pkg.instantiate()
```

Key packages: `Turing`, `MCMCChains`, `Distributions`, `Optim`, `Plots`, `StatsPlots`, `CSV`, `DataFrames`.

### Data (`LarvaStats/data/`)

| File | Contents |
|---|---|
| `Raw Cell Counts - All Genotypes - Raw Counts - WT.csv` | Per-larva, per-metamere terminal cell counts (0–4), wild type |
| `Raw Cell Counts - All Genotypes - Raw Counts - BNL MUTANT.csv` | Same, *bnl*−/+ |
| `Raw Cell Counts - All Genotypes - Raw Counts - F53S (GOF MEK).csv` | Same, MEK F53S gain-of-function |
| `Raw Cell Counts - All Genotypes - Raw Counts - RESCUE.csv` | Same, MEK F53S; *bnl*−/+ rescue |
| `Raw Cell Counts - All Genotypes - Binarized - *.csv` | Binarized versions of the above (used by `BayesianLarvaeAdjusted.jl`) |
| `CONTROL_pytag_puncta_branch_metrics_refactored_25IQR.csv` | Btl::pYtag FGFR activation metrics, wild type |
| `MUTANT_pytag_puncta_branch_metrics_refactored_25IQR.csv` | Same, *bnl*−/+ |

### Scripts

#### `Utils.jl`
Shared utilities included by all other `LarvaStats` scripts. 

#### `LiabilityThresholdCore.jl`
Defines the Turing.jl liability-threshold model (`liability_threshold`) and all associated utilities. 

Also contains functions for:
- Extracting posterior draws and computing MAP estimates.
- Running posterior predictive checks on metamere-level defect statistics.
- Computing larva-to-larva dispersion statistics.
- Leave-one-out cross-validation.

#### `LiabilityThresholdWtBnl.jl`
Fits the model to WT and *bnl*−/+ data only. Run this script to reproduce the two-genotype analysis. Produces:
- `plots/FitsForBnl.pdf` — MAP liability distributions for WT and *bnl*−/+
- `plots/effect_sizes_wt_bnl.pdf` — posterior boxplots of stochasticity σ, metamere-to-metamere variation, and the Bnl shift
- `plots/shift_effects.pdf` — max probability difference between metameres and total outcome variance as a function of genotype shift
- `plots/posterior_checks_metamere_defects_WT_bnl.pdf` — posterior predictive checks on max/min/avg defects per metamere (SI Figs. S8–S9)
- `plots/posterior_checks_larva_dispersion_wt_bnl.pdf` — posterior predictive checks on larva-to-larva variation
- `plots/stacked_bar_ppc_wt_bnl.pdf` — stacked bar posterior predictive samples by eye

#### `LiabilityThresholdAllGenotypes.jl`
Fits the model to all four genotypes, including leave-one-out cross-validation (fit on 3 genotypes, predict the held-out 4th). Produces:
- `plots/full_predicted.pdf` — empirical frequency vs. model-predicted probability (Fig. 6B)
- `plots/liability_dists_full_fit.pdf` — MAP liability distributions for all four genotypes
- `plots/liability_dists_prediction_rescue.pdf` — predicted distributions for rescue, from the model withheld from that genotype
- `plots/effect_sizes_full.pdf` — posterior boxplots of stochasticity, metamere variation, Bnl shift, and F53S shift
- `plots/posterior_checks_metamere_defects_total.pdf` — posterior predictive checks for all four genotypes (SI Figs. S8–S11)
- `plots/posterior_checks_larva_dispersion_total.pdf` — larva-to-larva dispersion posterior predictive checks for all four genotypes
- `plots/shift_effects_full.pdf` — metamere probability differences and variance as a function of genotype shift (Fig. 4E)
- `plots/stacked_bar_ppc.pdf` — stacked bar posterior predictive samples for all four genotypes

#### `BayesianLarvaeAdjusted.jl`
Tests whether metamere-specific effects are needed, separately for each genotype. Fits two hierarchical logistic models per genotype — a pooled model (no metamere effects) and a hierarchical model (per-metamere random effects) — both with larva-level random effects. Uses leave-one-larva-out cross-validation (ΔLOO) to compare them. Produces:
- `plots/larva_adjusted_pooled_v_hierarchical_<genotype>.pdf` — posterior and predictive check plots for each genotype (SI Figs. S1–S4)
- `plots/LOO_scores_pooled_v_hierarchical.pdf` — ΔLOO comparing models across all genotypes (SI Fig. S5)
- `plots/tau_posteriors.pdf` — posterior distributions of the metamere effect magnitude τ

#### `DemoPlots.jl`
Makes illustration plots for the paper (Fig. 4B–C). Produces:
- `plots/threshold_illustration.pdf` — Gaussian liability with a step function output
- `plots/simplex_traj.pdf` — map between the (μ, σ) parameter space and the probability simplex
- `plots/probabilities_traj.pdf` — Gaussian density and pie chart visualizations

#### `pYtagNormalityTest.jl`
Validates statistical assumptions for the Btl::pYtag FGFR activation analysis (SI Section III). Performs Shapiro–Wilk normality tests and F-tests of variance equality on log-transformed per-embryo means of puncta count and peak intensity. Produces `control_mutant_histograms.png`.

---

## ShapeAnalysis

### Dependencies

Julia packages (no `Project.toml`; install individually): `CSV`, `DataFrames`, `MultivariateStats`, `StatsBase`, `Plots`, `ReadVTK`, `HDF5`, `TiffImages`, `WriteVTK`, `Optimization`, `OptimizationOptimisers`, `SciMLSensitivity`, `Zygote`, `ForwardDiff`.

**MATLAB R2024a** is required for `DirectedSphericity.jl` and `ChordStats.jl`, which call MATLAB scripts in `src/` to compute alpha-shape surface areas and chord statistics.

### Data (`ShapeAnalysis/data/`)

| Directory/File | Contents |
|---|---|
| `raw/<date>_DSRFsfGFP[_bnl]/` | Per-experiment TIFF segmentation masks (one file per timepoint) |
| `raw/<date>_DSRFsfGFP[_bnl]_Data.csv` | Tracked cell coordinates and DSRF/Pnt intensities |
| `vtk/` | VTK-format 3D masks (generated by `MaskUnpack.jl`) |
| `processed/combined_data.csv` | Merged fluorescence + coordinate data with PCA alignment (from `UnpackData.jl`) |
| `processed/morphology.csv` | Shape features from VTK masks (from `MorphologySave.jl`) |
| `processed/data_with_morphology.csv` | Merged morphology + fluorescence dataset |
| `processed/data_with_PC.csv` | Adds PCA coordinates to the merged dataset (from `ShapePCA.jl`) |
| `processed/data_with_alignment.csv` | Final dataset with time-alignment offsets (from `AlignCoords.jl`) |
| `processed/sphericity/` | Per-experiment HDF5 files with sphericity and volume ratio |
| `processed/chords/` | Per-experiment HDF5 files with chord statistics |

Experiments labelled `*_bnl` are *bnl*−/+; the rest are wild type.

### Pipeline

Scripts must be run in the following order.

#### Step 1 — `UnpackData.jl`
Reads per-experiment CSV files of cell tracks (coordinates, DSRF::sfGFP and Pnt intensities). For each experiment, aligns the x–y coordinate frame so that PC1 points along the direction of branch elongation and increases with time. Computes:
- Inter-nuclear distance between the two cells in each metamere pair.
- Cell velocity (average displacement per frame).
- DSRF intensity normalized to the per-experiment 30th percentile.

Saves `data/processed/combined_data.csv`. Also produces `plots/intensity_normalizing.pdf` comparing raw and normalized intensity distributions across experiments.

#### Step 2 — `MaskUnpack.jl`
Converts 3D TIFF segmentation masks in `data/raw/` to VTK format in `data/vtk/`, applying a z-axis scaling factor of 9.39 to account for the axial voxel size. Must be run before any of the shape feature scripts.

#### Step 3 — `DirectedSphericity.jl`
Computes sphericity and convex volume ratio from the 3D VTK masks by calling the MATLAB script `src/SurfaceArea.m`, which uses MATLAB's `alphaShape` and `convhull` to compute surface area and volume. Results saved as HDF5 files in `data/processed/sphericity/`.

**Requires MATLAB R2024a** at `/Applications/MATLAB_R2024a.app/`.

#### Step 4 — `ChordStats.jl`
Computes chord statistics by calling the MATLAB script `src/ChordStats.m`. For each branch mask, random pairs of voxels are drawn and the fraction of the chord lying outside the alpha shape is measured — this quantifies how concave/branched the shape is. Results saved as HDF5 files in `data/processed/chords/`.

**Requires MATLAB R2024a** at `/Applications/MATLAB_R2024a.app/`.

#### Step 5 — `MorphologySave.jl`
Reads the VTK masks and the HDF5 outputs from steps 3–4. Computes the remaining shape features for each branch at each timepoint:
- **Elongation** — 90th–10th percentile range along the PC1 axis of the 3D point cloud.
- **Relative volume** — fraction of voxels in the distal tip region.
- **2D area, perimeter, circularity, anisotropy (eccentricity)** — computed from the maximum-intensity projection.

Merges with fluorescence data from step 1, checks that the join is complete, and saves `data/processed/data_with_morphology.csv`.

#### Step 6 — `ShapePCA.jl`
Applies PCA to the ten shape features (elongation, sphericity, chord fraction, volume ratio, area, anisotropy, perimeter, circularity, relative volume, inter-nuclear distance) using only wild-type branches to define the principal axes. Adds PC coordinates to the dataset and saves `data/processed/data_with_PC.csv`. Produces `plots/PCA_plots.pdf`.

#### Step 7 — `AlignCoords.jl`
Time-aligns the different imaging experiments by fitting a tanh sigmoid to the PC1 (morphology) trajectories and minimizing per-experiment time offsets (only timepoints ≤ 175 min are used for alignment). Also fits alignment offsets using the DSRF intensity trajectories for comparison.

Produces the final plots used in the paper:
- `plots/Unaligned_data.pdf` / `plots/Aligned_to_PC_data.pdf` — all shape features and DSRF before and after alignment
- `plots/DSRF_Intensity_in_time.pdf` / `plots/DSRF_Scaled_Intensity_in_time.pdf` — DSRF trajectories (Fig. 2G–H)
- `plots/DSRF_versus_Morphology.pdf` — overlay of DSRF and morphology PC1 trajectories
- `plots/DSRF_threshold_crossing_time_vs_T.pdf` — DSRF threshold crossing time vs. threshold level for WT and successful mutant branches
- `plots/PC_full_tmp2.pdf` — PCA scatter colored by aligned time (Fig. 3F)
- `plots/AlignCoords.pdf` — scatter of morphology vs. DSRF alignment time shifts

Saves the final analysis dataset to `data/processed/data_with_alignment.csv`.

### `src/Utils.jl`
Julia utilities shared across ShapeAnalysis scripts (image loading helpers, data reshaping).

### `src/SurfaceArea.m` and `src/ChordStats.m`
MATLAB scripts called by `DirectedSphericity.jl` and `ChordStats.jl`. They read segmentation data from a temporary HDF5 file and write results back to a second HDF5 file.
