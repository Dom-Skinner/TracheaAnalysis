# TracheaAnalysis

Code accompanying **"A quantitative model of a phenotypically variable genetic defect"** (Simpkins, Skinner et al., 2026).

The code implements two complementary analyses:

1. **Statistical modeling** of larval endpoint data (terminal cell counts per metamere) using a liability-threshold framework fit by Bayesian inference.
2. **Live imaging shape analysis** of dorsal branch morphology and DSRF::sfGFP fluorescence dynamics.

Raw data is not included and can be made available upon reasonable request.

---

## Repository layout

```
LarvaStats/        Julia scripts and data for the statistical analysis
ShapeAnalysis/     Julia scripts (+ MATLAB helpers) for shape and fluorescence analysis
PunctaCounting/    Python notebook and module for pYtag puncta quantification
```

---

## LarvaStats

Julia packages are managed via `LarvaStats/Project.toml`. To install:
```julia
using Pkg; Pkg.activate("LarvaStats"); Pkg.instantiate()
```

Raw data would be in `LarvaStats/data/` containing per-larva, per-metamere terminal cell count CSVs (raw counts 0–4 and binarized versions) for four genotypes (WT, *bnl*−/+, MEK F53S, MEK F53S/*bnl*−/+), plus Btl::pYtag FGFR activation metrics for WT and *bnl*−/+.

### Scripts

#### `Utils.jl` and `LiabilityThresholdCore.jl`
Shared utilities and the Turing.jl `liability_threshold` model, included by all analysis scripts. `LiabilityThresholdCore.jl` also contains functions for MAP estimation, posterior predictive checks, and leave-one-out cross-validation.

#### `LiabilityThresholdWtBnl.jl`
Fits the model to WT and *bnl*−/+ data. Produces MAP liability distributions (Fig. 4D), variability vs. shift plots (SI Fig. S4), and posterior predictive checks whose panels are assembled into SI Fig. S1.

#### `LiabilityThresholdAllGenotypes.jl`
Fits the model to all four genotypes with leave-one-genotype cross-validation. Produces MAP fits for all genotypes (Fig. 5I), effect size posteriors (Fig. 6A), variability shift plots (SI Fig. S4), stacked bar PPCs (SI Fig. S5), held-out predictions (SI Fig. S6), and a mutual information decomposition across inputs.

#### `BayesianLarvaeAdjusted.jl`
Tests for metamere-specific effects using hierarchical logistic models, separately per genotype. Fits a pooled (no metamere effects) and a hierarchical (per-metamere random effects) model, both with larva-level random effects, and compares them via leave-one-larva-out cross-validation. Posterior check panels assemble into SI Fig. S1; `plots/tau_posteriors.pdf` is panel E of SI Fig. S1.

#### `BayesianLREffect.jl`
Extends `BayesianLarvaeAdjusted.jl` with a per-observation left-right correlation parameter σ_lr, testing whether shared within-metamere effects beyond independence are needed. Produces posterior checks, LOO comparison, and concordant-pair PPCs per genotype.

#### `DemoPlots.jl`
Illustration plots for Fig. 4B–C: threshold illustration, probability simplex, and Gaussian/pie-chart visualizations.

#### `pYtagNormalityTest.jl`
Shapiro–Wilk normality tests and F-tests of variance equality on log-transformed pYtag activation metrics, validating assumptions used in the liability model (SI Section II).

---

## ShapeAnalysis

Julia packages are managed via `ShapeAnalysis/Project.toml`. To install:
```julia
using Pkg; Pkg.activate("ShapeAnalysis"); Pkg.instantiate()
```

**MATLAB R2024a** (at `/Applications/MATLAB_R2024a.app/`) is required for `DirectedSphericity.jl` and `ChordStats.jl`.

Raw data would be in `ShapeAnalysis/data/raw/` containining per-experiment TIFF segmentation masks and cell-tracking CSVs with coordinates and DSRF intensities. Processed outputs (VTK meshes, shape features, PCA coordinates, time-alignment offsets) accumulate in `data/processed/` as the pipeline runs. Experiments labelled `*_bnl` are *bnl*−/+.

### Pipeline (run in order)

1. **`UnpackData.jl`** — reads cell-tracking CSVs, aligns coordinate frames to PC1, computes inter-nuclear distance, cell velocity, and 30th-percentile-normalized DSRF intensity. Saves `data/processed/combined_data.csv`.

2. **`MaskUnpack.jl`** — converts TIFF segmentation masks to VTK format with z-axis scaling (factor 9.39).

3. **`DirectedSphericity.jl`** — computes sphericity and convex volume ratio from 3D masks via MATLAB `alphaShape`. *(Requires MATLAB)*

4. **`ChordStats.jl`** — computes chord statistics (fraction of random voxel-pair chords outside the alpha shape) as a measure of concavity. *(Requires MATLAB)*

5. **`MorphologySave.jl`** — assembles shape features (elongation, relative volume, 2D area/perimeter/circularity/eccentricity), merges with fluorescence data, saves `data/processed/data_with_morphology.csv`.

6. **`ShapePCA.jl`** — PCA on ten shape features using WT branches to define axes. Saves `data/processed/data_with_PC.csv`.

7. **`AlignCoords.jl`** — time-aligns experiments by fitting a tanh sigmoid to PC1 morphology trajectories. Produces DSRF trajectory plots (Fig. 2G–H), PCA scatter colored by time (Fig. 3F), and saves `data/processed/data_with_alignment.csv`.

8. **`DSRFCrossingTime.jl`** — loads `data_with_alignment.csv` and computes mean first DSRF threshold crossing time vs. threshold level for WT and successful *bnl*−/+ branches, using both aligned and raw time axes. Produces `plots/DSRF_threshold_crossing_time_vs_T.pdf` (SI Fig. S7).

`src/Utils.jl` provides Julia utilities shared across scripts. `src/SurfaceArea.m` and `src/ChordStats.m` are the MATLAB scripts called by steps 3–4.

---

## PunctaCounting

Python notebook and module for 3D quantification of Btl::pYtag puncta in confocal image stacks. Raw image data is not included.

#### `pytag_puncta_quantification_refactored.py`
Core module providing image loading, mask filtering via a metamere lookup CSV, and puncta quantification (`QuantParams`, `quantify_puncta_per_branch`). Puncta are detected as local maxima above a per-image threshold — either `median + k·IQR` or a high percentile.

#### `pytag_puncta_quantification_refactored.ipynb`
Interactive notebook covering single-image QC and visualization, comparison of thresholding strategies, and batch processing over all image/mask pairs. Outputs a long-form per-branch results CSV.
