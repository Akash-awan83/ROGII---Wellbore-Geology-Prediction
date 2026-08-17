# Rogii Wellbore Geology Prediction

A Kaggle competition project predicting **True Vertical Thickness (TVT)** for horizontal oil/gas wells from well-log and trajectory data, using exploratory analysis followed by a gradient-boosted residual model.

## Problem

Given a horizontal well's measured depth (MD), spatial trajectory (X, Y, Z), and gamma ray (GR) log, predict TVT for rows where it is unknown — using the known portion of the well (and a reference "typewell" log) as context. A simple geometric/slope-based estimate is provided as a **baseline**; the goal is to beat it with a learned model.

## Repository Contents

| Notebook | Purpose |
|---|---|
| `eda-rogii-wellbore-geology-prediction.ipynb` | Exploratory data analysis: well trajectory visualization, gamma ray signal behavior, typewell correlation, and feature relationships that motivated the modeling approach. |
| `rogee.ipynb` | Feature engineering, model training (XGBoost), cross-validation, and final submission generation. |

## Exploratory Analysis — Key Findings

- **TVT vs MD** shows a clear regime change: TVT rises sharply through the curve/build section, then flattens once the well goes horizontal — motivating features that capture *local slope* rather than a single global trend.
- **Z vs MD** exposes the well's turning point from vertical to horizontal, useful for splitting/contextualizing behavior along the wellbore.
- **Gamma Ray (GR)** is highly volatile (roughly 40–140 API, with occasional large spikes), and its relationship to TVT is informative but noisy — better used as a supporting signal than a direct predictor.
- **Typewell correlation**: comparing a well's GR signature (normalized) against its reference typewell showed strong correlation in TVT-aligned space, supporting the use of the typewell as an auxiliary feature source.
- **TVT vs Z** shows a strong negative correlation (r ≈ -0.96) in the vertical section, confirming Z as one of the most informative geometric features.

These observations directly informed the feature set used in the modeling notebook (rolling GR statistics, local slope estimates, and typewell-derived features).

## Modeling Approach

**Strategy:** predict the *residual* between the true TVT and the provided baseline estimate, rather than TVT directly — so the model only has to correct the baseline's errors instead of learning the full signal from scratch.

**Feature set (55 features)** built per well, including:
- Positional/geometric features (MD, X, Y, Z, distance from prediction start)
- Known-segment statistics (TVT range/mean/std, GR mean/std/min/max from the known portion of the well)
- Local trend features (slope of TVT vs MD/Z over the full known segment and the most recent 200 rows)
- Rolling GR statistics (window sizes 11, 51, 151) and GR differencing
- Typewell-derived features (typewell TVT/GR ranges, GR aligned at the baseline TVT)

**Model:** `XGBRegressor` (450 estimators, depth 5, learning rate 0.035, L1/L2 regularization), trained to predict the baseline residual.

**Validation:** 5-fold `GroupKFold`, grouped by `well_id`, so no well appears in both the training and validation split within a fold — preventing leakage across rows of the same well.

## Results

| Fold | Baseline RMSE | XGB RMSE | Improvement |
|---|---|---|---|
| 1 | 15.290 | 14.466 | 0.824 |
| 2 | 15.917 | 14.595 | 1.323 |
| 3 | 14.678 | 13.527 | 1.151 |
| 4 | 16.047 | 15.209 | 0.838 |
| 5 | 17.480 | 16.666 | 0.814 |

**Out-of-fold (OOF) performance:**
- Baseline RMSE: **15.910**
- XGBoost RMSE: **14.929**

The model consistently improves on the baseline across all folds, with the largest gains in wells where the baseline's linear-slope assumption breaks down.

**Top predictive features** (by mean XGBoost importance across folds): typewell TVT range, recent local slope (TVT vs MD), known-segment TVT range/std, and typewell TVT min — reflecting that both the well's own recent trend and the reference typewell strongly inform the prediction.

## How to Run

1. Place the competition data under `/kaggle/input/competitions/rogii-wellbore-geology-prediction/` (or update `find_data_root()` in `rogee.ipynb` to point to your local copy).
2. Run `eda-rogii-wellbore-geology-prediction.ipynb` to reproduce the exploratory analysis.
3. Run `rogee.ipynb` end-to-end to build features, train the 5-fold XGBoost ensemble, and generate `submission.csv`.

## Limitations & Future Work

- Curve/trend-matching approaches (e.g., DTW, cross-correlation on the GR curve) were explored conceptually but found to be a dead end for this task — the harder problem is the low-frequency TVT trend, not the high-frequency GR wiggle.
- Slope features are computed with simple robust linear fits; a smoothed/weighted local regression could reduce noise sensitivity.
- No hyperparameter search was run beyond manual tuning; a proper CV-based search could likely improve on the current result.
