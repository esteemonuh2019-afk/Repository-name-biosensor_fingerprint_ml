# Confidence Interval Analysis

## Purpose
Bootstrap confidence intervals quantify statistical uncertainty in per-chemical LOEO precision, recall, and F1 estimates for SSDD/V&V evidence.

## Inputs
- Per-chemical LOEO table: `outputs/tables/per_chemical_loeo.csv`
- Per-chemical rows analyzed: 6
- CI metrics: precision, recall, f1

## Method
- Resampling unit: per-chemical LOEO metric row
- Bootstrap resamples: 1000
- Confidence level: 95%
- Random seed: 42

## Confidence Intervals
| Metric | Mean | CI Lower | CI Upper | Confidence | Bootstrap Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| precision | 0.6236 | 0.4481 | 0.8082 | 0.95 | 1000 |
| recall | 0.6208 | 0.4382 | 0.7900 | 0.95 | 1000 |
| f1 | 0.6203 | 0.4438 | 0.7931 | 0.95 | 1000 |

## Aggregate Model Metrics
Aggregate metrics from `outputs/tables/model_metrics.json` are shown for context; confidence intervals above are estimated from per-chemical LOEO metric variability.

| Metric | Value |
| --- | ---: |
| accuracy | 0.9868 |
| macro_precision | 0.9867 |
| macro_recall | 0.9868 |
| macro_f1 | 0.9867 |

## Interpretation
These intervals summarize observed metric variability across chemicals. Wider bounds indicate less stable performance across the contaminant panel and should be interpreted alongside LOEO, leave-one-strain-out, and per-chemical analyses.
