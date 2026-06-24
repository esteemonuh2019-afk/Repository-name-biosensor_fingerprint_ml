# Repeated-Run Robustness Analysis

## Purpose
Repeated random-seed evaluation quantifies model performance stability under randomized train/test partitioning and Random Forest initialization.

## Inputs
- Feature table: `outputs/tables/features.csv`
- Runs: 10
- Seeds: 1, 7, 11, 21, 42, 101, 123, 202, 555, 999
- Metrics: accuracy, precision, recall, f1

## Method
- Model: Random Forest classifier
- Target: chemical identity
- Split: stratified randomized train/test split when class counts allow
- Test fraction: 0.20

## Run-Level Metrics
| Seed | Accuracy | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8203 | 0.8194 | 0.8211 | 0.8198 |
| 7 | 0.8299 | 0.8302 | 0.8306 | 0.8297 |
| 11 | 0.8121 | 0.8102 | 0.8128 | 0.8106 |
| 21 | 0.8176 | 0.8181 | 0.8180 | 0.8131 |
| 42 | 0.8285 | 0.8282 | 0.8296 | 0.8262 |
| 101 | 0.8313 | 0.8320 | 0.8323 | 0.8307 |
| 123 | 0.8326 | 0.8310 | 0.8334 | 0.8312 |
| 202 | 0.8505 | 0.8494 | 0.8509 | 0.8492 |
| 555 | 0.8134 | 0.8139 | 0.8142 | 0.8122 |
| 999 | 0.8340 | 0.8339 | 0.8349 | 0.8329 |

## Summary Statistics
| Metric | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.8270 | 0.0110 | 0.8121 | 0.8505 |
| precision | 0.8266 | 0.0109 | 0.8102 | 0.8494 |
| recall | 0.8278 | 0.0109 | 0.8128 | 0.8509 |
| f1 | 0.8256 | 0.0113 | 0.8106 | 0.8492 |

## Figure
- Boxplot: `outputs/figures/repeated_run_boxplot.png`

## Interpretation
Low standard deviation and narrow min/max ranges indicate stable classification performance across random seeds. Larger ranges should be reviewed alongside LOEO, leave-one-strain-out, and confidence interval analyses before making claims about deployment robustness.
