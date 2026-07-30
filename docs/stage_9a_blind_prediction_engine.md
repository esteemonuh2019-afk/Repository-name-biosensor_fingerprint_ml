# Stage 9A Blind Prediction Engine

## Purpose

Stage 9A provides a frozen blind-prediction workflow for whole-cell biosensor fingerprints. It accepts unseen biosensor data, applies the established canonical and feature pipelines, aligns the exact frozen feature profiles, and produces chemical identity, concentration, confidence, novelty, QC, and evidence reports.

The engine is not a benchmarking stage. It does not select models, select features, tune thresholds, or refit preprocessing from blind samples.

## Architecture

The package is `src/blind_prediction/`.

- `model_bundle.py` defines the frozen `FrozenModelBundle`, feature-profile loading, model-card writing, and bundle serialization.
- `prediction_engine.py` trains bundles, predicts blind samples, evaluates truth-reveal files, and runs simulated blind holdouts.
- `prediction_qc.py` enforces feature order, missing-feature gates, time-window compatibility, required-strain gates, finite-value checks, and severe novelty escalation.
- `novelty_detection.py` fits training-only distance and confidence thresholds and assesses blind samples.
- `confidence_scoring.py` calculates probability, margin, entropy, replicate, feature-completeness, novelty, and QC confidence components.
- `prediction_report.py` writes the required machine-readable and Markdown reports.

Command-line entry points:

```bash
python scripts/train_blind_prediction_models.py "C:\Users\USER\Desktop\biosensor_phase2_source_files" --output-dir "models/blind_prediction/v1"
python scripts/predict_blind_sample.py "<blind_sample_folder>" --model-dir "models/blind_prediction/v1" --output-dir "outputs/blind_prediction/<sample_name>"
python scripts/evaluate_blind_predictions.py "outputs/blind_prediction/<sample_name>" "<truth_file.csv>"
```

## Model Freezing

The model bundle contains fitted sklearn `Pipeline` objects for:

- chemical classification;
- global concentration regression;
- chemical-specific concentration regression where enough training rows and concentration levels exist.

The bundle also stores:

- exact classification and regression feature order;
- class labels;
- concentration units;
- selected feature profiles from Stage 8D;
- training distribution summaries;
- training-only novelty thresholds;
- feature-engine and canonical-schema versions;
- dependency versions;
- random seeds;
- model metrics and training metadata.

The serialized artifact is `model_bundle.joblib`. It is accompanied by `model_metadata.json`, `model_card.md`, feature manifests, distribution summaries, and `novelty_thresholds.json`.

## Leakage Prevention

Blind samples never influence:

- model selection;
- feature selection;
- preprocessing fitting;
- model fitting;
- probability/margin/entropy thresholds;
- distance thresholds;
- concentration intervals.

During prediction, the engine copies the blind dataframe, generates features, checks compatibility, aligns frozen feature order, and calls fitted pipelines. Extra columns are explicitly ignored with warnings. Missing selected features fail clearly.

## Feature Compatibility

Stage 9A uses the Stage 8D approved feature profiles:

- default classification profile;
- default regression profile.

The real-data Stage 8D profiles use 67 classification features and 67 regression features selected by RFE at 75%. Prediction requires those columns in the exact frozen order. Non-finite selected features are excluded row-wise; if no usable rows remain, prediction fails.

## QC Gates

Prediction QC supports:

- `PASS`
- `PASS WITH WARNINGS`
- `FAIL`

Gates include:

- canonical QC errors;
- missing required model features;
- non-finite selected features;
- insufficient measurement units;
- missing configured required strains;
- incompatible time window;
- failed feature rows;
- severe novelty / out-of-distribution status.

Failed QC does not silently produce an ordinary prediction. Predictions may still be written for audit, but `prediction_passed` is false and confidence is forced to `Unreliable`.

## Time-Window Compatibility

The bundle records the training time window and the maximum window required by selected features. If selected features include `window_12_24h_*`, blind samples must contain compatible 24 h coverage. The engine does not extrapolate missing time windows.

## Classification Strategy

Classification uses a frozen fitted classifier pipeline. The default model is Extra Trees because Stage 8D benchmark reruns used Extra Trees for selected feature subsets.

Outputs include:

- top predicted chemical;
- probability for every known class;
- top three candidates;
- prediction margin;
- entropy.

Probabilities are treated as model probabilities, not certainty, unless future calibration is explicitly added and stored in the bundle.

## Concentration Strategy

The default strategy is `chemical_specific_with_global_comparison`.

Prediction workflow:

```text
predicted chemical
-> chemical-specific regressor if valid
-> concentration estimate and interval
-> global regressor comparison
```

If the predicted chemical lacks a valid chemical-specific regressor, the concentration estimate is withheld rather than reported as if valid. A global comparison may be recorded as context, but it is not substituted as the primary chemical-specific prediction.

## Confidence Scoring

The composite confidence score is deterministic and transparent. Components:

- classifier probability;
- class-margin score;
- entropy score;
- replicate consistency;
- feature completeness;
- novelty component;
- QC component.

Categories:

- High
- Moderate
- Low
- Unreliable

QC failure or out-of-distribution status forces `Unreliable`.

## Novelty Detection

Novelty uses two complementary families.

Distance-based:

- distance to nearest training fingerprint;
- distance to predicted-class centroid;
- thresholds from training distributions only.

Confidence-based:

- maximum class probability;
- prediction margin;
- entropy.

Statuses:

- Within Training Distribution
- Borderline
- Out of Distribution
- Unable to Assess

## Evidence Reporting

The engine reports model-native tree feature importances when available. Direction is reported by comparing blind feature means against training feature means. Influential strains are estimated by aggregating predicted-chemical probability by strain where strain metadata are present.

SHAP is not implemented in Stage 9A.

## Blind Workflow

1. Train and freeze the bundle before blind data are interpreted.
2. Receive blind biosensor files.
3. Run `predict_blind_sample.py`.
4. Review QC, novelty, confidence, probabilities, concentration, and evidence.
5. Do not supply truth labels to the prediction command.

## Truth-Reveal Workflow

Truth labels are supplied only after prediction has been saved. `evaluate_blind_predictions.py` reads:

- saved blind prediction results;
- a separate truth file.

It reports chemical correctness, concentration error, interval coverage, confidence, novelty status, and failure analysis. The prediction command never reads the truth file.

## Simulated Blind Testing

Simulation mode holds out an entire group before training, trains on the remaining groups, predicts the held-out group as blind, then reveals labels only for evaluation. Supported groups include source file, experiment, replicate batch, or other scientifically meaningful group columns.

Neighbouring time-series records from the same measurement unit are not randomly split across train and blind sets.

## Limitations

- The current canonical schema expects chemical and concentration metadata; future wet-lab blind formatting should use non-informative placeholders and preserve strain/time/luminescence fields.
- Chemical-specific regression is only available for chemicals with enough training rows and concentration diversity.
- Probability calibration is not yet implemented.
- Novelty thresholds are empirical and depend on the current training distribution.
- Existing upstream QC limitations remain visible and are not corrected by Stage 9A.

## Future Wet-Lab Blind Test Protocol

1. Freeze the Stage 9A bundle and archive its model card.
2. Record bundle hash and model directory before blind data are generated.
3. Ingest blind source files without adding labels to the prediction workflow.
4. Save the full blind-prediction output directory.
5. Lock the prediction outputs.
6. Reveal truth in a separate truth file.
7. Run evaluation mode.
8. Report both prediction performance and QC/novelty context.
