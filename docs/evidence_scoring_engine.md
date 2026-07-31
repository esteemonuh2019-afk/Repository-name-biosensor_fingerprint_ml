# BSIP v4.0.0 Evidence Scoring Engine

The Evidence Scoring Engine independently evaluates the strength, completeness, consistency, uncertainty, and publication readiness of evidence supporting existing BSIP scientific claims.

It does not generate new claims, read raw experimental data, inspect model outputs, rerun analyses, use external services, or interpret scores as scientific truth.

## Inputs

Required claim package:

- `outputs/scientific_claims/claims.json`
- `outputs/scientific_claims/claim_validation.json`
- `outputs/scientific_claims/claim_summary.json`

Optional claim package files:

- `claim_dependencies.csv`
- `claim_evidence_scores.csv`
- `claim_publication_matrix.csv`
- `claims.csv`

Required reasoning graph package:

- `outputs/reasoning_graph/reasoning_graph.json`
- `outputs/reasoning_graph/reasoning_graph_validation.json`
- `outputs/reasoning_graph/reasoning_graph_summary.json`

Supported source schemas:

- Claim schema: `BSIP-3.2.0`
- Reasoning graph schema: `BSIP-3.1.0`

## Outputs

Default output directory:

- `outputs/evidence_scoring/`

Generated files:

- `evidence_scores.json`
- `evidence_scores.csv`
- `claim_confidence_matrix.csv`
- `evidence_dimension_breakdown.csv`
- `uncertainty_report.json`
- `reviewer_confidence_summary.json`
- `evidence_traceability.json`
- `evidence_scoring_validation.json`
- `evidence_scoring_summary.json`
- `evidence_scoring.md`

## Scoring Dimensions

Weights are versioned under `BSIP-EVIDENCE-RULES-4.0.0` and sum to exactly `1.0`.

- `TRACEABILITY`: `0.12`
- `SOURCE_VALIDATION`: `0.10`
- `OBSERVATION_SUPPORT`: `0.15`
- `INTERPRETATION_SUPPORT`: `0.10`
- `HYPOTHESIS_SUPPORT`: `0.12`
- `COMPETING_HYPOTHESIS_CONTROL`: `0.08`
- `EVIDENCE_GAP_BURDEN`: `0.10`
- `LIMITATION_COMPLETENESS`: `0.07`
- `INTERNAL_CONSISTENCY`: `0.08`
- `GENERALIZATION_SUPPORT`: `0.05`
- `REPRODUCIBILITY_SUPPORT`: `0.03`

## Evidence Levels

- `0-24.99`: `INSUFFICIENT`
- `25-44.99`: `LIMITED`
- `45-64.99`: `MODERATE`
- `65-79.99`: `STRONG`
- `80-100`: `VERY_STRONG`

## Uncertainty Model

Uncertainty is assessed independently from the evidence score. It reflects competing hypotheses, evidence gaps, internal-only validation, data-quality limitations, unresolved confounding, inconsistency, missing replication, and traceability completeness.

Uncertainty is not the inverse of the evidence score. A claim can have substantial internal support while retaining high uncertainty due to missing external validation or unresolved confounding.

## Publication Readiness

Readiness is independently assigned while respecting Claim Engine `publication_use` as a ceiling:

- `INTERNAL_REVIEW_ONLY` and `NOT_ELIGIBLE` cannot exceed `NOT_READY`
- `LIMITATION_ONLY` cannot exceed `LIMITATION_ONLY`
- `DISCUSSION_ELIGIBLE` cannot exceed `DISCUSSION_READY`
- `RESULTS_ELIGIBLE` may reach `RESULTS_READY`
- `HIGH_CONFIDENCE_RESULTS_READY` requires unusually strong evidence, low uncertainty, complete traceability, and genuine external validation

Internal validation is not external validation.

## Withholding Rules

Evidence score records are withheld when a critical condition applies:

- source validation failed
- duplicate claim IDs are present
- required schema version is unsupported
- critical graph dependencies are missing
- no supporting hypothesis exists
- no complete observation-to-interpretation-to-hypothesis traceability path exists
- the upstream claim is already withheld or unsupported

Withheld records are assigned `normalized_score = 0` and `publication_readiness = NOT_READY`.

## CLI

```bash
python scripts/build_evidence_scoring.py
```

Overwrite existing outputs:

```bash
python scripts/build_evidence_scoring.py --overwrite
```

Optional arguments:

- `--claims-dir`
- `--graph-dir`
- `--output-dir`
- `--overwrite`
- `--strict`
- `--software-version`

## Limitations

Evidence scores are deterministic evidence-support indices. They are not probabilities, Bayesian posterior probabilities, p-values, causal certainty, proof of mechanism, evidence of novelty, or evidence of external validity.
