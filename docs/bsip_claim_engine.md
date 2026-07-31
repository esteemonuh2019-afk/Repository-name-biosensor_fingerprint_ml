# BSIP v3.2.0 Claim Engine

The Claim Engine transforms validated BSIP hypotheses into explicit, evidence-bounded scientific claims. It does not read raw experimental data, model reports, feature tables, figures, or external sources.

## Inputs

The engine reads only:

- `outputs/scientific_hypotheses/hypotheses.json`
- `outputs/scientific_hypotheses/hypothesis_validation.json`
- `outputs/scientific_hypotheses/hypothesis_summary.json`
- `outputs/scientific_hypotheses/hypothesis_dependencies.csv`
- `outputs/scientific_hypotheses/hypothesis_competition_map.csv`
- `outputs/reasoning_graph/reasoning_graph.json`
- `outputs/reasoning_graph/reasoning_graph_validation.json`
- `outputs/reasoning_graph/reasoning_graph_summary.json`

## Scientific Boundary

Claims are descriptive and conservative. The engine may state what current validated evidence supports, preserve competing explanations, identify limitations, and withhold unsupported claims. It does not introduce new observations, interpretations, hypotheses, mechanisms, literature comparisons, clinical claims, regulatory claims, deployment claims, recommendations, or manuscript sections.

## Output Files

The default output directory is `outputs/scientific_claims/`.

Generated files:

- `claims.json`
- `claims.csv`
- `claims.md`
- `claim_validation.json`
- `claim_summary.json`
- `claim_dependencies.csv`
- `claim_evidence_scores.csv`
- `claim_publication_matrix.csv`

## Claim IDs

Claim IDs are deterministic:

- `CLM-CHEMICAL_DISCRIMINATION-0001`
- `CLM-CONCENTRATION_INFORMATION-0001`
- `CLM-TEMPORAL_INFORMATION-0001`
- `CLM-FEATURE_REPRESENTATION-0001`
- `CLM-STRAIN_CONTRIBUTION-0001`
- `CLM-DATA_QUALITY-0001`
- `CLM-GENERALIZATION-0001`
- `CLM-SYSTEM_LEVEL_PERFORMANCE-0001`

## Validation

Validation checks include:

- required fields and ID/category consistency
- source hypothesis dependencies
- reasoning-graph node dependencies
- complete observation-to-interpretation-to-hypothesis traceability for active claims
- missing limitations
- causal, mechanistic, novelty, recommendation, deployment, clinical, regulatory, and external-validation overclaim language
- evidence-score boundaries and score-to-strength mapping
- publication-use policy
- deterministic ordering

## Evidence Score

The evidence score is a deterministic 0-100 support index based on source hypothesis status/confidence, support breadth, graph traceability, validation status, competing hypotheses, and evidence gaps. It is not a probability that a claim is true.

## CLI

```bash
python scripts/build_scientific_claims.py --project-root "." --hypotheses-dir "outputs/scientific_hypotheses" --reasoning-graph-dir "outputs/reasoning_graph" --output-dir "outputs/scientific_claims" --overwrite
```

The command exits with status `0` only when generation succeeds and no critical claim validation issue remains.
