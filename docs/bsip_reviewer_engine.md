# BSIP v4.1.0 Reviewer Engine

The Reviewer Engine evaluates existing BSIP reasoning artifacts for scientific review readiness. It does not read raw data, retrain models, recompute analyses, generate scientific interpretations, write manuscript prose, compare against literature, or predict journal outcomes.

## Inputs

Required claim artifacts:

- `outputs/scientific_claims/claims.json`
- `outputs/scientific_claims/claim_validation.json`
- `outputs/scientific_claims/claim_summary.json`
- `outputs/scientific_claims/claim_publication_matrix.csv`

Required evidence-scoring artifacts:

- `outputs/evidence_scoring/evidence_scores.json`
- `outputs/evidence_scoring/evidence_scoring_validation.json`
- `outputs/evidence_scoring/evidence_scoring_summary.json`
- `outputs/evidence_scoring/reviewer_confidence_summary.json`
- `outputs/evidence_scoring/uncertainty_report.json`
- `outputs/evidence_scoring/evidence_traceability.json`

Required reasoning-graph artifacts:

- `outputs/reasoning_graph/reasoning_graph.json`
- `outputs/reasoning_graph/reasoning_graph_validation.json`
- `outputs/reasoning_graph/reasoning_graph_summary.json`

Optional supervisor artifacts:

- `outputs/supervisor_results_2/selected_figures.csv`
- `outputs/supervisor_results_2/selected_tables.csv`
- `outputs/supervisor_results_2/report_validation.json`

## Outputs

Default output directory:

- `outputs/scientific_review/`

Generated files:

- `review_findings.json`
- `review_findings.csv`
- `reviewer_report.md`
- `reviewer_validation.json`
- `reviewer_summary.json`
- `reviewer_blockers.csv`
- `reviewer_claim_matrix.csv`
- `reviewer_revision_requirements.csv`
- `reviewer_figure_matrix.csv`
- `reviewer_publication_assessment.json`

## Finding Model

Each `ReviewFinding` includes deterministic IDs, reviewer type, category, severity, blocking status, publication risk, affected source IDs, rationale, evidence summary, revision requirement, limitations, rule IDs, software version, tags, and metadata.

Finding IDs are deterministic per reviewer type:

- `REV-SCIENTIFIC-0001`
- `REV-STATISTICAL-0001`
- `REV-EVIDENCE-0001`
- `REV-VALIDATION-0001`
- `REV-REPRODUCIBILITY-0001`
- `REV-FIGURE-0001`
- `REV-WRITING-0001`
- `REV-PUBLICATION-0001`

## Reviewer Modules

- Scientific reviewer: checks claim support boundaries, competing explanations, and limitation alignment.
- Statistical reviewer: checks internal versus external validation, task metric boundaries, regression uncertainty, and overstatement risk.
- Evidence reviewer: checks evidence level, uncertainty, evidence gaps, competing hypotheses, traceability, and publication-readiness consistency.
- Validation reviewer: checks source validation and external-validation boundaries.
- Reproducibility reviewer: checks workflow traceability and separates computational reproducibility from biological replicate evidence.
- Figure reviewer: checks selected figure/table metadata and explicit claim-level visual links.
- Writing reviewer: checks structured claim text for unsupported causal, mechanistic, novelty, proof, or external-validation wording.
- Publication reviewer: applies deterministic publication-readiness policy to reviewer findings.

## Publication Policy

- Any `CRITICAL` finding gives `INTERNAL_REVIEW_ONLY`.
- One or more `MAJOR` findings gives `NEEDS_MAJOR_REVISION`.
- Only `MODERATE` findings give `NEEDS_MODERATE_REVISION`.
- Only `MINOR` findings give `NEEDS_MINOR_REVISION`.
- No material findings and at least one `RESULTS_READY` claim gives `READY_FOR_DRAFT_MANUSCRIPT`.

Absence of external validation alone does not prevent manuscript drafting, but it prevents definitive generalization claims.

## CLI

```bash
python scripts/build_scientific_review.py
```

Overwrite existing outputs:

```bash
python scripts/build_scientific_review.py --overwrite
```

Optional arguments:

- `--project-root`
- `--claims-dir`
- `--evidence-scoring-dir`
- `--reasoning-graph-dir`
- `--supervisor-dir`
- `--output-dir`
- `--strict`
- `--software-version`

## Validation

The engine validates source readability, source schema versions, source validation status, duplicate IDs, claim and graph references, severity/blocking policy, revision requirements, reviewer boundary language, deterministic ordering, recommendation policy, and output readability.

Validation issues are written to `reviewer_validation.json`.
