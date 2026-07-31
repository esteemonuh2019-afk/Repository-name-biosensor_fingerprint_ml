# BSIP v4.2.0 Manuscript Engine

The Manuscript Engine generates a conservative, evidence-traceable internal scientific manuscript draft from validated BSIP artifacts.

It does not read raw experimental data, recompute metrics, retrain models, retrieve literature, fabricate references, infer mechanisms, claim novelty, or mark unresolved reviewer blockers as resolved.

## Source Hierarchy

Required source packages:

- Observation Engine outputs in `outputs/scientific_observations/`
- Interpretation Engine outputs in `outputs/scientific_interpretations/`
- Hypothesis Engine outputs in `outputs/scientific_hypotheses/`
- Claim Engine outputs in `outputs/scientific_claims/`
- Evidence Scoring Engine outputs in `outputs/evidence_scoring/`
- Reviewer Engine outputs in `outputs/scientific_review/`
- Reasoning Graph outputs in `outputs/reasoning_graph/`
- Supervisor selected figure and table metadata in `outputs/supervisor_results_2/`

The engine stops without generating a manuscript when a required source file is missing, source validation fails critically, claim IDs are duplicated, manuscript-eligible claims are untraceable, or `reviewer_publication_assessment.json` has `manuscript_drafting_allowed=false`.

## Outputs

Default output directory:

- `outputs/scientific_manuscript/`

Generated files:

- `manuscript_draft.md`
- `manuscript_draft.docx`
- `manuscript_results.md`
- `manuscript_discussion.md`
- `manuscript_limitations.md`
- `manuscript_conclusion.md`
- `figure_captions.md`
- `table_captions.md`
- `manuscript_sentence_traceability.csv`
- `manuscript_claim_matrix.csv`
- `manuscript_figure_matrix.csv`
- `manuscript_table_matrix.csv`
- `manuscript_validation.json`
- `manuscript_summary.json`
- `manuscript_revision_flags.csv`
- `manuscript_manifest.json`

## Results Policy

Results sentences are based on validated observations and selected figure or table metadata. Quantitative Results sentences must link to observation IDs and source numbers present in the linked observation records.

Reviewer and evidence-scoring policy overrides upstream claim placement. A `RESULTS_ELIGIBLE` claim that is downgraded to `DISCUSSION_READY` is not used as a definitive Results conclusion.

## Discussion Policy

Discussion sentences may use validated claims, interpretations, hypotheses, evidence scores, uncertainty labels, and reviewer findings. They remain qualified and separate from Results.

The engine does not compare against literature because no verified literature package is supplied.

## Limitation Policy

Limitations are built from material reviewer findings, limitation-only claims, and reasoning-graph evidence-gap nodes. Blocking and major reviewer findings receive revision flags and remain unresolved until author review.

## Reviewer Enforcement

The Reviewer Engine publication assessment controls drafting status:

- `manuscript_drafting_allowed=false`: no manuscript is generated.
- `NEEDS_MAJOR_REVISION`: an internal draft may be generated, but it is labelled `REVISION_REQUIRED`.
- `definitive_generalization_allowed=false`: definitive external-generalization language is blocked.

## Sentence-Level Provenance

Every non-placeholder sentence receives a deterministic ID:

- `SENT-RESULTS-0001`
- `SENT-DISCUSSION-0001`
- `SENT-LIMITATIONS-0001`
- `SENT-CONCLUSION-0001`
- `SENT-FIGURE-0001`
- `SENT-TABLE-0001`

Traceability is exported in `manuscript_sentence_traceability.csv`.

## Figure And Table Captions

Captions are generated only from selected supervisor metadata. Captions include the selected figure or table ID, title, source file where available, and the analysis stage or row count when recorded.

The engine does not infer visual statistics or visual significance.

## DOCX Generation

The DOCX is generated as a deterministic Office Open XML package with:

- title page,
- heading hierarchy,
- revision-warning box,
- Results, Discussion, Limitations, Conclusion,
- figure and table captions,
- appendix of unresolved reviewer findings,
- appendix of sentence traceability summary.

The document is labelled as an internal scientific draft and is not labelled as submission ready.

## Validation Rules

Validation checks include source validation, drafting permission, unique IDs, sentence traceability, quantitative support, interpretive support, limitation support, language restrictions, reviewer blocker representation, publication boundaries, Results/Discussion separation, figure/table references, deterministic ordering, and output readability.

## CLI

```bash
python scripts/build_scientific_manuscript.py --overwrite
```

All source and output directories can be overridden:

```bash
python scripts/build_scientific_manuscript.py `
  --project-root "." `
  --observations-dir "outputs/scientific_observations" `
  --interpretations-dir "outputs/scientific_interpretations" `
  --hypotheses-dir "outputs/scientific_hypotheses" `
  --claims-dir "outputs/scientific_claims" `
  --evidence-dir "outputs/evidence_scoring" `
  --review-dir "outputs/scientific_review" `
  --graph-dir "outputs/reasoning_graph" `
  --supervisor-results "outputs/supervisor_results_2" `
  --output-dir "outputs/scientific_manuscript" `
  --overwrite
```

Optional:

- `--strict`
- `--software-version`
- `--title`
- `--author`

## Known Limitations

The engine creates an internal draft only. It does not create a full Abstract, Introduction, or Methods section because those require author-approved scope, verified literature, and validated experimental-design metadata.
