# BSIP Reasoning Layer Contract

## Purpose

The BSIP reasoning layer separates factual data handling from higher-level scientific
reasoning. Each layer has a distinct responsibility and must not quietly perform work that
belongs to a later layer.

## Layer Separation

### Data Layer

The data layer stores and exposes completed analysis outputs.

It may contain:

- raw and canonical data
- feature tables
- model benchmark outputs
- QC summaries
- selected result inventories
- provenance indexes
- validation reports

It may not create observations, interpretations, hypotheses, or recommendations by itself.

### Observation Layer

The observation layer converts validated data-layer outputs into factual scientific
observations.

It may state:

- model identity reported by an authoritative source
- metric values
- sample counts and row counts
- QC status
- file presence
- selected feature-family names
- strain-ablation output values

It may not state:

- whether a metric is good or poor
- biological meaning
- causal explanation
- publication readiness
- recommended next action
- literature comparison
- hypothesis

Output objects from this layer must include evidence, provenance, confidence, status, and
limitations.

### Interpretation Layer

The interpretation layer may explain what factual observations mean in a scientific context.

It may discuss:

- performance implications
- consistency across observations
- biological plausibility
- limitations of interpretation
- relationships among observed results

It must cite observation IDs and must not invent observations.

### Hypothesis Layer

The hypothesis layer may propose testable scientific hypotheses derived from observations and
interpretations.

It may state:

- candidate biological mechanisms
- follow-up experimental hypotheses
- measurable predictions

It must label each hypothesis as speculative and trace it to upstream observations and
interpretations.

### Recommendation Layer

The recommendation layer may propose actions.

It may state:

- analysis follow-ups
- validation needs
- experiment design suggestions
- manuscript preparation steps

It must cite upstream observations, interpretations, and hypotheses. Recommendations must not
be emitted by the Observation Engine.

## Downstream Engines

The Observation Engine contract is designed to support later modules:

- Interpretation Engine
- Hypothesis Engine
- Supervisor Engine
- Reviewer Engine
- Manuscript Engine

Each downstream engine should consume structured objects, not scrape prose, whenever possible.

## Contract Rules

- Data facts must originate in validated outputs.
- Observations must be factual and evidence-backed.
- Interpretations must cite observations.
- Hypotheses must cite observations and interpretations.
- Recommendations must cite upstream reasoning objects.
- Missing evidence must produce incomplete or not-assessable status, not invented content.
- No layer may silently rewrite source metrics or provenance.

## Example Boundary

Observation:

```text
Classification metadata lists Extra Trees as rank 1 with selection_metric=f1_macro_mean.
```

Interpretation:

```text
The selected classifier is the strongest candidate within the evaluated benchmark set.
```

Hypothesis:

```text
The model may be using strain-specific temporal response differences to distinguish chemicals.
```

Recommendation:

```text
Run external validation on independently collected samples before manuscript claims.
```

Only the first sentence belongs to the Observation Engine.
