# BSIP Interpretation Scientific Policy

Version: BSIP 2.1.0

## Purpose

This policy defines the scientific language boundary for the BSIP Scientific Interpretation Engine. The engine converts validated factual observations into conservative scientific meaning while preserving uncertainty, traceability, and strict separation from hypotheses and recommendations.

## Allowed Interpretation Language

Allowed claims are evidence-grounded and conservative. Preferred terms include:

- `suggests`
- `indicates`
- `is consistent with`
- `is associated with`
- `supports the presence of`
- `remains limited by`
- `cannot yet establish`

Allowed examples:

- The available classification observations suggest that biosensor fingerprints contain information associated with chemical-class discrimination.
- The regression observations indicate that concentration-related information is present while remaining limited by unexplained target variance.
- The blind-prediction observations do not establish external validation performance because true labels were absent.

## Forbidden Claims

The engine must not claim:

- causation,
- biological mechanism,
- publication readiness,
- clinical utility,
- regulatory suitability,
- field-deployment readiness,
- literature superiority,
- statistical significance without significance evidence.

Forbidden or restricted terms include:

- `proves`
- `confirms`
- `demonstrates conclusively`
- `causes`
- `results in`
- `biologically explains`
- `publication-ready`
- `clinically useful`
- `field-ready`

## Causal Boundaries

Interpretation may describe association, consistency, or evidence presence. It must not state that a chemical, strain, feature, or model result caused an outcome unless a causal analysis explicitly supports that claim.

Disallowed:

- The feature family caused improved model performance.
- The pathway explains the biosensor response.
- The result proves chemical identity.

Allowed:

- The feature-engineering observations indicate that the selected temporal feature family was associated with higher reported benchmark values.

## Recommendation Boundaries

The Interpretation Engine must not recommend experiments, actions, optimization steps, deployment, or publication decisions.

Rejected terms include:

- `should test`
- `should perform`
- `recommend`
- `future experiment`
- `ought to`

Recommendations belong to a later Recommendation or Supervisor layer.

## Hypothesis Boundaries

The Interpretation Engine must not generate explanatory hypotheses. It may preserve limitations and assumptions but must not propose mechanisms.

Rejected terms include:

- `we hypothesize`
- `may be caused by`
- `mechanism is`
- `pathway explains`

Hypotheses belong to the future Hypothesis Engine.

## Literature-Comparison Boundaries

The Interpretation Engine must not compare project results with published work or external benchmarks. Such comparisons require a separate curated literature evidence layer.

Rejected terms include:

- `compared with literature`
- `compared to literature`
- `previous studies`
- `published studies`
- `state of the art`
- `state-of-the-art`
- `literature benchmark`

## Blind-Validation Boundaries

Blind-prediction observations must not be converted into external validation claims unless true labels are available and validation metrics are explicitly observed.

Rejected overclaims include:

- external validation was achieved,
- external validation performance is available,
- blind validation accuracy,
- blind validation F1,
- true-label validation performance.

Allowed wording:

- The available blind-prediction observations do not establish external validation performance because true labels were absent.

## Uncertainty Language

Interpretations must preserve:

- assumptions,
- limitations,
- confidence,
- contradiction status,
- supporting Observation IDs,
- reasoning rule IDs,
- source Observation schema version.

Confidence must not be inferred from model metric magnitude alone. Confidence is assigned from observation coherence, completeness, contradiction status, QC context, and dependency validity.

## Conflict Handling

If supporting and contradicting observations are both present, the interpretation status must be `CONFLICTED`.

Contradicting observations must be recorded in:

- `contradicting_observation_ids`,
- an `InterpretationEvidenceLink` with `direction = CONTRADICTING`.

Conflicts must not be hidden by selective wording. Downstream narrative layers may summarize conflicts, but may not erase the structured contradiction record.
