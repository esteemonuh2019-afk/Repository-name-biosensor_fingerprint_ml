# BSIP v3.1.0 Reasoning Graph Engine

The Reasoning Graph Engine constructs a deterministic directed evidence graph from existing BSIP reasoning outputs. It does not run the Observation, Interpretation, Hypothesis, or Workflow engines, and it does not perform scientific reasoning.

## Inputs

The graph builder reads validated artifacts from:

- `outputs/scientific_observations/observations.json`
- `outputs/scientific_observations/observation_validation.json`
- `outputs/scientific_interpretations/interpretations.json`
- `outputs/scientific_interpretations/interpretation_validation.json`
- `outputs/scientific_interpretations/interpretation_dependencies.csv`
- `outputs/scientific_hypotheses/hypotheses.json`
- `outputs/scientific_hypotheses/hypothesis_validation.json`
- `outputs/scientific_hypotheses/hypothesis_dependencies.csv`
- `outputs/scientific_hypotheses/hypothesis_competition_map.csv`
- `outputs/workflow/workflow_manifest.json`

No upstream engines are invoked by the graph builder.

## Node Types

- `Dataset`
- `Observation`
- `Interpretation`
- `Hypothesis`
- `EvidenceGap`
- `ValidationSummary`
- `Workflow`

## Edge Types

- `supports`
- `derived_from`
- `limited_by`
- `competes_with`
- `validated_by`
- `generated_by`
- `belongs_to`

All edges are directional. Competition records are preserved as directed edges exactly as represented in the hypothesis artifacts.

## Validation

The validator checks:

- every edge source and target exists
- every interpretation has at least one observation parent
- every hypothesis has at least one interpretation parent
- every hypothesis is reachable from an observation through `supports` edges
- there are no orphan nodes
- there are no cycles in evidence edges
- nodes and edges are ordered deterministically

Evidence-cycle validation covers `supports`, `derived_from`, and `limited_by` edges. Workflow, validation, generation, and competition links are metadata edges and are not treated as scientific evidence chains.

## Query API

`ReasoningGraphQueries` supports deterministic graph traversal:

- `find_support_chain(node_id)`
- `find_downstream(node_id)`
- `find_upstream(node_id)`
- `find_competing_hypotheses(hypothesis_id)`
- `find_evidence_gaps(hypothesis_id)`

Module-level wrappers with the same names are also exported from `src.scientific_reasoning.reasoning_graph`.

## Outputs

The exporter writes:

- `outputs/reasoning_graph/reasoning_graph.json`
- `outputs/reasoning_graph/reasoning_graph.graphml`
- `outputs/reasoning_graph/reasoning_graph_summary.json`
- `outputs/reasoning_graph/reasoning_graph_validation.json`
- `outputs/reasoning_graph/reasoning_graph_statistics.csv`

## Command

```bash
python scripts/build_reasoning_graph.py --project-root . --output-dir outputs/reasoning_graph --overwrite
```

The command exits with status `0` only when graph validation passes.
