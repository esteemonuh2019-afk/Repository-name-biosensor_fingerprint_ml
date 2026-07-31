# BSIP v3.0.0 Workflow Engine

## Purpose

The Workflow Engine coordinates the released BSIP reasoning engines:

1. Observation Engine
2. Interpretation Engine
3. Hypothesis Engine

It does not perform scientific reasoning. It executes existing engines, validates stage outputs, stops on critical failures, supports resume, and writes a workflow manifest and report.

## Source Boundary

The workflow starts from the configured supervisor-results package used by the Observation Engine. By default this is:

```text
outputs/supervisor_results_2/
```

The workflow does not read raw experimental data and does not parse classification, regression, QC, fingerprint, feature-engineering, interpretation, or hypothesis internals beyond each stage's public output package validation files.

## Output Layout

With the default `--output-root outputs`, stage outputs are:

- `outputs/scientific_observations/`
- `outputs/scientific_interpretations/`
- `outputs/scientific_hypotheses/`
- `outputs/workflow/`

Workflow outputs:

- `outputs/workflow/workflow_manifest.json`
- `outputs/workflow/workflow_report.md`

## Stage Gates

After each stage, the workflow validates the expected output package files and validation JSON. If the stage validation fails critically, the workflow stops and does not execute later stages.

Expected output package validation files:

- Observation: `observation_validation.json`
- Interpretation: `interpretation_validation.json`
- Hypothesis: `hypothesis_validation.json`

## Manifest

`workflow_manifest.json` includes:

- workflow ID,
- timestamp,
- software version,
- completed stages,
- failed stages,
- stage durations,
- output directories,
- validation summaries,
- source dataset,
- generated files,
- overall status,
- detailed stage records.

## Report

`workflow_report.md` summarizes:

- completed stages,
- outputs,
- validation status,
- warnings,
- overall reasoning-output readiness.

The report is orchestration metadata only and does not contain manuscript text.

## Resume

When `--resume` is used, each stage output directory is validated before execution. If a stage already has a complete, validation-passed output package, the stage is recorded as `SKIPPED` and the workflow continues to the next stage.

If an existing stage output package is incomplete or invalid, the workflow attempts to execute that stage. Without `--overwrite`, a non-empty invalid output directory may fail safely because the underlying engine protects existing outputs.

## CLI

```powershell
python scripts\run_bsip_workflow.py `
  --project-root "." `
  --output-root "outputs"
```

Supported flags:

- `--project-root`
- `--output-root`
- `--overwrite`
- `--resume`

## Non-Goals

The Workflow Engine does not:

- modify released engine public APIs,
- rewrite stage engines,
- perform observation extraction,
- perform interpretation or hypothesis reasoning,
- retrain models,
- read raw experimental data,
- generate manuscript text.
