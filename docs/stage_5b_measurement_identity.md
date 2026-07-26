# Stage 5B Measurement Identity

## Previous Problem

Stage 5A duplicate checks grouped measurements by strain, chemical, concentration, replicate, well, and time. Because the real source files do not provide `Plate_ID` or `Well_ID`, that key collapsed separate source-position curve blocks into duplicate logical measurements.

The Stage 5A real-data audit reported 30,697 duplicate rows in 14,438 legacy logical groups. The task brief also referenced an earlier 1,296-row duplicate figure; the Stage 5A document notes that the 1,296 count was not reproducible in the current audit path.

## Measurement Unit Definition

`Measurement_Unit_ID` identifies one experimental unit measured repeatedly over time. One canonical measurement is:

```text
one luminescence value
for one Measurement_Unit_ID
at one Time_Minutes value
within one Experiment_ID and Source_File
```

Strain, chemical, concentration, and replicate labels must remain internally consistent within a measurement unit, but they are not the measurement key.

## Source-Specific Identity Rules

The builder assigns `Measurement_Unit_ID` using the best available source evidence:

1. `Plate_ID` plus `Well_ID`
2. `Well_ID`
3. explicit replicate/sample structure combined with source-position curve blocks
4. stable source measurement-column position
5. deterministic source-position curve start

The inspected real CSV and Excel files are long-format tables with no physical plate or well columns. They provide source rows, replicate labels, worksheet names for Excel, and a luminescence measurement column. For these files, unit IDs are deterministic source-position IDs such as `unit_r034753__col007_luminescence`.

CSV duplicate measurement columns with the same display name are expanded rather than collapsed. The source column position is encoded in `Measurement_Unit_ID` and flagged with `measurement_unit_id_uses_source_column`.

Excel active worksheet names are retained in `Worksheet`. The real Excel workbooks use `Sheet1`; synthetic tests verify non-default worksheet names are preserved.

## Keys

Canonical measurement key:

```text
Experiment_ID, Source_File, Measurement_Unit_ID, Time_Minutes
```

Time-series grouping key:

```text
Experiment_ID, Source_File, Measurement_Unit_ID
```

QC uses the grouping key for monotonic-time checks, duplicate-timepoint checks, duration-related grouping, and future feature-extraction handoff.

## Synthetic Identifier Policy

Synthetic unit IDs are deterministic and derived only from stable source position. They do not use luminescence values, random UUIDs, dates, or import time.

Rows with source-position-generated IDs receive `measurement_unit_id_synthetic`. Replicate labels inferred from clear adjacent source-row structure receive `replicate_id_inferred_from_source_position`.

## Real-Data Results

Final Stage 5B audit output:

```text
outputs/qc/stage_5b_measurement_identity_replicate_repair
```

Counts from the corrected real-data run:

| Metric | Count |
|---|---:|
| Canonical rows before correction | 1,834,346 |
| Canonical rows after correction | 1,834,346 |
| Rows retained | 1,834,346 |
| Measurement units detected | 6,041 |
| Synthetic measurement units | 6,041 |
| Unresolved Measurement_Unit_ID rows | 0 |
| Missing Replicate_ID rows | 1 |
| Exact duplicate rows | 0 |
| Legacy logical duplicate rows | 30,697 |
| Legacy logical duplicate groups | 14,438 |
| Corrected measurement-key duplicate rows | 52 |
| Corrected measurement-key duplicate groups | 26 |
| Conflicting duplicate rows | 52 |
| Ambiguous measurement identity rows | 1 |
| Non-monotonic series | 0 |
| Duplicate time-point groups | 26 |

The corrected key reclassifies 30,645 of the 30,697 current legacy duplicate rows as separate measurement units. The 52 rows that remain are true corrected measurement-key duplicates, all with conflicting `Luminescence_Raw` values.

Relative to the older 1,296-row duplicate count named in the task brief, the corrected duplicate count is 52. A record-level mapping of that older 1,296-row set is not available in the repository, so the precise row-by-row false-positive split for that obsolete count cannot be reconstructed. Count-wise, the corrected audit is 1,244 rows lower than that earlier figure.

## Unresolved Limitations

- The real source files still have no physical `Plate_ID` or `Well_ID`, so all detected units are synthetic source-position units.
- One malformed retained BL032 CSV row has missing concentration, time, luminescence, and replicate fields. It receives a synthetic unit ID but remains an ambiguous invalid retained record.
- The 52 remaining duplicate measurement-key rows are duplicate time points within the same source-position unit and have conflicting raw luminescence values. They are retained and reported, not averaged or deleted.
- The Stage 5A legacy ambiguous replicate counters remain high because the legacy key cannot distinguish source-position units when plate/well identifiers are absent. The corrected measurement-key ambiguity count is 1 row.

## Future Dataset Implications

Future importers should prefer explicit plate/well or sample identifiers when available. When sources only provide long-format row blocks, source-position synthetic IDs are acceptable for retention and QC, but they should remain flagged so downstream analysis can decide whether the provenance is strong enough for modeling.

No feature extraction or downstream analysis was started in Stage 5B.
