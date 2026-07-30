# Stage 7A Distance Matrix Optimisation

## Problem

Individual-level pairwise distance matrices scale as `N x N`. With 9,485 valid individual fingerprints, one full matrix contains 89,965,225 cells. Writing that matrix as CSV produced files of roughly 1.2 to 1.3 GB per metric, far above GitHub's 100 MB file limit.

The problem is not the distance metric itself. The problem is writing every pairwise individual comparison by default.

## Default Policy

The default distance mode is now `consensus`.

Consensus mode groups individual fingerprints by:

- `Strain`
- `Chemical`
- `Concentration`

The consensus fingerprint value is the median feature value within the group. The summary file also reports mean, standard deviation, coefficient of variation, finite count, replicate count, and QC status. The policy does not average across strains and does not average across concentrations.

## Distance Modes

`--distance-mode none`

Writes no distance matrices. Use this when only the fingerprint table is needed.

`--distance-mode consensus`

Default. Writes:

- `consensus_distance_matrix_euclidean.csv`
- `consensus_distance_matrix_cosine.csv`
- `consensus_distance_matrix_manhattan.csv`
- `consensus_distance_matrix_correlation.csv`

`--distance-mode individual`

Explicit opt-in. Writes historical individual-level files:

- `distance_matrix_euclidean.csv`
- `distance_matrix_cosine.csv`
- `distance_matrix_manhattan.csv`
- `distance_matrix_correlation.csv`

Individual mode is blocked above `--max-individual-distance-rows`, which defaults to `2000`, unless `--allow-large-distance-matrix` is supplied.

## Size Estimation

Before distance calculation, the CLI reports:

- row count
- matrix dimensions
- cell count
- estimated memory bytes
- estimated CSV bytes per matrix

The estimate is conservative and deterministic. It is intended to stop accidental gigabyte-scale outputs before allocation or file writing begins.

## CLI Examples

```bash
python scripts/build_fingerprint_dataset.py "C:\Users\USER\Desktop\biosensor_phase2_source_files"
python scripts/build_fingerprint_dataset.py "C:\Users\USER\Desktop\biosensor_phase2_source_files" --distance-mode none
python scripts/build_fingerprint_dataset.py --feature-file outputs/features/feature_dataset.csv --distance-mode individual --max-individual-distance-rows 2000
python scripts/build_fingerprint_dataset.py --feature-file outputs/features/feature_dataset.csv --distance-mode individual --allow-large-distance-matrix
```

## Downstream Implications

Consensus distances are appropriate for chemical-level overview, manuscript tables, initial similarity screening, and later PCA or clustering prototypes where replicate-level redundancy would dominate the matrix.

Individual distances remain available for specialised analyses, but they must be requested deliberately. Downstream PCA, clustering, machine learning, and blind prediction stages should use the fingerprint and consensus QC summaries to choose the appropriate resolution instead of regenerating full individual matrices by default.
