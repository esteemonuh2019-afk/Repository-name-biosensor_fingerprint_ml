# Stage 7B Exploratory Fingerprint Analysis

## Purpose

Stage 7B examines whether validated whole-cell biosensor fingerprints show exploratory structure by chemical, concentration, strain, experiment duration, and source-data type. It is an unsupervised exploratory stage. It does not implement classification, regression, blind prediction, supervised feature importance, or target-based feature selection.

## Analysis Architecture

The package `src/exploratory_analysis/` contains:

- `pca_analysis.py`: deterministic PCA, scaling, loadings, explained variance, and top feature contributors.
- `clustering_analysis.py`: hierarchical clustering with mathematically guarded distance/linkage combinations.
- `fingerprint_heatmaps.py`: heatmap source tables and publication-oriented figures.
- `exploratory_qc.py`: input QC, replicate-to-consensus distances, concentration trajectories, and strain dispersion.
- `exploratory_dataset.py`: structured `ExploratoryAnalysisResult` and output writing.

Consensus fingerprints are the primary exploratory unit. Individual fingerprints are retained for replicate-to-consensus analysis and optional secondary PCA only.

## PCA Preprocessing

PCA uses validated, finite, normalised fingerprint features. Metadata columns are excluded explicitly. Supported scaling methods are:

- `zscore`
- `robust`
- `minmax`
- `none`

Rows with non-finite feature values are excluded and counted. Values are not imputed. Constant features are excluded from PCA and reported. Component signs are oriented deterministically by making the largest absolute loading positive.

The PCA outputs include:

- scores
- loadings
- explained variance
- cumulative explained variance
- top contributing features per component

PCA figures are visual summaries only. They do not establish chemical classifiability.

## Clustering Rules

Hierarchical clustering uses consensus fingerprints. Supported distances:

- Euclidean
- Cosine
- Correlation

Supported linkage methods:

- Ward, Euclidean only
- Average
- Complete

Cluster assignments and dendrogram tables are exploratory. The analysis does not assign biological meaning automatically.

## Heatmap Definitions

Stage 7B writes both heatmap tables and figures:

- Chemical x feature consensus fingerprint heatmap: median scaled feature values by chemical.
- Strain x chemical heatmap: median fingerprint magnitude by strain and chemical.
- Chemical similarity heatmap: pairwise distance among chemical-level median fingerprints.
- Concentration-response fingerprint heatmap: ordered strain/chemical/concentration rows by numeric concentration.
- Feature loading heatmap: PCA loadings by feature and component.

Large tables remain complete in CSV outputs. Figures may limit displayed labels to preserve readability.

## Replicate-Consistency Analysis

Individual fingerprints are compared with their strain/chemical/concentration consensus fingerprint. The output reports:

- distance to consensus
- group replicate count
- insufficient replicate flags
- unusually distant replicate flags

Outliers are not deleted automatically.

## Concentration-Trajectory Analysis

For each strain and chemical with at least two numeric concentrations, observations are ordered by numeric concentration, not lexical label. Adjacent fingerprint distances are reported. The trajectory label describes monotonicity of fingerprint-vector norm and is an exploratory descriptor, not a dose-response claim.

## Strain Contribution

The strain dispersion summary reports:

- consensus count
- chemical count
- distance to strain centroid
- mean feature standard deviation
- candidate informative strain flag

The flag is unsupervised and descriptive. It does not rank supervised predictive importance.

## Limitations

Stage 7B inherits upstream limitations from canonical QC, feature QC, and fingerprint QC. Known duplicate measurement units, excluded feature rows, and feature-validation warnings must remain visible.

This stage does not prove separability, classify chemicals, estimate concentration, or validate dose response. It identifies patterns that can motivate later statistical testing and supervised validation.

## Transition Criteria For Supervised ML

Before supervised ML begins, the project should document:

- which fingerprint resolution will be used: consensus or individual;
- which QC-excluded rows remain out of scope;
- how correlated features will be handled;
- whether concentration labels are sufficiently standardised;
- whether replicate design supports biological reproducibility claims.

Only after those choices are documented should classification, regression, blind prediction, or target-based feature selection begin.
