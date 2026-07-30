"""Fingerprint dataset container and output utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.fingerprint.fingerprint_qc import (
    FingerprintQCResult,
    render_fingerprint_qc_report,
)
from src.fingerprint.fingerprint_similarity import (
    estimate_distance_matrix_size,
    write_distance_matrix_csv,
)


DISTANCE_METRICS: tuple[str, ...] = ("euclidean", "cosine", "manhattan", "correlation")
DEFAULT_DISTANCE_MODE = "consensus"
DEFAULT_MAX_INDIVIDUAL_DISTANCE_ROWS = 2000
DistanceMode = Literal["none", "consensus", "individual"]


@dataclass(frozen=True)
class FingerprintDataset:
    """Validated fingerprint matrix plus scientific metadata."""

    dataframe: pd.DataFrame
    normalized_dataframe: pd.DataFrame
    consensus_dataframe: pd.DataFrame
    consensus_normalized_dataframe: pd.DataFrame
    consensus_summary: pd.DataFrame
    metadata: dict[str, Any]
    feature_names: list[str]
    feature_version: str
    fingerprint_version: str
    qc: FingerprintQCResult
    summary: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    normalization_method: str
    normalization_parameters: dict[str, Any]
    consensus_normalization_parameters: dict[str, Any]
    consensus_group_columns: list[str]
    excluded_dataframe: pd.DataFrame

    def write_outputs(
        self,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
        distance_mode: DistanceMode | str = DEFAULT_DISTANCE_MODE,
        max_individual_distance_rows: int = DEFAULT_MAX_INDIVIDUAL_DISTANCE_ROWS,
        allow_large_distance_matrix: bool = False,
        distance_chunk_size: int = 128,
    ) -> list[Path]:
        """Write fingerprint datasets, summaries, QC report, and distances."""

        distance_mode = _canonical_distance_mode(distance_mode)
        distance_outputs = self.distance_output_plan(
            distance_mode=distance_mode,
            max_individual_distance_rows=max_individual_distance_rows,
            allow_large_distance_matrix=allow_large_distance_matrix,
        )
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        output_paths = [
            target / "fingerprint_dataset.csv",
            target / "fingerprint_dataset_normalized.csv",
            target / "consensus_fingerprint_dataset.csv",
            target / "consensus_fingerprint_summary.csv",
            target / "fingerprint_summary.json",
            target / "fingerprint_qc_report.md",
        ]
        output_paths.extend(target / output["filename"] for output in distance_outputs)

        existing = [path for path in output_paths if path.exists()]
        if existing and not overwrite:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Fingerprint output files already exist. Use --overwrite to replace: "
                f"{formatted}"
            )

        created: list[Path] = []
        dataset_path = target / "fingerprint_dataset.csv"
        normalized_path = target / "fingerprint_dataset_normalized.csv"
        consensus_path = target / "consensus_fingerprint_dataset.csv"
        consensus_summary_path = target / "consensus_fingerprint_summary.csv"
        summary_path = target / "fingerprint_summary.json"
        report_path = target / "fingerprint_qc_report.md"

        self.dataframe.to_csv(dataset_path, index=False, encoding="utf-8")
        created.append(dataset_path)
        self.normalized_dataframe.to_csv(normalized_path, index=False, encoding="utf-8")
        created.append(normalized_path)
        self.consensus_dataframe.to_csv(consensus_path, index=False, encoding="utf-8")
        created.append(consensus_path)
        self.consensus_summary.to_csv(consensus_summary_path, index=False, encoding="utf-8")
        created.append(consensus_summary_path)
        summary_path.write_text(
            json.dumps(self._serialisable_summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        created.append(summary_path)
        report_path.write_text(
            render_fingerprint_qc_report(self.summary, self.qc),
            encoding="utf-8",
        )
        created.append(report_path)

        for output in distance_outputs:
            path = target / output["filename"]
            source = self._distance_source_dataframe(output["scope"])
            write_distance_matrix_csv(
                source,
                feature_names=self.feature_names,
                metric=output["metric"],
                output_path=path,
                label_column=output["label_column"],
                chunk_size=distance_chunk_size,
            )
            created.append(path)

        return created

    def distance_output_plan(
        self,
        *,
        distance_mode: DistanceMode | str = DEFAULT_DISTANCE_MODE,
        max_individual_distance_rows: int = DEFAULT_MAX_INDIVIDUAL_DISTANCE_ROWS,
        allow_large_distance_matrix: bool = False,
    ) -> list[dict[str, Any]]:
        """Return distance outputs after applying the individual-mode safety guard."""

        distance_mode = _canonical_distance_mode(distance_mode)
        if distance_mode == "none":
            return []
        if distance_mode == "consensus":
            return [
                {
                    "scope": "consensus",
                    "metric": metric,
                    "filename": f"consensus_distance_matrix_{metric}.csv",
                    "label_column": "Consensus_ID",
                    "estimate": estimate_distance_matrix_size(len(self.consensus_normalized_dataframe)),
                }
                for metric in DISTANCE_METRICS
            ]
        row_count = int(len(self.normalized_dataframe))
        if row_count > int(max_individual_distance_rows) and not allow_large_distance_matrix:
            estimate = estimate_distance_matrix_size(row_count)
            raise ValueError(
                "Individual distance matrix refused: "
                f"{row_count} rows exceeds --max-individual-distance-rows "
                f"{max_individual_distance_rows}. Estimated matrix is "
                f"{estimate['rows']} x {estimate['columns']} with "
                f"{estimate['cells']} cells and approximately "
                f"{estimate['estimated_csv_bytes']} CSV bytes. Use "
                "--allow-large-distance-matrix only when this output is intentional."
            )
        return [
            {
                "scope": "individual",
                "metric": metric,
                "filename": f"distance_matrix_{metric}.csv",
                "label_column": "Fingerprint_ID",
                "estimate": estimate_distance_matrix_size(row_count),
            }
            for metric in DISTANCE_METRICS
        ]

    def distance_estimates(self) -> dict[str, dict[str, int]]:
        """Return deterministic size estimates for individual and consensus matrices."""

        return {
            "individual": estimate_distance_matrix_size(len(self.normalized_dataframe)),
            "consensus": estimate_distance_matrix_size(len(self.consensus_normalized_dataframe)),
        }

    def _distance_source_dataframe(self, scope: str) -> pd.DataFrame:
        if scope == "consensus":
            return self.consensus_normalized_dataframe
        if scope == "individual":
            return self.normalized_dataframe
        raise ValueError(f"Unsupported distance scope: {scope}")

    def _serialisable_summary(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "feature_names": list(self.feature_names),
            "feature_version": self.feature_version,
            "fingerprint_version": self.fingerprint_version,
            "normalization_method": self.normalization_method,
            "normalization_parameters": self.normalization_parameters,
            "consensus_normalization_parameters": self.consensus_normalization_parameters,
            "consensus_group_columns": list(self.consensus_group_columns),
            "distance_estimates": self.distance_estimates(),
            "summary": self.summary,
            "qc": self.qc.to_summary_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def _canonical_distance_mode(distance_mode: DistanceMode | str) -> str:
    mode = str(distance_mode).strip().casefold()
    if mode not in {"none", "consensus", "individual"}:
        raise ValueError("distance_mode must be one of: none, consensus, individual.")
    return mode
