"""Fingerprint dataset container and output utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.fingerprint.fingerprint_qc import (
    FingerprintQCResult,
    render_fingerprint_qc_report,
)
from src.fingerprint.fingerprint_similarity import write_distance_matrix_csv


@dataclass(frozen=True)
class FingerprintDataset:
    """Validated fingerprint matrix plus scientific metadata."""

    dataframe: pd.DataFrame
    normalized_dataframe: pd.DataFrame
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
    excluded_dataframe: pd.DataFrame

    def write_outputs(
        self,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
        write_distances: bool = True,
        distance_chunk_size: int = 128,
    ) -> list[Path]:
        """Write fingerprint datasets, summaries, QC report, and distances."""

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        output_paths = [
            target / "fingerprint_dataset.csv",
            target / "fingerprint_dataset_normalized.csv",
            target / "fingerprint_summary.json",
            target / "fingerprint_qc_report.md",
        ]
        if write_distances:
            output_paths.extend(
                target / f"distance_matrix_{metric}.csv"
                for metric in ("euclidean", "cosine", "manhattan", "correlation")
            )

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
        summary_path = target / "fingerprint_summary.json"
        report_path = target / "fingerprint_qc_report.md"

        self.dataframe.to_csv(dataset_path, index=False, encoding="utf-8")
        created.append(dataset_path)
        self.normalized_dataframe.to_csv(normalized_path, index=False, encoding="utf-8")
        created.append(normalized_path)
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

        if write_distances:
            for metric in ("euclidean", "cosine", "manhattan", "correlation"):
                path = target / f"distance_matrix_{metric}.csv"
                write_distance_matrix_csv(
                    self.normalized_dataframe,
                    feature_names=self.feature_names,
                    metric=metric,
                    output_path=path,
                    label_column="Fingerprint_ID",
                    chunk_size=distance_chunk_size,
                )
                created.append(path)

        return created

    def _serialisable_summary(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "feature_names": list(self.feature_names),
            "feature_version": self.feature_version,
            "fingerprint_version": self.fingerprint_version,
            "normalization_method": self.normalization_method,
            "normalization_parameters": self.normalization_parameters,
            "summary": self.summary,
            "qc": self.qc.to_summary_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }
