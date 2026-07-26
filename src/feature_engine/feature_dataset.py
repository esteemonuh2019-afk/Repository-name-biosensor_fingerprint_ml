"""Feature dataset container and output writing utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.feature_engine.feature_qc import FeatureQCResult, render_feature_qc_report


@dataclass(frozen=True)
class FeatureDataset:
    """Canonical feature extraction result."""

    dataframe: pd.DataFrame
    metadata: dict[str, Any]
    summary: dict[str, Any]
    qc: FeatureQCResult

    def write_outputs(
        self,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> list[Path]:
        """Write the Stage 6B feature dataset, summary, and QC report."""

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        feature_path = target / "feature_dataset.csv"
        summary_path = target / "feature_summary.json"
        report_path = target / "feature_qc_report.md"

        output_paths = [feature_path, summary_path, report_path]
        existing = [path for path in output_paths if path.exists()]
        if existing and not overwrite:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Feature output files already exist. Use overwrite=True to replace: "
                f"{formatted}"
            )

        self.dataframe.to_csv(feature_path, index=False, encoding="utf-8")
        summary_path.write_text(
            json.dumps(self._serialisable_summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report_path.write_text(
            render_feature_qc_report(self.qc, self.summary),
            encoding="utf-8",
        )
        return output_paths

    def _serialisable_summary(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "qc": self.qc.to_summary_dict(),
        }

