"""Select readable figures from inventory-listed files."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .report_models import SelectedSource


FIGURE_PREFERENCES: List[Tuple[str, List[str]]] = [
    ("workflow", ["workflow"]),
    ("consensus_fingerprint_heatmap", ["consensus_fingerprint_heatmap"]),
    ("pca_plot", ["pca_plot", "pca_scores"]),
    ("chemical_similarity_heatmap", ["chemical_similarity_heatmap"]),
    ("hierarchical_dendrogram", ["hierarchical_dendrogram", "dendrogram"]),
    ("confusion_matrix", ["confusion_matrix"]),
    ("classification_feature_importance", ["classification_feature_importance"]),
    ("prediction_vs_actual", ["prediction_vs_actual"]),
    ("residual_plot", ["residual"]),
    ("regression_feature_importance", ["regression_feature_importance"]),
    ("feature_family_comparison", ["feature_family", "ablation"]),
    ("performance_vs_feature_count", ["performance_vs_feature_count"]),
    ("strain_contribution_heatmap", ["chemical_specific_strain_heatmap", "strain_chemical_heatmap"]),
    ("leave_one_strain_out", ["leave_one_strain_out", "loeo"]),
]


def select_figures(sources: Iterable[SelectedSource], output_dir: Path) -> List[Dict[str, str]]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        source
        for source in sources
        if source.exists and source.source_kind == "figure" and Path(source.resolved_path).is_file()
    ]

    selected: List[Dict[str, str]] = []
    used_paths = set()
    for figure_id, patterns in FIGURE_PREFERENCES:
        match = None
        for source in candidates:
            lower = source.source_file.lower()
            if source.resolved_path in used_paths:
                continue
            if any(pattern in lower for pattern in patterns):
                match = source
                break
        if not match:
            continue
        destination_name = f"{figure_id}{Path(match.resolved_path).suffix.lower()}"
        destination = figure_dir / destination_name
        shutil.copy2(match.resolved_path, destination)
        used_paths.add(match.resolved_path)
        selected.append(
            {
                "figure_id": figure_id,
                "title": figure_id.replace("_", " ").title(),
                "source_file": match.source_file,
                "source_run": match.selected_run,
                "output_file": str(Path("figures") / destination_name),
                "status": "SELECTED",
                "notes": "Copied from inventory-listed readable figure.",
            }
        )
    return selected
