"""Conservative, source-grounded scientific interpretation text."""

from __future__ import annotations

from typing import Any, Dict, List


def compose_scientific_interpretation(package: Any) -> List[str]:
    classification_model = (
        package.classification_results.get("selected_model", {}).get("model_name") or "MISSING"
    )
    regression_model = package.regression_results.get("selected_model", {}).get("model_name") or "MISSING"
    blind = package.project_summary.get("blind_prediction_context", {})
    blind_status = blind.get("true_labels_included")

    statements = [
        (
            "The classification benchmark's own best-model metadata identifies "
            f"{classification_model} as the selected classifier; primary classification "
            "metrics in this package are restricted to that model."
        ),
        (
            "The regression benchmark's own best-model metadata identifies "
            f"{regression_model} as the selected regressor; primary regression metrics "
            "in this package are restricted to that model."
        ),
        (
            "Feature-family, feature-selection, and strain-ablation outputs are reported "
            "as separate analyses and are not used to replace the primary benchmark metrics."
        ),
    ]
    if blind_status is False:
        statements.append(
            "The selected blind-prediction run did not include true labels, so it is reported "
            "as prediction context rather than real blind-validation evidence."
        )
    return statements


def derive_limitations(
    dataset_summary: Dict[str, Any],
    qc_summary: Dict[str, Any],
    blind_summary: Dict[str, Any],
    documentation_sources: List[str],
) -> List[Dict[str, Any]]:
    limitations = []
    if qc_summary.get("canonical_qc_passed") is False:
        limitations.append(
            {
                "limitation": "Canonical QC did not fully pass.",
                "status": "ACTIVE",
                "source_file": qc_summary.get("canonical_qc_source"),
                "notes": "Reported QC errors and warnings are retained in the summary tables.",
            }
        )
    if qc_summary.get("feature_qc_passed") is False:
        limitations.append(
            {
                "limitation": "Feature QC did not fully pass.",
                "status": "ACTIVE",
                "source_file": qc_summary.get("feature_summary_source"),
                "notes": "Failed and warning feature rows are included as package limitations.",
            }
        )
    if blind_summary.get("true_labels_included") is False:
        limitations.append(
            {
                "limitation": "No real blind validation labels were available in the selected blind-prediction output.",
                "status": "ACTIVE",
                "source_file": blind_summary.get("source_file"),
                "notes": "The report avoids claiming blind-validation performance.",
            }
        )
    if documentation_sources:
        limitations.append(
            {
                "limitation": "Project documentation records dataset, biosensor, laboratory, and uncertainty constraints.",
                "status": "DOCUMENTED",
                "source_file": "; ".join(documentation_sources),
                "notes": "Documentation is listed in the selected inventory and cited as limitation context.",
            }
        )
    if not limitations:
        limitations.append(
            {
                "limitation": "No additional limitations were extracted from selected machine-readable sources.",
                "status": "MISSING",
                "source_file": None,
                "notes": "Review source reports before external circulation.",
            }
        )
    return limitations
