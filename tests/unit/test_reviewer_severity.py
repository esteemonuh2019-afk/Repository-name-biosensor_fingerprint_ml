from src.scientific_reasoning.reviewer import PublicationRisk, ReviewCategory, Severity
from src.scientific_reasoning.reviewer.policies import default_blocking, publication_risk_for


def test_critical_findings_are_blocking() -> None:
    assert default_blocking(ReviewCategory.TRACEABILITY, Severity.CRITICAL) is True
    assert publication_risk_for(Severity.CRITICAL, blocking=True) is PublicationRisk.BLOCKING


def test_information_findings_are_not_blocking() -> None:
    assert default_blocking(ReviewCategory.FIGURE_SUPPORT, Severity.INFORMATION) is False
    assert publication_risk_for(Severity.INFORMATION, blocking=False) is PublicationRisk.NONE


def test_external_validation_major_is_blocking() -> None:
    assert default_blocking(
        ReviewCategory.EXTERNAL_VALIDATION,
        Severity.MAJOR,
        ("REVIEW-VALIDATION-EXTERNAL-001",),
    )
