"""Read-only discovery of biosensor source files."""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

CSV_24H_CANDIDATE = "csv_24h_candidate"
EXCEL_12H_CANDIDATE = "excel_12h_candidate"
UNKNOWN_SUPPORTED_FILE = "unknown_supported_file"

_STRAIN_PATTERN = re.compile(r"BL\d{3}(?:ab)?", re.IGNORECASE)
_DURATION_HINT_PATTERN = re.compile(
    r"(?<!\d)((?:12|24)\s*h(?:ours?|rs?)?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BiosensorFileRecord:
    """Metadata discovered from a supported biosensor source filename."""

    absolute_path: str
    filename: str
    extension: str
    source_type: str
    file_size_bytes: int
    strain_label_from_filename: str | None
    duration_hint_from_filename: str | None


@dataclass(frozen=True)
class BiosensorDiscoveryResult:
    """Read-only discovery result and non-fatal warnings."""

    files: list[BiosensorFileRecord]
    warnings: list[str]


def discover_biosensor_files(folder_path: str | Path) -> BiosensorDiscoveryResult:
    """Discover supported biosensor source files in a folder without reading contents."""

    folder = Path(folder_path).expanduser()
    if not folder.exists():
        raise FileNotFoundError(str(folder))
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))

    records = [
        _record_from_path(path)
        for path in folder.iterdir()
        if _is_discoverable_file(path)
    ]
    records = sorted(records, key=lambda record: (record.filename.casefold(), record.filename))

    warnings = _build_warnings(folder, records)
    return BiosensorDiscoveryResult(files=records, warnings=warnings)


def _is_discoverable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    if path.name.startswith("~$"):
        return False
    if _has_hidden_attribute(path):
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _has_hidden_attribute(path: Path) -> bool:
    try:
        attributes = path.stat().st_file_attributes
    except AttributeError:
        return False
    except OSError:
        return False

    return bool(attributes & stat.FILE_ATTRIBUTE_HIDDEN)


def _record_from_path(path: Path) -> BiosensorFileRecord:
    absolute_path = path.resolve()
    extension = path.suffix.lower()
    duration_hint = _infer_duration_hint(path.name)

    return BiosensorFileRecord(
        absolute_path=str(absolute_path),
        filename=path.name,
        extension=extension,
        source_type=_classify_source_type(extension, duration_hint),
        file_size_bytes=path.stat().st_size,
        strain_label_from_filename=_infer_strain_label(path.name),
        duration_hint_from_filename=duration_hint,
    )


def _infer_strain_label(filename: str) -> str | None:
    match = _STRAIN_PATTERN.search(filename)
    if match is None:
        return None
    return match.group(0)


def _infer_duration_hint(filename: str) -> str | None:
    match = _DURATION_HINT_PATTERN.search(filename)
    if match is None:
        return None
    return match.group(1).replace(" ", "")


def _classify_source_type(extension: str, duration_hint: str | None) -> str:
    normalized_duration = _normalize_duration_hint(duration_hint)

    if extension == ".csv" and normalized_duration in {None, "24h"}:
        return CSV_24H_CANDIDATE
    if extension == ".xlsx" and normalized_duration in {None, "12h"}:
        return EXCEL_12H_CANDIDATE
    return UNKNOWN_SUPPORTED_FILE


def _normalize_duration_hint(duration_hint: str | None) -> str | None:
    if duration_hint is None:
        return None

    normalized = duration_hint.casefold()
    if normalized.startswith("12"):
        return "12h"
    if normalized.startswith("24"):
        return "24h"
    return None


def _build_warnings(
    folder: Path,
    records: list[BiosensorFileRecord],
) -> list[str]:
    warnings: list[str] = []

    if not records:
        warnings.append(f"No supported biosensor source files found in: {folder.resolve()}")
        return warnings

    for record in records:
        if record.strain_label_from_filename is None:
            warnings.append(f"Could not infer expected strain from filename: {record.filename}")

    duplicate_groups: dict[tuple[str, str], list[str]] = {}
    for record in records:
        if record.strain_label_from_filename is None:
            continue
        key = (record.strain_label_from_filename, record.source_type)
        duplicate_groups.setdefault(key, []).append(record.filename)

    for (strain_label, source_type), filenames in sorted(duplicate_groups.items()):
        if len(filenames) <= 1:
            continue
        joined_filenames = ", ".join(sorted(filenames, key=lambda name: name.casefold()))
        warnings.append(
            "Duplicate strain/source-type candidates found for "
            f"{strain_label} / {source_type}: {joined_filenames}"
        )

    return warnings
