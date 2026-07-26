"""Print a concise summary for one biosensor CSV candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion.csv_reader import read_biosensor_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", help="Path to one biosensor CSV candidate.")
    args = parser.parse_args(argv)

    try:
        result = read_biosensor_csv(args.csv_file)
    except (FileNotFoundError, IsADirectoryError, ValueError, UnicodeError) as error:
        print(f"CSV inspection failed: {error}", file=sys.stderr)
        return 1

    print(f"filename: {result.source_file}")
    print(f"encoding: {result.encoding}")
    print(f"delimiter: {result.delimiter!r}")
    print(f"inferred strain: {result.strain_label_original or ''}")
    print(f"rows: {result.row_count}")
    print(f"columns: {result.column_count}")
    print("first columns: " + ", ".join(result.original_columns[:8]))
    print("warnings:")
    if result.warnings:
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("- none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
