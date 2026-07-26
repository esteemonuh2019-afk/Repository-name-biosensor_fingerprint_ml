"""Print a concise summary for one biosensor Excel candidate workbook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion.excel_reader import read_biosensor_excel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excel_file", help="Path to one biosensor Excel workbook.")
    args = parser.parse_args(argv)

    try:
        result = read_biosensor_excel(args.excel_file)
    except (FileNotFoundError, IsADirectoryError, ValueError) as error:
        print(f"Excel inspection failed: {error}", file=sys.stderr)
        return 1

    print(f"filename: {result.filename}")
    print("worksheets: " + ", ".join(result.worksheet_names))
    print(f"active sheet: {result.active_worksheet}")
    print(f"rows: {result.row_count}")
    print(f"columns: {result.column_count}")
    print(f"inferred strain: {result.inferred_strain or ''}")
    print("warnings:")
    if result.warnings:
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("- none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
