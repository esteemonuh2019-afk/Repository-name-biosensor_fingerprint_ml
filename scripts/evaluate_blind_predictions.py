"""Evaluate saved blind predictions after truth is revealed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blind_prediction import evaluate_blind_predictions  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = evaluate_blind_predictions(args.prediction_dir, args.truth_file)
    except (FileNotFoundError, ValueError) as error:
        print(f"Stage 9A blind evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("truth_file", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
