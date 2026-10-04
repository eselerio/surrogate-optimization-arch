"""Compatibility entry point for the canonical article workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_article_v3_5000 import (  # noqa: E402
    AUTHORIZED_DATASET_TOTALS,
    DEFAULT_RUN_ID,
    main as run_article,
    profile_for_dataset_total,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the canonical 10,000-accepted-state article study."
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("ARTICLE_V3_RUN_ID", "article_full_10000_001"),
        help="Immutable result identifier.",
    )
    parser.add_argument(
        "--dataset-count",
        type=int,
        choices=AUTHORIZED_DATASET_TOTALS,
        default=10_000,
        help="Accepted development-plus-holdout state count.",
    )
    parser.add_argument(
        "--through", choices=("generation", "assessment", "complete"),
        default="complete",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    profile = profile_for_dataset_total(arguments.dataset_count)
    run_article(
        arguments.run_id,
        arguments.through,
        profile=profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
