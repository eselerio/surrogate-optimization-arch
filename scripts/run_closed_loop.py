"""Command-line entry point for the staged closed-loop article workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from closed_loop.workflow import ClosedLoopWorkflow, STAGES, WorkflowError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the physically constrained closed-loop study."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "params_closed_loop.json",
        help="Resolved closed-loop JSON configuration.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("CLOSED_LOOP_PROFILE"),
        help="Execution profile (unit, test_2000, or full).",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("CLOSED_LOOP_RUN_ID"),
        help="Immutable run identifier; may also use CLOSED_LOOP_RUN_ID.",
    )
    parser.add_argument(
        "--through",
        choices=STAGES,
        default="complete",
        help="Stop after this independently checkpointed stage.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Optional results-root override, primarily for isolated testing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    profile = arguments.profile or config["execution"]["default_profile"]
    if not arguments.run_id:
        parser.error("--run-id or CLOSED_LOOP_RUN_ID is required")
    try:
        workflow = ClosedLoopWorkflow(
            config_path=arguments.config,
            profile=profile,
            run_id=arguments.run_id,
            repository_root=REPOSITORY_ROOT,
            results_root=arguments.results_root,
        )
        manifest = workflow.run(through=arguments.through)
    except WorkflowError as exc:
        print(f"closed-loop workflow stopped: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "profile": manifest["profile"],
                "status": manifest["status"],
                "run_root": str(workflow.run_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

