import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from closed_loop.manuscript_v3 import (
    ARTICLE_FULL,
    TEST_500,
    AssessmentResult,
    StudyProfile,
    create_design,
)
from closed_loop.projection import (
    LeastSquaresDiagnostics,
    QuadraticFeatureMap,
    QuadraticSurrogate,
)
from scripts import run_article_v3_5000 as runner


ROOT = Path(__file__).resolve().parents[1]


def tiny_profile() -> StudyProfile:
    return StudyProfile(
        name="tiny_test", development_count=5, test_count=2,
        robustness_count=1, layer_count=3,
        development_seed=1, test_seed=2, robustness_seed=3,
        parallel_workers=1, article_eligible=False, enforce_admission_gate=False,
    )


def tiny_design(profile: StudyProfile) -> dict[str, object]:
    decision = 0.5 * (runner.DECISION_LOWER + runner.DECISION_UPPER)
    influent = 0.5 * (runner.INFLUENT_LOWER + runner.INFLUENT_UPPER)
    return {
        "development_decisions": np.tile(decision, (profile.development_count, 1)),
        "development_influents": np.tile(influent, (profile.development_count, 1)),
        "test_decisions": np.tile(decision, (profile.test_count, 1)),
        "test_influents": np.tile(influent, (profile.test_count, 1)),
        "robustness_influents": np.tile(influent, (profile.robustness_count, 1)),
        "generators": {
            "development": {"seed": profile.development_seed},
            "test": {"seed": profile.test_seed},
            "robustness": {"seed": profile.robustness_seed},
        },
    }


def accepted_diagnostics(count: int) -> pd.DataFrame:
    return pd.DataFrame({
        "row": np.arange(count),
        "accepted": np.ones(count, dtype=bool),
        "root_difference_inf": np.full(count, 1.0e-8),
        "branch_agreement": np.ones(count, dtype=bool),
        "mass_residual_start_1": np.full(count, 1.0e-10),
        "mass_residual_start_2": np.full(count, 1.0e-10),
        "state_negativity_start_1": np.zeros(count),
        "state_negativity_start_2": np.zeros(count),
        "rate_negativity_start_1": np.zeros(count),
        "rate_negativity_start_2": np.zeros(count),
        "largest_real_eigenvalue_start_1": np.full(count, -1.0),
        "largest_real_eigenvalue_start_2": np.full(count, -1.0),
        "stability_agreement_start_1": np.full(count, 1.0e-8),
        "stability_agreement_start_2": np.full(count, 1.0e-8),
        "feed_tss_start_1": np.full(count, 100.0),
        "feed_tss_start_2": np.full(count, 100.0),
        "external_solids_loss_start_1": np.full(count, 10.0),
        "external_solids_loss_start_2": np.full(count, 10.0),
    })


def migration_fixture(run: Path):
    """Create a three-row predecessor tree and its exact migration authorization."""

    decisions = np.arange(21, dtype=float).reshape(3, 7)
    influents = np.arange(60, dtype=float).reshape(3, 20)
    targets = np.arange(12, dtype=float).reshape(3, 4)
    states_1 = np.arange(15, dtype=float).reshape(3, 5)
    states_2 = states_1 + 0.25
    runner.atomic_json(run / "inputs/generator_records.json", {"seed": 123})
    runner.atomic_npz(
        run / "datasets/design.npz",
        development_decisions=decisions,
        development_influents=influents,
    )
    runner.atomic_npz(
        run / "datasets/development/mechanistic_rows_v3.npz",
        targets=targets,
        states_start_1=states_1,
        states_start_2=states_2,
    )
    diagnostics = pd.DataFrame({
        "row": [0, 1, 2], "accepted": [True, False, True],
    })
    runner.atomic_dataframe(
        run / "datasets/development/mechanistic_diagnostics.csv", diagnostics,
    )
    for index in (0, 2):
        runner.atomic_npz(
            run / f"datasets/development/rows/row_{index:06d}.npz",
            contract_hash=np.asarray("legacy-row-contract"),
            decision=decisions[index], influent=influents[index],
            target=targets[index], state_start_1=states_1[index],
            state_start_2=states_2[index],
            record_json=np.asarray(json.dumps({
                "row": index, "accepted": True,
            }, sort_keys=True)),
        )
    old_files = {
        "stable.py": sha256(b"stable-v2").hexdigest(),
        "runner.py": sha256(b"runner-v2").hexdigest(),
    }
    new_files = {
        "stable.py": sha256(b"stable-v2").hexdigest(),
        "runner.py": sha256(b"runner-v3").hexdigest(),
        "replacement.py": sha256(b"replacement-v3").hexdigest(),
    }
    predecessor = {
        "runner_schema": 2, "run_id": "unit-run", "fixed_dataset_total": 3,
        "profile": {"count": 3}, "source_digest": runner.source_digest(old_files),
        "source_files": old_files,
    }
    successor = {
        **predecessor, "runner_schema": 3,
        "source_digest": runner.source_digest(new_files), "source_files": new_files,
    }
    runner.establish_contract(run, predecessor)
    artifacts = {
        relative: runner.file_digest(run / relative)
        for relative in (
            "inputs/generator_records.json", "datasets/design.npz",
            "datasets/development/mechanistic_rows_v3.npz",
            "datasets/development/mechanistic_diagnostics.csv",
        )
    }
    authorization = runner.SourceContractMigrationAuthorization(
        migration_id="unit-generation-replacement-v1",
        run_id="unit-run", authorized_date="2026-08-23", reason="unit test",
        predecessor_runner_schema=2, successor_runner_schema=3,
        predecessor_source_digest=predecessor["source_digest"],
        predecessor_contract_file_digest=runner.file_digest(
            run / "inputs/contract.json"
        ),
        allowed_changed_source_files=frozenset({"runner.py", "replacement.py"}),
        required_changed_source_files=frozenset({"runner.py", "replacement.py"}),
        required_artifact_digests=artifacts,
        expected_accepted_rows=2, expected_rejected_rows=1,
    )
    return predecessor, successor, authorization


def assessment_migration_fixture(run: Path):
    """Create and apply two chained migrations with one retained stage marker."""

    _, schema_3, first_authorization = migration_fixture(run)
    with patch.object(
        runner, "GENERATION_REPLACEMENT_MIGRATION", first_authorization,
    ):
        runner.establish_contract(
            run, schema_3, authorize_generation_replacement_migration=True,
        )
    predecessor = json.loads((run / "inputs/contract.json").read_text())
    snapshot_relative = (
        "inputs/contract_migrations/unit-assessment-v1-predecessor-source"
    )
    snapshot = run / snapshot_relative
    snapshot_content = {
        "stable.py": b"stable-v2",
        "runner.py": b"runner-v3",
        "replacement.py": b"replacement-v3",
    }
    for relative, content in snapshot_content.items():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    marker = run / "models/ridge_complete.json"
    runner.atomic_json(marker, {
        "source_digest": predecessor["source_digest"], "artifacts": {},
    })
    successor_files = {
        **predecessor["source_files"],
        "runner.py": sha256(b"runner-v4").hexdigest(),
    }
    successor = {
        **{key: value for key, value in predecessor.items()
           if key != "contract_migrations"},
        "runner_schema": 4,
        "source_digest": runner.source_digest(successor_files),
        "source_files": successor_files,
        "assessment_gate_execution_policy": runner.ASSESSMENT_GATE_EXECUTION_POLICY,
    }
    authorization = runner.AssessmentRecoveryMigrationAuthorization(
        migration_id="unit-assessment-v1", run_id="unit-run",
        authorized_date="2026-08-23", reason="unit assessment recovery",
        predecessor_runner_schema=3, successor_runner_schema=4,
        predecessor_source_digest=predecessor["source_digest"],
        predecessor_contract_file_digest=runner.file_digest(
            run / "inputs/contract.json"
        ),
        required_prior_migration_ids=("unit-generation-replacement-v1",),
        predecessor_source_snapshot=snapshot_relative,
        allowed_changed_source_files=frozenset({"runner.py"}),
        required_changed_source_files=frozenset({"runner.py"}),
        required_artifact_digests={},
        expected_effective_design_digest="effective",
        expected_ridge_input_digest="ridge-input",
    )
    retained = {
        "schema": 1,
        "source_digest": predecessor["source_digest"],
        "effective_design_digest": "effective",
        "ridge_input_digest": "ridge-input",
        "pinned_artifacts": {},
        "stages": {
            "ridge": {
                "checkpoint": marker.relative_to(run).as_posix(),
                "checkpoint_sha256": runner.file_digest(marker),
                "artifact_source_digest": predecessor["source_digest"],
                "artifacts": {},
            },
        },
    }
    with patch.object(
        runner, "ASSESSMENT_RECOVERY_MIGRATION", authorization,
    ), patch.object(
        runner, "_validate_retained_assessment_checkpoints",
        return_value=retained,
    ):
        runner.establish_contract(
            run, successor, authorize_assessment_recovery_migration=True,
        )
    return predecessor, successor, authorization, marker


def optimization_migration_fixture(run: Path):
    """Create and apply the third migration with assessment carry-forward."""

    _, schema_4, _, ridge_marker = assessment_migration_fixture(run)
    predecessor = json.loads((run / "inputs/contract.json").read_text())
    snapshot_relative = (
        "inputs/contract_migrations/unit-optimization-v1-predecessor-source"
    )
    snapshot = run / snapshot_relative
    snapshot_content = {
        "stable.py": b"stable-v2",
        "runner.py": b"runner-v4",
        "replacement.py": b"replacement-v3",
    }
    for relative, content in snapshot_content.items():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    assessment_artifact = run / "metrics/assessment_payload.csv"
    runner.atomic_dataframe(assessment_artifact, pd.DataFrame({"value": [1.0]}))
    gate_path = run / "metrics/admission_gate.json"
    runner.atomic_json(gate_path, {
        "passed": False,
        "execution_policy": runner.ASSESSMENT_GATE_EXECUTION_POLICY,
        "optimization_permitted": True,
    })
    assessment_marker = run / "metrics/assessment_complete.json"
    runner.atomic_json(assessment_marker, {
        "source_digest": predecessor["source_digest"],
        "input_digest": "assessment-input",
        "passed": False,
        "artifacts": {
            assessment_artifact.relative_to(run).as_posix(): runner.file_digest(
                assessment_artifact
            ),
            gate_path.relative_to(run).as_posix(): runner.file_digest(gate_path),
        },
    })
    successor_files = {
        **predecessor["source_files"],
        "runner.py": sha256(b"runner-v5").hexdigest(),
    }
    successor = {
        **{key: value for key, value in predecessor.items()
           if key != "contract_migrations"},
        "runner_schema": 5,
        "source_digest": runner.source_digest(successor_files),
        "source_files": successor_files,
        "optimization_protocol": runner.OPTIMIZATION_PROTOCOL,
    }
    authorization = runner.OptimizationProtocolMigrationAuthorization(
        migration_id="unit-optimization-v1", run_id="unit-run",
        authorized_date="2026-08-23", reason="unit optimization migration",
        predecessor_runner_schema=4, successor_runner_schema=5,
        predecessor_source_digest=predecessor["source_digest"],
        predecessor_contract_file_digest=runner.file_digest(
            run / "inputs/contract.json"
        ),
        required_prior_migration_ids=(
            "unit-generation-replacement-v1", "unit-assessment-v1",
        ),
        predecessor_source_snapshot=snapshot_relative,
        allowed_changed_source_files=frozenset({"runner.py"}),
        required_changed_source_files=frozenset({"runner.py"}),
        required_artifact_digests={
            "metrics/assessment_complete.json": runner.file_digest(assessment_marker),
            "metrics/admission_gate.json": runner.file_digest(gate_path),
            "metrics/assessment_payload.csv": runner.file_digest(assessment_artifact),
        },
        expected_assessment_input_digest="assessment-input",
    )
    retention = {
        "schema": 2,
        "predecessor_source_digest": predecessor["source_digest"],
        "effective_design_digest": "effective",
        "ridge_input_digest": "ridge-input",
        "assessment_input_digest": "assessment-input",
        "pinned_artifacts": dict(authorization.required_artifact_digests),
        "stages": {
            "ridge": {
                "checkpoint": ridge_marker.relative_to(run).as_posix(),
                "checkpoint_sha256": runner.file_digest(ridge_marker),
                "artifact_source_digest": json.loads(
                    ridge_marker.read_text()
                )["source_digest"],
                "artifacts": {},
            },
            "assessment": {
                "checkpoint": assessment_marker.relative_to(run).as_posix(),
                "checkpoint_sha256": runner.file_digest(assessment_marker),
                "artifact_source_digest": predecessor["source_digest"],
                "artifacts": {
                    assessment_artifact.relative_to(run).as_posix(): (
                        runner.file_digest(assessment_artifact)
                    ),
                },
            },
        },
    }
    with patch.object(
        runner, "SINGLE_START_EXACT_QP_MIGRATION", authorization,
    ), patch.object(
        runner, "_validate_retained_optimization_protocol_checkpoints",
        return_value=retention,
    ):
        runner.establish_contract(
            run, successor, authorize_single_start_exact_qp_migration=True,
        )
    return predecessor, successor, authorization, assessment_marker, ridge_marker


def casewise_migration_fixture(run: Path):
    """Apply the historical schema-6 common-reference migration."""

    optimization_migration_fixture(run)
    predecessor = json.loads((run / "inputs/contract.json").read_text())
    snapshot_relative = (
        "inputs/contract_migrations/unit-casewise-v1-predecessor-source"
    )
    snapshot = run / snapshot_relative
    for relative, content in {
        "stable.py": b"stable-v2",
        "runner.py": b"runner-v5",
        "replacement.py": b"replacement-v3",
    }.items():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    case_directory = run / "optimization/nominal"
    route_path = case_directory / "surrogate.json"
    runner.atomic_json(route_path, {"route_contract": "retained-route"})
    marker_path = case_directory / "case_complete.json"
    runner.atomic_json(marker_path, {
        "stage": "optimization_case",
        "case": "nominal",
    })
    artifacts = {
        route_path.relative_to(run).as_posix(): runner.file_digest(route_path),
    }
    marker_set_digest = "unit-case-marker-set"
    retention = {
        "schema": 3,
        "predecessor_source_digest": predecessor["source_digest"],
        "effective_design_digest": "effective",
        "ridge_input_digest": "ridge-input",
        "assessment_input_digest": "assessment-input",
        "case_marker_set_digest": marker_set_digest,
        "pinned_artifacts": {},
        "retired_unfinished_stage": "validation/untouched_test_equivalence",
        "stages": {
            "optimization/nominal": {
                "checkpoint": marker_path.relative_to(run).as_posix(),
                "checkpoint_sha256": runner.file_digest(marker_path),
                "artifact_source_digest": predecessor["source_digest"],
                "artifacts": artifacts,
            },
        },
    }
    successor_files = {
        **predecessor["source_files"],
        "runner.py": sha256(b"runner-v6").hexdigest(),
    }
    successor = {
        **{
            key: value for key, value in predecessor.items()
            if key != "contract_migrations"
        },
        "runner_schema": 6,
        "source_digest": runner.source_digest(successor_files),
        "source_files": successor_files,
        "validation_protocol": "casewise_exact_common_reference_v1",
    }
    authorization = runner.CasewiseComparisonMigrationAuthorization(
        migration_id="unit-casewise-v1",
        run_id="unit-run",
        authorized_date="2026-08-24",
        reason="unit casewise comparison migration",
        predecessor_runner_schema=5,
        successor_runner_schema=6,
        predecessor_source_digest=predecessor["source_digest"],
        predecessor_contract_file_digest=runner.file_digest(
            run / "inputs/contract.json"
        ),
        required_prior_migration_ids=(
            "unit-generation-replacement-v1",
            "unit-assessment-v1",
            "unit-optimization-v1",
        ),
        predecessor_source_snapshot=snapshot_relative,
        allowed_changed_source_files=frozenset({"runner.py"}),
        required_changed_source_files=frozenset({"runner.py"}),
        required_artifact_digests={},
        expected_case_marker_set_digest=marker_set_digest,
    )
    with (
        patch.object(runner, "CASEWISE_COMMON_REFERENCE_MIGRATION", authorization),
        patch.object(
            runner,
            "_validate_retained_casewise_comparison_checkpoints",
            return_value=retention,
        ),
        patch.object(
            runner, "COMPARISON_PROTOCOL", "casewise_exact_common_reference_v1",
        ),
    ):
        runner.establish_contract(
            run,
            successor,
            authorize_casewise_common_reference_migration=True,
        )
    return successor, authorization, retention, case_directory


def publish_mock_replacement_block(
    output: Path,
    decisions: np.ndarray,
    influents: np.ndarray,
    profile: StudyProfile,
    *,
    accepted: bool = True,
) -> runner.MechanisticBlockResult:
    count = len(decisions)
    targets = np.full((count, profile.response_count), 2.0)
    diagnostics = accepted_diagnostics(count)
    if not accepted:
        diagnostics.loc[0, "accepted"] = False
    provenance = pd.DataFrame({
        "accepted_slot": np.arange(count),
        "source_candidate_id": [f"candidate-{index}" for index in range(count)],
        "source_candidate_round": np.zeros(count, dtype=int),
        "source_candidate_index": np.arange(count),
        "source_candidate_ordinal": np.arange(count),
        "replaced_base_candidate": np.zeros(count, dtype=bool),
    })
    attempt_rows = []
    for index in range(count):
        relative = f"rows/row_{index:06d}.npz"
        checkpoint = output / relative
        runner.atomic_npz(checkpoint, value=np.asarray([index]))
        attempt_rows.append({
            "candidate_id": f"candidate-{index}",
            "accepted": bool(diagnostics.loc[index, "accepted"]),
            "rejection_reason": (
                "accepted" if bool(diagnostics.loc[index, "accepted"])
                else "branch_disagreement"
            ),
            "checkpoint_path": relative,
            "checkpoint_sha256": runner.file_digest(checkpoint),
        })
    attempts = pd.DataFrame(attempt_rows)
    runner.atomic_npz(
        output / "mechanistic_accepted_v3.npz", targets=targets,
        states_start_1=np.zeros((count, 1)), states_start_2=np.zeros((count, 1)),
    )
    runner.atomic_npz(
        output / "accepted_inputs.npz", decisions=decisions, influents=influents,
        source_candidate_id=provenance["source_candidate_id"].to_numpy(str),
    )
    runner.atomic_dataframe(output / "accepted_diagnostics.csv", diagnostics)
    runner.atomic_dataframe(output / "all_attempts.csv", attempts)
    runner.atomic_dataframe(output / "accepted_provenance.csv", provenance)
    runner.atomic_dataframe(output / "base_checkpoint_migration.csv", attempts)
    runner.atomic_json(output / "replacement_summary.json", {
        "accepted": count, "supplemental_round_count": 0,
    })
    return runner.MechanisticBlockResult(
        decisions=np.asarray(decisions), influents=np.asarray(influents),
        targets=targets, diagnostics=diagnostics, attempts=attempts,
        provenance=provenance,
    )


def assessment_fixture(raw_nrmse: float = 0.8) -> AssessmentResult:
    test_count, response_count = 2, 3
    metrics = pd.DataFrame([{
        "method": "raw", "block": "complete_response", "coordinate": "ALL",
        "nrmse": raw_nrmse,
    }])
    qp = pd.DataFrame([
        {"row": row, "projection_input": kind, "accepted": True}
        for row in range(test_count)
        for kind in ("raw_prediction", "mechanistic_target")
    ])
    violations = pd.DataFrame([
        {
            "case": f"test_{row:04d}", "method": method,
            "mass_conservation_violation_max": 2.0 if method == "raw" else 1.0e-9,
            "nonnegativity_violation_max": 3.0 if method == "raw" else 1.0e-12,
        }
        for row in range(test_count)
        for method in ("raw", "projected", "mechanistic")
    ])
    feasibility = pd.DataFrame({"row": range(test_count), "bound_passed": [True, True]})
    values = np.zeros((test_count, response_count))
    return AssessmentResult(
        metrics=metrics, violations=violations, qp_diagnostics=qp,
        feasibility=feasibility, raw=values.copy(), projected=values.copy(),
        projected_targets=values.copy(),
    )


TRUST_LIMITS = {
    "correction": 0.4,
    "regularized_leverage": 1.0,
    "particulate_split": 1.0,
    "reactor_residual": 1.0,
}


class ArticleV3FiveThousandContractTests(unittest.TestCase):
    def test_full_profile_is_exactly_four_thousand_plus_one_thousand(self) -> None:
        runner.validate_authorized_profile(ARTICLE_FULL)
        self.assertEqual(ARTICLE_FULL.development_count, 4_000)
        self.assertEqual(ARTICLE_FULL.test_count, 1_000)
        self.assertEqual(ARTICLE_FULL.development_count + ARTICLE_FULL.test_count, 5_000)
        self.assertEqual(ARTICLE_FULL.robustness_count, 10)
        self.assertEqual(ARTICLE_FULL.layer_count, 10)
        self.assertTrue(ARTICLE_FULL.article_eligible)
        self.assertTrue(ARTICLE_FULL.enforce_admission_gate)

    def test_full_design_is_distinct_from_smoke_test(self) -> None:
        self.assertNotEqual(ARTICLE_FULL.development_seed, TEST_500.development_seed)
        self.assertNotEqual(ARTICLE_FULL.test_seed, TEST_500.test_seed)
        design = create_design(ARTICLE_FULL)
        runner.validate_design(design, ARTICLE_FULL)
        self.assertEqual(design["development_decisions"].shape, (4_000, 7))
        self.assertEqual(design["development_influents"].shape, (4_000, 20))
        self.assertEqual(design["test_decisions"].shape, (1_000, 7))
        self.assertEqual(design["test_influents"].shape, (1_000, 20))
        self.assertEqual(design["robustness_influents"].shape, (10, 20))

    def test_json_profile_matches_executable_profile(self) -> None:
        payload = json.loads((ROOT / "config/params_manuscript_v3.json").read_text())
        profile = payload["profiles"]["article_full"]
        self.assertEqual(profile["development_count"], 4_000)
        self.assertEqual(profile["test_count"], 1_000)
        self.assertTrue(profile["continue_after_article_admission_gate_failure"])
        self.assertEqual(runner.ASSESSMENT_GATE_EXECUTION_POLICY, "advisory_continue")

    def test_source_contract_covers_every_scientific_route_implementation(self) -> None:
        required = {
            "scripts/run_article_v3_5000.py",
            "scripts/build_main_closed_loop_v3.py",
            "main_closed_loop.ipynb",
            "closed_loop/__init__.py",
            "closed_loop/design.py",
            "closed_loop/model.py",
            "closed_loop/manuscript_v3.py",
            "closed_loop/projection.py",
            "closed_loop/surrogate.py",
            "closed_loop/workflow.py",
            "closed_loop/v3_smooth.py",
            "closed_loop/v3_surrogate_nlp.py",
            "closed_loop/v3_active_set.py",
            "closed_loop/v3_parallel.py",
            "closed_loop/v3_shared_unit.py",
            "closed_loop/v3_trust.py",
            "closed_loop/v3_reporting.py",
            "closed_loop/v3_replacement_generation.py",
            "config/params_manuscript_v3.json",
            "pyproject.toml",
            "uv.lock",
        }
        self.assertTrue(required.issubset(set(runner.SOURCE_FILES)))
        manifest = runner.source_file_digests()
        self.assertTrue(required.issubset(manifest))
        self.assertTrue(all(len(manifest[name]) == 64 for name in required))

    def test_mutable_article_documentation_is_outside_source_contract(self) -> None:
        self.assertFalse(
            any(
                Path(relative).as_posix().startswith("article/")
                for relative in runner.SOURCE_FILES
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "code.py").write_text("VALUE = 1\n", encoding="utf-8")
            article = root / "article" / "wip_v3" / "manuscript.tex"
            article.parent.mkdir(parents=True)
            article.write_text("first draft\n", encoding="utf-8")
            with patch.object(runner, "ROOT", root), patch.object(
                runner, "SOURCE_FILES", ("code.py",),
            ):
                before = runner.source_file_digests()
                contract = runner._build_contract(
                    "article_full_5000_doc_edit_test", tiny_profile(), before,
                )
                runner.establish_contract(root / "run", contract)
                article.write_text("continuously revised draft\n", encoding="utf-8")
                runner.assert_source_unchanged(before)
                self.assertEqual(before, runner.source_file_digests())
                resumed_contract = runner._build_contract(
                    "article_full_5000_doc_edit_test",
                    tiny_profile(),
                    runner.source_file_digests(),
                )
                runner.establish_contract(root / "run", resumed_contract)

                (root / "code.py").write_text("VALUE = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "computational source changed",
                ):
                    runner.assert_source_unchanged(before)

    def test_source_contract_rejects_reintroducing_article_documentation(self) -> None:
        with patch.object(
            runner, "SOURCE_FILES", ("article/wip_v3/manuscript.tex",),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "must exclude mutable article documentation",
            ):
                runner.source_file_digests()

    def test_run_directory_cannot_alias_preflight_or_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = runner.resolve_run_directory("article_full_5000_trial1", root)
            self.assertEqual(accepted.parent, root.resolve())
            for value in (
                "test_500_l5_revision_001", "article_v3_full_001",
                "../article_full_5000_x", "article_full_5000_..", "C:\\temp",
            ):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    runner.resolve_run_directory(value, root)

    def test_contract_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            first = {"source_digest": "a", "fixed_dataset_total": 5_000}
            runner.establish_contract(run, first)
            runner.establish_contract(run, dict(first))
            with self.assertRaisesRegex(RuntimeError, "choose a new run id"):
                runner.establish_contract(run, {**first, "source_digest": "b"})

    def test_assessment_batch_contract_binds_stage_source_and_inputs(self) -> None:
        baseline = runner._assessment_batch_contract(
            stage="whole_system_holdout_projection_audit",
            source_id="source-a",
            input_id="input-a",
        )
        self.assertEqual(
            baseline,
            runner._assessment_batch_contract(
                stage="whole_system_holdout_projection_audit",
                source_id="source-a",
                input_id="input-a",
            ),
        )
        variants = (
            runner._assessment_batch_contract(
                stage="shared_unit_holdout_projection_audit",
                source_id="source-a",
                input_id="input-a",
            ),
            runner._assessment_batch_contract(
                stage="whole_system_holdout_projection_audit",
                source_id="source-b",
                input_id="input-a",
            ),
            runner._assessment_batch_contract(
                stage="whole_system_holdout_projection_audit",
                source_id="source-a",
                input_id="input-b",
            ),
        )
        self.assertTrue(all(value != baseline for value in variants))

    def test_explicit_migration_retains_verified_rows_and_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            _, successor, authorization = migration_fixture(run)
            retained_paths = [
                run / "datasets/development/rows/row_000000.npz",
                run / "datasets/development/rows/row_000002.npz",
            ]
            retained_before = [runner.file_digest(path) for path in retained_paths]
            with patch.object(
                runner, "GENERATION_REPLACEMENT_MIGRATION", authorization,
            ):
                with self.assertRaisesRegex(RuntimeError, "explicit authorized"):
                    runner.establish_contract(run, successor)
                runner.establish_contract(
                    run, successor,
                    authorize_generation_replacement_migration=True,
                )
                # Once pinned, ordinary same-source resume no longer needs the flag.
                runner.establish_contract(run, successor)
            self.assertEqual(
                retained_before,
                [runner.file_digest(path) for path in retained_paths],
            )
            migrated = json.loads((run / "inputs/contract.json").read_text())
            self.assertEqual(migrated["source_digest"], successor["source_digest"])
            self.assertEqual(len(migrated["contract_migrations"]), 1)
            history = migrated["contract_migrations"][0]
            record_path = run / history["record"]
            self.assertEqual(runner.file_digest(record_path), history["record_digest"])
            record = json.loads(record_path.read_text())
            self.assertEqual(
                record["predecessor"]["source_digest"],
                authorization.predecessor_source_digest,
            )
            self.assertEqual(
                record["successor"]["source_digest"], successor["source_digest"],
            )
            retained = json.loads((
                run / record["retained_checkpoint_manifest"]["path"]
            ).read_text())
            self.assertEqual(retained["retained_accepted_count"], 2)
            self.assertEqual(retained["excluded_original_candidate_indices"], [1])
            self.assertEqual(len(retained["checkpoints"]), 2)
            archived = run / history["predecessor_contract"]
            self.assertEqual(
                runner.file_digest(archived),
                history["predecessor_contract_digest"],
            )

    def test_migration_refuses_artifact_tampering_and_arbitrary_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            _, successor, authorization = migration_fixture(run)
            drifted = {
                **successor,
                "source_files": {
                    **successor["source_files"], "unrelated.py": "e" * 64,
                },
            }
            drifted["source_digest"] = runner.source_digest(drifted["source_files"])
            with patch.object(
                runner, "GENERATION_REPLACEMENT_MIGRATION", authorization,
            ), self.assertRaisesRegex(RuntimeError, "arbitrary source drift"):
                runner.establish_contract(
                    run, drifted,
                    authorize_generation_replacement_migration=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            _, successor, authorization = migration_fixture(run)
            with (run / "datasets/development/mechanistic_diagnostics.csv").open(
                "a", encoding="utf-8",
            ) as stream:
                stream.write("tampered")
            with patch.object(
                runner, "GENERATION_REPLACEMENT_MIGRATION", authorization,
            ), self.assertRaisesRegex(RuntimeError, "pinned artifact bytes changed"):
                runner.establish_contract(
                    run, successor,
                    authorize_generation_replacement_migration=True,
                )

    def test_assessment_recovery_migration_appends_chain_and_authorizes_exact_stage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            predecessor, successor, authorization, marker = (
                assessment_migration_fixture(run)
            )
            migrated = json.loads((run / "inputs/contract.json").read_text())
            self.assertEqual(
                [entry["migration_id"] for entry in migrated["contract_migrations"]],
                ["unit-generation-replacement-v1", "unit-assessment-v1"],
            )
            second = migrated["contract_migrations"][1]
            archived = json.loads((run / second["predecessor_contract"]).read_text())
            self.assertEqual(
                archived["contract_migrations"], migrated["contract_migrations"][:1],
            )
            runner._validate_migration_history(run, migrated)
            with patch.object(
                runner, "ASSESSMENT_RECOVERY_MIGRATION", authorization,
            ):
                self.assertTrue(runner._checkpoint_source_is_authorized(
                    run, stage="ridge", checkpoint=marker,
                    observed_source_id=predecessor["source_digest"],
                    current_source_id=successor["source_digest"],
                ))
                self.assertFalse(runner._checkpoint_source_is_authorized(
                    run, stage="generation/development", checkpoint=marker,
                    observed_source_id=predecessor["source_digest"],
                    current_source_id=successor["source_digest"],
                ))
            # An ordinary same-source resume validates both history entries.
            runner.establish_contract(run, successor)

    def test_assessment_recovery_rejects_snapshot_or_checkpoint_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            _, successor, authorization, marker = assessment_migration_fixture(run)
            marker.write_text("tampered", encoding="utf-8")
            with patch.object(
                runner, "ASSESSMENT_RECOVERY_MIGRATION", authorization,
            ):
                self.assertFalse(runner._checkpoint_source_is_authorized(
                    run, stage="ridge", checkpoint=marker,
                    observed_source_id=authorization.predecessor_source_digest,
                    current_source_id=successor["source_digest"],
                ))

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            _, _, _, _ = assessment_migration_fixture(run)
            snapshot = (
                run / "inputs/contract_migrations/"
                "unit-assessment-v1-predecessor-source/runner.py"
            )
            snapshot.write_text("tampered", encoding="utf-8")
            contract = json.loads((run / "inputs/contract.json").read_text())
            with self.assertRaisesRegex(RuntimeError, "source snapshot changed"):
                runner._validate_migration_history(run, contract)

    def test_single_start_exact_qp_migration_reuses_assessment_and_extends_chain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            predecessor, successor, authorization, assessment, ridge = (
                optimization_migration_fixture(run)
            )
            migrated = json.loads((run / "inputs/contract.json").read_text())
            self.assertEqual(migrated["runner_schema"], 5)
            self.assertEqual(
                migrated["optimization_protocol"], runner.OPTIMIZATION_PROTOCOL,
            )
            self.assertEqual(
                [entry["migration_id"] for entry in migrated["contract_migrations"]],
                [
                    "unit-generation-replacement-v1",
                    "unit-assessment-v1",
                    "unit-optimization-v1",
                ],
            )
            runner._validate_migration_history(run, migrated)
            with patch.object(
                runner, "SINGLE_START_EXACT_QP_MIGRATION", authorization,
            ):
                self.assertTrue(runner._checkpoint_source_is_authorized(
                    run,
                    stage="assessment",
                    checkpoint=assessment,
                    observed_source_id=predecessor["source_digest"],
                    current_source_id=successor["source_digest"],
                ))
                reused_gate = runner.load_assessment_checkpoint(
                    run,
                    source_id=successor["source_digest"],
                    input_id="assessment-input",
                )
                self.assertIsNotNone(reused_gate)
                self.assertFalse(reused_gate["passed"])
                ridge_source = json.loads(ridge.read_text())["source_digest"]
                self.assertTrue(runner._checkpoint_source_is_authorized(
                    run,
                    stage="ridge",
                    checkpoint=ridge,
                    observed_source_id=ridge_source,
                    current_source_id=successor["source_digest"],
                ))

    def test_casewise_schema_six_migration_is_preserved_in_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            successor, _, _, _ = casewise_migration_fixture(run)
            migrated = json.loads((run / "inputs/contract.json").read_text())
            self.assertEqual(migrated["runner_schema"], 6)
            self.assertEqual(
                migrated["validation_protocol"],
                "casewise_exact_common_reference_v1",
            )
            self.assertEqual(
                [entry["migration_id"] for entry in migrated["contract_migrations"]],
                [
                    "unit-generation-replacement-v1",
                    "unit-assessment-v1",
                    "unit-optimization-v1",
                    "unit-casewise-v1",
                ],
            )
            runner.establish_contract(run, successor)
            retained_manifest = json.loads((
                run
                / "inputs/contract_migrations/unit-casewise-v1-retained.json"
            ).read_text())
            self.assertEqual(
                retained_manifest["retired_unfinished_stage"],
                "validation/untouched_test_equivalence",
            )
            self.assertIn("optimization/nominal", retained_manifest["stages"])

    def test_schema_seven_poll_migration_requires_flag_and_loads_retained_route(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            _, _, prior_retention, case_directory = casewise_migration_fixture(run)
            predecessor = json.loads((run / "inputs/contract.json").read_text())
            snapshot_relative = (
                "inputs/contract_migrations/unit-poll-v2-predecessor-source"
            )
            snapshot = run / snapshot_relative
            for relative, content in {
                "stable.py": b"stable-v2",
                "runner.py": b"runner-v6",
                "replacement.py": b"replacement-v3",
            }.items():
                path = snapshot / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            successor_files = {
                **predecessor["source_files"],
                "runner.py": sha256(b"runner-v7").hexdigest(),
            }
            successor = {
                **{
                    key: value for key, value in predecessor.items()
                    if key != "contract_migrations"
                },
                "runner_schema": 7,
                "source_digest": runner.source_digest(successor_files),
                "source_files": successor_files,
                # This fixture exercises the historical schema-6 -> schema-7
                # transition even when the live runner has advanced to v3.
                "validation_protocol": "casewise_exact_common_reference_v2",
            }
            marker_set_digest = str(prior_retention["case_marker_set_digest"])
            retention = {
                **prior_retention,
                "predecessor_source_digest": predecessor["source_digest"],
            }
            authorization = runner.CasewiseComparisonMigrationAuthorization(
                migration_id="unit-poll-v2",
                run_id="unit-run",
                authorized_date="2026-08-24",
                reason="unit finite-poll refinement migration",
                predecessor_runner_schema=6,
                successor_runner_schema=7,
                predecessor_source_digest=predecessor["source_digest"],
                predecessor_contract_file_digest=runner.file_digest(
                    run / "inputs/contract.json"
                ),
                required_prior_migration_ids=(
                    "unit-generation-replacement-v1",
                    "unit-assessment-v1",
                    "unit-optimization-v1",
                    "unit-casewise-v1",
                ),
                predecessor_source_snapshot=snapshot_relative,
                allowed_changed_source_files=frozenset({"runner.py"}),
                required_changed_source_files=frozenset({"runner.py"}),
                required_artifact_digests={},
                expected_case_marker_set_digest=marker_set_digest,
                retired_casewise_snapshot=None,
            )
            with (
                patch.object(
                    runner,
                    "CONVERGENCE_POLL_REFINEMENT_MIGRATION",
                    authorization,
                ),
                patch.object(
                    runner,
                    "_validate_retained_casewise_comparison_checkpoints",
                    return_value=retention,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "explicit authorized migration"
                ):
                    runner.establish_contract(run, successor)
                runner.establish_contract(
                    run,
                    successor,
                    authorize_convergence_poll_refinement_migration=True,
                )

                migrated = json.loads((run / "inputs/contract.json").read_text())
                self.assertEqual(migrated["runner_schema"], 7)
                self.assertEqual(
                    migrated["validation_protocol"],
                    "casewise_exact_common_reference_v2",
                )
                self.assertEqual(
                    [entry["migration_id"] for entry in migrated["contract_migrations"]],
                    [
                        "unit-generation-replacement-v1",
                        "unit-assessment-v1",
                        "unit-optimization-v1",
                        "unit-casewise-v1",
                        "unit-poll-v2",
                    ],
                )

                sentinel = object()
                starts = np.asarray(runner.EXACT_QP_CENTER_START).reshape(1, 7)
                with patch.object(
                    runner, "_load_complete_route", return_value=sentinel,
                ) as load, patch.object(runner, "RUNNER_SCHEMA", 7):
                    restored = runner._load_retained_complete_route(
                        case_directory,
                        route="surrogate",
                        route_protocol=runner.EXACT_QP_SINGLE_START_PROTOCOL,
                        starts=starts,
                    )
                self.assertIs(restored, sentinel)
                self.assertEqual(
                    load.call_args.kwargs["route_contract"], "retained-route"
                )
                np.testing.assert_array_equal(load.call_args.kwargs["starts"], starts)

    def test_single_start_migration_requires_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            _, successor, authorization, _, _ = optimization_migration_fixture(run)
            # Reconstruct the predecessor bytes from the third migration archive.
            history = json.loads((run / "inputs/contract.json").read_text())[
                "contract_migrations"
            ]
            predecessor_archive = run / history[-1]["predecessor_contract"]
            (run / "inputs/contract.json").write_bytes(predecessor_archive.read_bytes())
            with patch.object(
                runner, "SINGLE_START_EXACT_QP_MIGRATION", authorization,
            ), self.assertRaisesRegex(RuntimeError, "explicit authorized migration"):
                runner.establish_contract(run, successor)

    def test_migration_history_rejects_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            assessment_migration_fixture(run)
            contract = json.loads((run / "inputs/contract.json").read_text())
            contract["contract_migrations"].reverse()
            with self.assertRaisesRegex(RuntimeError, "migration chain"):
                runner._validate_migration_history(run, contract)

    def test_atomic_helpers_publish_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner.atomic_json(root / "item.json", {"answer": np.int64(42)})
            runner.atomic_npz(root / "item.npz", values=np.arange(3))
            runner.atomic_dataframe(root / "item.csv", pd.DataFrame({"x": [1, 2]}))
            self.assertEqual(json.loads((root / "item.json").read_text()), {"answer": 42})
            with np.load(root / "item.npz", allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["values"], np.arange(3))
            self.assertEqual(pd.read_csv(root / "item.csv")["x"].tolist(), [1, 2])
            self.assertFalse(any(path.suffix == ".tmp" for path in root.iterdir()))

    def test_generation_reuses_validated_block_manifests(self) -> None:
        profile = tiny_profile()
        design = tiny_design(profile)
        source_files = {"unit-test": "bound-source"}

        def generator(
            decisions, influents, supplied_profile, output, *, block=None,
        ):
            return publish_mock_replacement_block(
                output, decisions, influents, supplied_profile,
            )

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(
                    runner, "generate_mechanistic_block_with_replacements",
                    side_effect=generator,
                ) as mocked, \
                    patch.object(runner, "assert_source_unchanged"):
                first = runner.run_generation(
                    run, design, profile=profile, source_files=source_files,
                )
                self.assertEqual(mocked.call_count, 2)
                second = runner.run_generation(
                    run, design, profile=profile, source_files=source_files,
                )
                self.assertEqual(mocked.call_count, 2)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            summary = pd.read_csv(run / "metrics/mechanistic_generation_summary.csv")
            self.assertTrue(summary["reused_complete_checkpoint"].all())
            self.assertTrue((run / "datasets/effective_design.npz").is_file())

    def test_generation_reuses_explicitly_carried_forward_source(self) -> None:
        profile = tiny_profile()
        design = tiny_design(profile)
        old_sources = {"unit-test": "old-source"}
        new_sources = {"unit-test": "new-source"}

        def generator(
            decisions, influents, supplied_profile, output, *, block=None,
        ):
            return publish_mock_replacement_block(
                output, decisions, influents, supplied_profile,
            )

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(
                runner, "generate_mechanistic_block_with_replacements",
                side_effect=generator,
            ) as mocked, patch.object(runner, "assert_source_unchanged"):
                first = runner.run_generation(
                    run, design, profile=profile, source_files=old_sources,
                )
                preserved_paths = (
                    run / "datasets/development/block_complete.json",
                    run / "datasets/test/block_complete.json",
                    run / "datasets/effective_design.npz",
                    run / "datasets/effective_design_manifest.json",
                    run / "metrics/mechanistic_generation_summary.csv",
                )
                before = {
                    path: runner.file_digest(path) for path in preserved_paths
                }
                with patch.object(
                    runner, "_checkpoint_source_is_authorized", return_value=True,
                ):
                    second = runner.run_generation(
                        run, design, profile=profile, source_files=new_sources,
                    )
            self.assertEqual(mocked.call_count, 2)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            self.assertEqual(
                before,
                {path: runner.file_digest(path) for path in preserved_paths},
            )

    def test_generation_gate_rejects_failed_row_before_publishing_manifest(self) -> None:
        profile = tiny_profile()
        design = tiny_design(profile)

        def generator(
            decisions, influents, supplied_profile, output, *, block=None,
        ):
            return publish_mock_replacement_block(
                output, decisions, influents, supplied_profile, accepted=False,
            )

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(
                    runner, "generate_mechanistic_block_with_replacements",
                    side_effect=generator,
                ), \
                    patch.object(runner, "assert_source_unchanged"):
                with self.assertRaisesRegex(RuntimeError, "unaccepted"):
                    runner.run_generation(
                        run, design, profile=profile,
                        source_files={"unit-test": "bound-source"},
                    )
            self.assertFalse(
                (run / "datasets/development/block_complete.json").exists()
            )

    def test_replacement_inputs_become_effective_design_without_mutating_base(self) -> None:
        profile = tiny_profile()
        design = tiny_design(profile)
        base_development = design["development_decisions"].copy()

        def generator(
            decisions, influents, supplied_profile, output, *, block=None,
        ):
            accepted_decisions = np.asarray(decisions).copy()
            if block == "development":
                accepted_decisions[1, 0] = runner.DECISION_LOWER[0]
            return publish_mock_replacement_block(
                output, accepted_decisions, influents, supplied_profile,
            )

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(
                    runner, "generate_mechanistic_block_with_replacements",
                    side_effect=generator,
                ), patch.object(runner, "assert_source_unchanged"):
                result = runner.run_generation(
                    run, design, profile=profile,
                    source_files={"unit-test": "bound-source"},
                )
            np.testing.assert_array_equal(
                design["development_decisions"], base_development,
            )
            self.assertEqual(
                result.design["development_decisions"][1, 0],
                runner.DECISION_LOWER[0],
            )
            with np.load(run / "datasets/effective_design.npz") as stored:
                np.testing.assert_array_equal(
                    stored["development_decisions"],
                    result.design["development_decisions"],
                )

    def test_admission_gate_is_development_only_and_tracks_holdout(self) -> None:
        gate = runner.evaluate_admission_gate(
            assessment_fixture(), correction_limit=0.4,
            trust_limits=TRUST_LIMITS,
            development_oof_projection_accepted=np.ones(5, dtype=bool),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=0.5,
            test_count=2,
        )
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["physical_audit_maxima"]["raw"]["passed"])
        self.assertTrue(gate["physical_audit_maxima"]["projected"]["passed"])
        self.assertTrue(gate["physical_audit_maxima"]["mechanistic"]["passed"])
        self.assertEqual(gate["admission_gate_scope"], "development_only")
        self.assertFalse(gate["post_selection_holdout_checks_are_admission_gates"])

        holdout_failure = assessment_fixture()
        holdout_failure.qp_diagnostics.loc[:, "accepted"] = False
        holdout_failure.feasibility.loc[:, "bound_passed"] = False
        holdout_failure.violations.loc[
            holdout_failure.violations["method"].isin(["projected", "mechanistic"]),
            "mass_conservation_violation_max",
        ] = 2.0
        descriptive = runner.evaluate_admission_gate(
            holdout_failure, correction_limit=0.4,
            trust_limits=TRUST_LIMITS,
            development_oof_projection_accepted=np.ones(5, dtype=bool),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=0.5,
            test_count=2,
        )
        self.assertTrue(descriptive["passed"])
        self.assertFalse(descriptive["all_projection_qp_audits_passed"])
        self.assertFalse(descriptive["all_finite_distance_bounds_passed"])
        self.assertFalse(descriptive["projected_physical_audits_passed"])
        self.assertFalse(descriptive["mechanistic_physical_audits_passed"])

        failed = runner.evaluate_admission_gate(
            assessment_fixture(), correction_limit=0.51,
            trust_limits={**TRUST_LIMITS, "correction": 0.51},
            development_oof_projection_accepted=np.ones(5, dtype=bool),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=0.5,
            test_count=2,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["execution_policy"], "advisory_continue")
        self.assertTrue(failed["optimization_permitted"])
        self.assertEqual(
            failed["failure_action"],
            "record advisory failure and continue without refitting",
        )
        self.assertTrue(runner.assessment_gate_allows_optimization(False))

    def test_development_oof_projection_rejection_is_advisory(self) -> None:
        gate = runner.evaluate_admission_gate(
            assessment_fixture(), correction_limit=0.4,
            trust_limits=TRUST_LIMITS,
            development_oof_projection_accepted=np.array(
                [True, False, True], dtype=bool,
            ),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=0.5,
            test_count=2,
        )
        self.assertFalse(
            gate["all_development_oof_projection_qp_audits_passed"]
        )
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["optimization_permitted"])

    def test_inventory_coordinate_has_its_own_development_oof_gate(self) -> None:
        gate = runner.evaluate_admission_gate(
            assessment_fixture(),
            correction_limit=0.4,
            trust_limits=TRUST_LIMITS,
            development_oof_projection_accepted=np.ones(5, dtype=bool),
            development_oof_complete_nrmse=0.5,
            development_oof_inventory_nrmse=1.01,
            test_count=2,
        )
        self.assertFalse(
            gate["development_oof_clarifier_inventory_nrmse_below_one"]
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["post_selection_holdout_is_confirmatory"])

    def test_assessment_persists_development_projection_acceptance(self) -> None:
        profile = tiny_profile()
        design = tiny_design(profile)
        response_count = profile.surrogate_response_count
        mechanistic_response_count = profile.mechanistic_response_count
        model = SimpleNamespace(
            response_count=response_count,
            response_scale=np.ones(response_count),
            feature_map=SimpleNamespace(
                transform=lambda decisions, influents: np.ones((len(decisions), 1)),
            ),
        )
        callbacks = SimpleNamespace(
            split_rows=lambda theta, raw, projected, influent: np.array([3.0, 4.0]),
            reactor_rows=lambda theta, raw, projected, influent: np.array([6.0, 8.0]),
        )
        accepted = np.array([True, False, True, True, True], dtype=bool)
        trust = SimpleNamespace(
            correction_limit=0.4,
            split_limit=1.0,
            reactor_limit=1.0,
            callbacks=callbacks,
            development_values=np.zeros((profile.development_count, 3)),
            out_of_fold_projected=np.zeros((profile.development_count, response_count)),
            out_of_fold_projection_accepted=accepted,
            split_scale=np.ones(1),
        )
        surrogate_assets = SimpleNamespace(
            leverage_precision=np.ones((1, 1)),
            trust_thresholds=SimpleNamespace(regularized_leverage=1.0),
            trust_callbacks=callbacks,
        )
        assessment = assessment_fixture()
        assessment = AssessmentResult(
            metrics=assessment.metrics,
            violations=assessment.violations,
            qp_diagnostics=assessment.qp_diagnostics,
            feasibility=assessment.feasibility,
            raw=np.zeros((profile.test_count, response_count)),
            projected=np.zeros((profile.test_count, response_count)),
            projected_targets=np.zeros((profile.test_count, response_count)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(
                runner, "fit_or_resume_ridge",
                return_value=(
                    model,
                    np.zeros((profile.development_count, response_count)),
                    "ridge-input",
                ),
            ), patch.object(
                runner, "fit_direct_assets", return_value=SimpleNamespace(),
            ), patch.object(
                runner, "calibrate_trust_diagnostics", return_value=trust,
            ), patch.object(
                runner, "build_surrogate_assets", return_value=surrogate_assets,
            ), patch.object(
                runner, "assess_raw_projected_mechanistic",
                return_value=assessment,
            ), patch.object(
                runner, "assert_source_unchanged",
            ), patch.object(
                runner, "_artifact_hashes", return_value={},
            ) as artifact_hashes:
                result = runner.run_assessment(
                    run,
                    design,
                    np.zeros((profile.development_count, mechanistic_response_count)),
                    np.zeros((profile.test_count, mechanistic_response_count)),
                    profile=profile,
                    source_files={"unit-test": "bound-source"},
                )
            trust_frame = pd.read_csv(run / "metrics/trust_development_oof.csv")
            np.testing.assert_array_equal(
                trust_frame["projection_qp_accepted"].to_numpy(dtype=bool),
                accepted,
            )
            with np.load(run / "models/trust_calibration.npz") as stored:
                np.testing.assert_array_equal(
                    stored["out_of_fold_projection_accepted"], accepted,
                )
            with np.load(
                run / "datasets/development/surrogate_responses_inventory_v1.npz"
            ) as stored:
                self.assertEqual(
                    stored["responses"].shape,
                    (profile.development_count, profile.surrogate_response_count),
                )
                self.assertEqual(str(stored["schema"].item()), runner.RESPONSE_SCHEMA)
            holdout_trust = pd.read_csv(
                run / "metrics/trust_post_selection_holdout.csv"
            )
            self.assertEqual(
                list(holdout_trust.columns),
                [
                    "row", "correction", "regularized_leverage",
                    "particulate_split", "reactor_residual",
                ],
            )
            np.testing.assert_allclose(
                holdout_trust[
                    [
                        "correction", "regularized_leverage",
                        "particulate_split", "reactor_residual",
                    ]
                ].to_numpy(),
                np.tile(
                    [0.0, 1.0, np.sqrt(12.5), np.sqrt(50.0)],
                    (profile.test_count, 1),
                ),
            )
            published_paths = artifact_hashes.call_args.args[1]
            self.assertIn(
                run / "metrics/trust_post_selection_holdout.csv",
                published_paths,
            )
            self.assertFalse(result.passed)
            self.assertFalse(
                result.gate["all_development_oof_projection_qp_audits_passed"]
            )
            self.assertTrue(result.gate["optimization_permitted"])

    def test_main_continues_after_complete_advisory_gate_failure(self) -> None:
        generation = runner.GenerationResult(
            design={}, development_targets=np.empty((0, 0)),
            test_targets=np.empty((0, 0)),
        )
        analysis = runner.AnalysisBundle(
            passed=False, model=None, direct_assets=None, surrogate_assets=None,
            assessment=None, gate={"passed": False},
        )
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(runner, "resolve_run_directory", return_value=run), \
                    patch.object(runner, "source_file_digests", return_value={}), \
                    patch.object(runner, "establish_contract"), \
                    patch.object(runner, "load_or_create_design", return_value={}), \
                    patch.object(runner, "assert_source_unchanged"), \
                    patch.object(runner, "run_generation", return_value=generation), \
                    patch.object(runner, "_assessment_binding", return_value="input"), \
                    patch.object(runner, "load_assessment_checkpoint", return_value=None), \
                    patch.object(runner, "run_assessment", return_value=analysis), \
                    patch.object(
                        runner, "run_optimization_stage", return_value=True,
                    ) as optimization:
                runner.main("article_full_5000_unit", "complete")
            optimization.assert_called_once()
            state = json.loads((run / "run_state.json").read_text())
            self.assertEqual(state["status"], "complete_with_validation_failures")
            self.assertFalse(state["admission_gate_passed"])
            self.assertTrue(state["optimization_validation_passed"])
            self.assertFalse(state["scientific_validation_passed"])
            self.assertEqual(
                state["assessment_gate_execution_policy"], "advisory_continue",
            )

    def test_main_does_not_mask_hard_assessment_failure(self) -> None:
        generation = runner.GenerationResult(
            design={}, development_targets=np.empty((0, 0)),
            test_targets=np.empty((0, 0)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(runner, "resolve_run_directory", return_value=run), \
                    patch.object(runner, "source_file_digests", return_value={}), \
                    patch.object(runner, "establish_contract"), \
                    patch.object(runner, "load_or_create_design", return_value={}), \
                    patch.object(runner, "assert_source_unchanged"), \
                    patch.object(runner, "run_generation", return_value=generation), \
                    patch.object(runner, "_assessment_binding", return_value="input"), \
                    patch.object(runner, "load_assessment_checkpoint", return_value=None), \
                    patch.object(
                        runner, "run_assessment", side_effect=RuntimeError("numerical"),
                    ), patch.object(runner, "run_optimization_stage") as optimization:
                with self.assertRaisesRegex(RuntimeError, "numerical"):
                    runner.main("article_full_5000_unit", "complete")
            optimization.assert_not_called()
            state = json.loads((run / "run_state.json").read_text())
            self.assertEqual(state["stage"], "assessment")
            self.assertEqual(state["status"], "failed")

    def test_ridge_bundle_round_trip_and_tamper_detection(self) -> None:
        rows, responses = 10, 2
        feature_map = QuadraticFeatureMap(
            decision_center=np.zeros(1), decision_scale=np.ones(1),
            influent_center=np.zeros(1), influent_scale=np.ones(1),
            term_center=np.zeros(5), term_scale=np.ones(5),
        )
        diagnostics = LeastSquaresDiagnostics(
            sample_count=rows, feature_count=6, response_count=responses,
            rank_tolerance=1e-12, smallest_singular_value=1.0,
            largest_singular_value=2.0, condition_number=2.0,
            optimality_residual=1e-12, coefficient_agreement=1e-12,
            acceptance_threshold=1e-10,
        )
        model = QuadraticSurrogate(
            feature_map=feature_map, response_center=np.zeros(responses),
            response_scale=np.ones(responses), coefficients=np.zeros((responses, 6)),
            diagnostics=diagnostics, ridge_penalty=float(runner.RIDGE_GRID[-1]),
        )
        scores = pd.DataFrame([
            {
                "fold": fold, "gamma": gamma, "raw_nrmse": 1.0,
                "selected": bool(gamma == runner.RIDGE_GRID[-1]),
            }
            for fold in range(1, 6) for gamma in runner.RIDGE_GRID
        ])
        membership = np.tile(np.arange(1, 6), 2)
        result = SimpleNamespace(
            model=model, scores=scores, fold_membership=membership,
            out_of_fold_raw=np.zeros((rows, responses)), elapsed_seconds=1.0,
        )
        decisions = np.zeros((rows, 1))
        influents = np.zeros((rows, 1))
        targets = np.zeros((rows, responses))
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            old_sources = {"unit-test": "old-source"}
            new_sources = {"unit-test": "new-source"}
            old_source_id = runner.source_digest(old_sources)
            input_id = runner._ridge_input_digest(decisions, influents, targets)
            runner.save_ridge(
                run, result, input_id=input_id, source_id=old_source_id,
            )
            preserved_paths = (
                run / "models/ridge_complete.json",
                run / "models/ridge_surrogate.npz",
                run / "metrics/ridge_cross_validation.csv",
                run / "metrics/ridge_fold_membership.csv",
            )
            before = {path: runner.file_digest(path) for path in preserved_paths}
            restored = runner._load_ridge(
                run, decisions=decisions, influents=influents, targets=targets,
                input_id=input_id, source_id=old_source_id,
            )
            self.assertIsNotNone(restored)
            np.testing.assert_array_equal(restored[1], result.out_of_fold_raw)
            with patch.object(
                runner, "_checkpoint_source_is_authorized", return_value=True,
            ), patch.object(runner, "cross_validate_ridge") as refit:
                carried_model, carried_oof, carried_input = runner.fit_or_resume_ridge(
                    run, decisions, influents, targets, source_files=new_sources,
                )
            refit.assert_not_called()
            self.assertEqual(carried_input, input_id)
            self.assertEqual(carried_model.ridge_penalty, model.ridge_penalty)
            np.testing.assert_array_equal(carried_oof, result.out_of_fold_raw)
            self.assertEqual(
                before,
                {path: runner.file_digest(path) for path in preserved_paths},
            )
            with (run / "metrics/ridge_cross_validation.csv").open("a") as stream:
                stream.write("tampered")
            self.assertIsNone(runner._load_ridge(
                run, decisions=decisions, influents=influents, targets=targets,
                input_id=input_id, source_id=old_source_id,
            ))

    def test_optimization_hook_rejects_nonarticle_profile(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unauthorized dataset total"):
            runner.run_optimization_stage(
                run=Path("unused"), profile=tiny_profile(), design={},
                development_targets=np.empty((0, 0)), test_targets=np.empty((0, 0)),
                analysis=None, source_files={},
            )


if __name__ == "__main__":
    unittest.main()
