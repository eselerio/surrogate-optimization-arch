import json
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_article_v3_5000 as runner


ROOT = Path(__file__).resolve().parents[1]


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _retained_stage(source: Path) -> dict[str, object]:
    checkpoint = source / "models/ridge_complete.json"
    artifact = source / "models/ridge_surrogate.npz"
    runner.atomic_json(checkpoint, {"stage": "ridge", "complete": True})
    _write_bytes(artifact, b"retained fitted model")
    return {
        "schema": 3,
        "predecessor_source_digest": "unit-source",
        "effective_design_digest": "unit-design",
        "ridge_input_digest": "unit-ridge",
        "assessment_input_digest": "unit-assessment",
        "case_marker_set_digest": "unit-cases",
        "stages": {
            "ridge": {
                "checkpoint": checkpoint.relative_to(source).as_posix(),
                "checkpoint_sha256": runner.file_digest(checkpoint),
                "artifact_source_digest": "unit-source",
                "artifacts": {
                    artifact.relative_to(source).as_posix(): runner.file_digest(artifact),
                },
            },
        },
    }


class ArticleV3RunForkTests(unittest.TestCase):
    def test_schema_twelve_defaults_name_a_new_folder_and_no_minimum_srt_protocol(self) -> None:
        self.assertEqual(runner.LEGACY_RUN_ID, "article_full_5000_001")
        self.assertEqual(runner.DEFAULT_RUN_ID, "article_full_5000_002")
        self.assertEqual(runner.RUNNER_SCHEMA, 12)
        self.assertEqual(
            runner.COMPARISON_PROTOCOL,
            "casewise_exact_common_reference_no_minimum_srt_v4",
        )

    def test_reduced_response_fork_copies_generation_but_not_old_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            old_files, new_files = {"unit.py": "old"}, {"unit.py": "new"}
            old_digest = runner.source_digest(old_files)
            source_contract = {
                "runner_schema": 9,
                "source_digest": old_digest,
                "source_files": old_files,
                "contract_migrations": [],
            }
            successor_contract = {
                "runner_schema": 10,
                "run_id": target.name,
                "source_digest": runner.source_digest(new_files),
                "source_files": new_files,
                "response_schema": {"name": runner.RESPONSE_SCHEMA},
            }
            for block in ("development", "test"):
                artifact = source / "datasets" / block / "mechanistic_accepted_v3.npz"
                runner.atomic_npz(artifact, targets=np.zeros((1, 170)))
                runner.atomic_npz(
                    source / "datasets" / block / "rows" / "row_000000.npz",
                    target=np.zeros(170),
                )
                runner.atomic_npz(
                    source / "datasets" / block
                    / "surrogate_responses_inventory_v1.npz",
                    responses=np.zeros((1, 161)),
                )
                runner.atomic_json(
                    source / "datasets" / block / "block_complete.json",
                    {
                        "source_digest": old_digest,
                        "artifacts": {
                            artifact.relative_to(source).as_posix(): runner.file_digest(artifact),
                        },
                    },
                )
            design_path = source / "datasets" / "design.npz"
            partition_path = source / "inputs" / "frozen_accepted_partition.json"
            source_design_path = (
                source / "inputs" / "frozen_accepted" / "source_design_50000.npz"
            )
            runner.atomic_npz(design_path, values=np.zeros(1))
            runner.atomic_json(partition_path, {"schema": 1})
            runner.atomic_npz(source_design_path, values=np.zeros(1))
            runner.atomic_json(
                source / "datasets" / "frozen_accepted_complete.json",
                {
                    "source_digest": old_digest,
                    "artifacts": {
                        path.relative_to(source).as_posix(): runner.file_digest(path)
                        for path in (design_path, partition_path, source_design_path)
                    },
                },
            )
            runner.atomic_json(source / "inputs" / "contract.json", source_contract)
            runner.atomic_json(source / "inputs" / "generator_records.json", {"seed": 1})
            _write_bytes(source / "models" / "ridge_surrogate.npz", b"superseded")
            _write_bytes(source / "optimization" / "nominal" / "surrogate.npz", b"superseded")
            _write_bytes(
                source / "datasets" / "assessment" / "post_selection_holdout.npz",
                b"superseded",
            )

            runner._initialize_reduced_response_fork(
                target,
                source_run=source,
                source_contract=source_contract,
                successor_contract=successor_contract,
            )

            self.assertTrue(
                (target / "datasets/development/mechanistic_accepted_v3.npz").is_file()
            )
            self.assertTrue(
                (target / "datasets/development/rows/row_000000.npz").is_file()
            )
            self.assertTrue(
                (target / "inputs/frozen_accepted/source_design_50000.npz").is_file()
            )
            self.assertFalse((
                target / "datasets/development/surrogate_responses_inventory_v1.npz"
            ).exists())
            self.assertFalse((
                target / "datasets/test/surrogate_responses_inventory_v1.npz"
            ).exists())
            self.assertFalse((
                target / "datasets/assessment/post_selection_holdout.npz"
            ).exists())
            self.assertFalse((target / "models/ridge_surrogate.npz").exists())
            self.assertFalse((target / "optimization/nominal/surrogate.npz").exists())
            migrated = json.loads((target / "inputs/contract.json").read_text())
            record = json.loads((
                target / migrated["contract_migrations"][-1]["record"]
            ).read_text())
            self.assertEqual(
                record["run_fork"]["recomputed_scope"],
                "response_transform_fit_assessment_optimization_replay_timing_reporting",
            )

    def test_copy_reusable_files_is_hash_verified_and_excludes_new_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            _write_bytes(source / "datasets/design.npz", b"5000-row design")
            _write_bytes(
                source / "datasets/development/accepted_inputs.npz",
                b"4000 accepted development rows",
            )
            _write_bytes(
                source / "inputs/contract_migrations/prior.json",
                b"prior immutable migration",
            )
            runner.atomic_json(
                source / "inputs/generator_records.json", {"seed": 100_042},
            )
            retention = _retained_stage(source)
            _write_bytes(
                source / "metrics/case_common_reference_comparison.csv",
                b"stale comparison must be recomputed",
            )
            _write_bytes(
                source / "report/tables/article_table.csv",
                b"stale report must be recomputed",
            )
            _write_bytes(
                source / "optimization/nominal/surrogate_local_convergence.json",
                b"stale certificate must be recomputed",
            )

            copied = runner._copy_reusable_files(source, target, retention)

            expected = {
                "datasets/design.npz",
                "datasets/development/accepted_inputs.npz",
                "inputs/contract_migrations/prior.json",
                "inputs/generator_records.json",
                "models/ridge_complete.json",
                "models/ridge_surrogate.npz",
            }
            self.assertEqual(set(copied), expected)
            for relative, digest in copied.items():
                self.assertEqual(runner.file_digest(target / relative), digest)
            self.assertFalse(
                (target / "metrics/case_common_reference_comparison.csv").exists()
            )

            self.assertFalse((target / "report/tables/article_table.csv").exists())
            self.assertFalse(
                (
                    target
                    / "optimization/nominal/surrogate_local_convergence.json"
                ).exists()
            )

            # The rerun owns independent bytes; later source changes do not
            # mutate the retained 5,000-row inputs in the new folder.
            _write_bytes(source / "datasets/design.npz", b"changed source")
            self.assertEqual(
                (target / "datasets/design.npz").read_bytes(), b"5000-row design"
            )

    def test_marker_closure_copies_content_addressed_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            relative_archive = (
                "inputs/contract_migrations/unit-predecessor-casewise"
            )
            source_archive = source / relative_archive
            target_archive = target / relative_archive
            marker_relative = (
                "optimization/nominal/casewise_comparison_complete.json"
            )
            artifact_relative = "optimization/nominal/reference_payload.npz"
            artifact = source / artifact_relative
            _write_bytes(artifact, b"historical exact-reference payload")
            digest = runner.file_digest(artifact)
            runner.atomic_json(source_archive / marker_relative, {
                "stage": "casewise_comparison",
                "artifacts": {artifact_relative: digest},
            })
            target_marker = target_archive / marker_relative
            target_marker.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_archive / marker_relative, target_marker)

            copied = runner._copy_artifact_archive_marker_closure(
                source, target, relative_archive,
            )

            blob_relative = f"{relative_archive}/_artifact_blobs/{digest}"
            self.assertEqual(copied, {blob_relative: digest})
            blob = target / blob_relative
            self.assertEqual(blob.read_bytes(), b"historical exact-reference payload")
            runner._validate_artifact_archive_marker_closure(target_archive)

            # Closure is self-contained: validation no longer depends on the
            # old run's live output path once the hash-addressed blob exists.
            artifact.unlink()
            runner._validate_artifact_archive_marker_closure(target_archive)
            _write_bytes(blob, b"tampered closure blob")
            with self.assertRaisesRegex(RuntimeError, "unresolved completion-marker"):
                runner._validate_artifact_archive_marker_closure(target_archive)

            runner.atomic_json(source_archive / marker_relative, {
                "stage": "casewise_comparison",
                "artifacts": {artifact_relative: "../not-a-sha256"},
            })
            with self.assertRaisesRegex(RuntimeError, "invalid SHA-256"):
                runner._copy_artifact_archive_marker_closure(
                    source, target, relative_archive,
                )

            runner.atomic_json(target_marker, {
                "stage": "casewise_comparison",
                "artifacts": {"../escaped.npz": digest},
            })
            with self.assertRaisesRegex(RuntimeError, "unsafe artifact reference"):
                runner._validate_artifact_archive_marker_closure(target_archive)

    def test_marker_closure_follows_completion_markers_stored_as_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            relative_archive = (
                "inputs/contract_migrations/unit-predecessor-casewise"
            )
            source_archive = source / relative_archive
            target_archive = target / relative_archive
            marker_relative = (
                "optimization/nominal/casewise_comparison_complete.json"
            )
            nested_relative = (
                "optimization/nominal/surrogate_local_convergence_complete.json"
            )
            live_directory = source / "optimization/nominal"
            certificate = live_directory / "surrogate_certified.npz"
            _write_bytes(certificate, b"historical convergence certificate")
            certificate_digest = runner.file_digest(certificate)
            nested_marker = live_directory / "surrogate_local_convergence_complete.json"
            runner.atomic_json(nested_marker, {
                "stage": "surrogate_local_convergence",
                "artifacts": {
                    "surrogate_certified.npz": certificate_digest,
                },
            })
            nested_digest = runner.file_digest(nested_marker)
            runner.atomic_json(source_archive / marker_relative, {
                "stage": "casewise_comparison",
                "artifacts": {nested_relative: nested_digest},
            })
            target_marker = target_archive / marker_relative
            target_marker.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_archive / marker_relative, target_marker)

            copied = runner._copy_artifact_archive_marker_closure(
                source, target, relative_archive,
            )

            nested_blob = target_archive / "_artifact_blobs" / nested_digest
            certificate_blob = (
                target_archive / "_artifact_blobs" / certificate_digest
            )
            self.assertEqual(
                set(copied),
                {
                    nested_blob.relative_to(target).as_posix(),
                    certificate_blob.relative_to(target).as_posix(),
                },
            )
            runner._validate_artifact_archive_marker_closure(target_archive)

            nested_marker.unlink()
            certificate.unlink()
            runner._validate_artifact_archive_marker_closure(target_archive)
            _write_bytes(certificate_blob, b"tampered nested payload")
            with self.assertRaisesRegex(RuntimeError, "unresolved completion-marker"):
                runner._validate_artifact_archive_marker_closure(target_archive)

            runner.atomic_json(source_archive / marker_relative, {
                "stage": "casewise_comparison",
                "artifacts": {"../escaped_complete.json": nested_digest},
            })
            with self.assertRaisesRegex(RuntimeError, "unsafe artifact reference"):
                runner._copy_artifact_archive_marker_closure(
                    source, target, relative_archive,
                )

    def test_marker_closure_checks_duplicate_markers_in_each_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "archive"
            payload = archive / "optimization/case_b/payload.npz"
            _write_bytes(payload, b"case-local payload")
            digest = runner.file_digest(payload)
            marker_payload = {
                "stage": "duplicate_marker_fixture",
                "artifacts": {"payload.npz": digest},
            }
            runner.atomic_json(
                archive / "optimization/case_a/case_complete.json",
                marker_payload,
            )
            runner.atomic_json(
                archive / "optimization/case_b/case_complete.json",
                marker_payload,
            )

            with self.assertRaisesRegex(RuntimeError, "unresolved completion-marker"):
                runner._validate_artifact_archive_marker_closure(archive)

    def test_pinned_schema_seven_fork_is_restricted_to_default_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            source_id = "article_full_5000_unit_v2"
            source = results / source_id
            target = results / "article_full_5000_not_the_authorized_default"
            old_files = {"runner.py": "a" * 64}
            new_files = {"runner.py": "b" * 64}
            common = {
                "profile": {"name": "unit-5000"},
                "fixed_dataset_total": 5_000,
                "development_test_split": [4_000, 1_000],
                "python": "unit-python",
                "platform": "unit-platform",
                "runtime_versions": {"unit": "1"},
                "assessment_gate_execution_policy": (
                    runner.ASSESSMENT_GATE_EXECUTION_POLICY
                ),
                "optimization_protocol": runner.OPTIMIZATION_PROTOCOL,
                "preflight_artifacts_permitted": False,
                "full_run_admission_gate_bypass_permitted": False,
            }
            source_contract = {
                **common,
                "runner_schema": 7,
                "run_id": source_id,
                "source_digest": runner.source_digest(old_files),
                "source_files": old_files,
                "validation_protocol": "casewise_exact_common_reference_v2",
                "contract_migrations": [{"migration_id": "unit-prior"}],
            }
            successor_contract = {
                **common,
                "runner_schema": runner.RUNNER_SCHEMA,
                "run_id": target.name,
                "source_digest": runner.source_digest(new_files),
                "source_files": new_files,
                "validation_protocol": runner.COMPARISON_PROTOCOL,
            }
            runner.atomic_json(source / "inputs/contract.json", source_contract)
            authorization = replace(
                runner.POLL_LINESEARCH_FORK_MIGRATION,
                run_id=source_id,
                predecessor_source_digest=source_contract["source_digest"],
                predecessor_contract_file_digest=runner.file_digest(
                    source / "inputs/contract.json"
                ),
                allowed_changed_source_files=frozenset({"runner.py"}),
                required_changed_source_files=frozenset({"runner.py"}),
            )
            with (
                patch.object(
                    runner,
                    "resolve_run_directory",
                    side_effect=lambda run_id: results / run_id,
                ),
                patch.object(runner, "_validate_migration_history"),
                patch.object(
                    runner, "POLL_LINESEARCH_FORK_MIGRATION", authorization,
                ),
                self.assertRaisesRegex(RuntimeError, "neither the pinned v2 predecessor"),
            ):
                runner.initialize_reused_run(
                    target,
                    source_run_id=source_id,
                    successor_contract=successor_contract,
                )
            self.assertFalse(target.exists())

    def test_initialize_same_source_rerun_records_lineage_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            source_id = "article_full_5000_unit_source"
            target_id = "article_full_5000_unit_target"
            source = results / source_id
            target = results / target_id
            _write_bytes(source / "datasets/design.npz", b"5000-row design")
            _write_bytes(
                source / "inputs/contract_migrations/prior-record.json",
                b"prior record",
            )
            runner.atomic_json(
                source / "inputs/generator_records.json", {"seed": 100_042},
            )
            retention = _retained_stage(source)
            _write_bytes(
                source / "metrics/case_common_reference_comparison.csv",
                b"old result",
            )

            source_files = {"README.md": runner.file_digest(ROOT / "README.md")}
            common = {
                "runner_schema": runner.RUNNER_SCHEMA,
                "profile": {"name": "unit-5000"},
                "fixed_dataset_total": 5_000,
                "development_test_split": [4_000, 1_000],
                "source_digest": runner.source_digest(source_files),
                "source_files": source_files,
                "python": "unit-python",
                "platform": "unit-platform",
                "runtime_versions": {"unit": "1"},
                "assessment_gate_execution_policy": (
                    runner.ASSESSMENT_GATE_EXECUTION_POLICY
                ),
                "optimization_protocol": runner.OPTIMIZATION_PROTOCOL,
                "validation_protocol": runner.COMPARISON_PROTOCOL,
                "preflight_artifacts_permitted": False,
                "full_run_admission_gate_bypass_permitted": False,
            }
            source_contract = {
                **common,
                "run_id": source_id,
                "contract_migrations": [{"migration_id": "prior-unit"}],
            }
            successor_contract = {**common, "run_id": target_id}
            runner.atomic_json(source / "inputs/contract.json", source_contract)

            resolver = lambda run_id: results / run_id
            with (
                patch.object(runner, "resolve_run_directory", side_effect=resolver),
                patch.object(runner, "_validate_migration_history"),
                patch.object(
                    runner,
                    "_validate_retained_casewise_comparison_checkpoints",
                    return_value=retention,
                ),
            ):
                runner.initialize_reused_run(
                    target,
                    source_run_id=source_id,
                    successor_contract=successor_contract,
                )
                before = {
                    path.relative_to(target).as_posix(): runner.file_digest(path)
                    for path in target.rglob("*")
                    if path.is_file()
                }
                runner.initialize_reused_run(
                    target,
                    source_run_id=source_id,
                    successor_contract=successor_contract,
                )
                after = {
                    path.relative_to(target).as_posix(): runner.file_digest(path)
                    for path in target.rglob("*")
                    if path.is_file()
                }

            self.assertEqual(before, after)
            contract = json.loads((target / "inputs/contract.json").read_text())
            self.assertEqual(contract["run_id"], target_id)
            latest = contract["contract_migrations"][-1]
            record = json.loads((target / latest["record"]).read_text())
            self.assertEqual(
                record["run_fork"],
                {
                    "source_run_id": source_id,
                    "target_run_id": target_id,
                    "self_contained": True,
                    "recomputed_scope": (
                        "casewise_certification_reference_timing_reporting"
                    ),
                },
            )
            reuse_reference = record["reused_artifact_manifest"]
            reuse = json.loads((target / reuse_reference["path"]).read_text())
            self.assertEqual(reuse["copy_mode"], "independent_byte_copy")
            self.assertEqual(reuse["source_run_id"], source_id)
            self.assertEqual(reuse["target_run_id"], target_id)
            self.assertEqual(reuse["file_count"], len(reuse["files"]))
            self.assertEqual(
                reuse["file_set_digest"],
                runner._canonical_json_digest(reuse["files"]),
            )
            self.assertTrue((target / "datasets/design.npz").is_file())
            self.assertTrue((target / "models/ridge_surrogate.npz").is_file())
            self.assertFalse(
                (target / "metrics/case_common_reference_comparison.csv").exists()
            )

            # Idempotent initialization must still authenticate every copied
            # byte; an existing folder is not trustworthy merely because its
            # lineage journal remains intact.
            _write_bytes(target / "datasets/design.npz", b"tampered rerun data")
            with (
                patch.object(runner, "resolve_run_directory", side_effect=resolver),
                patch.object(runner, "_validate_migration_history"),
                self.assertRaisesRegex(RuntimeError, "reused artifact changed"),
            ):
                runner.initialize_reused_run(
                    target,
                    source_run_id=source_id,
                    successor_contract=successor_contract,
                )

    def test_reuse_manifest_detects_a_changed_copied_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            artifact = run / "datasets/design.npz"
            _write_bytes(artifact, b"original")
            files = {"datasets/design.npz": runner.file_digest(artifact)}
            manifest_path = (
                run / "inputs/contract_migrations/unit-reused-files.json"
            )
            runner.atomic_json(manifest_path, {
                "schema": 1,
                "file_set_digest": runner._canonical_json_digest(files),
                "files": files,
            })
            reference = {
                "path": manifest_path.relative_to(run).as_posix(),
                "sha256": runner.file_digest(manifest_path),
            }
            _write_bytes(artifact, b"tampered")
            with self.assertRaisesRegex(RuntimeError, "reused artifact changed"):
                runner._validate_fork_reuse_manifest(
                    run, reference, verify_files=True,
                )

    def test_main_forwards_explicit_reuse_source_before_work_starts(self) -> None:
        target = Path("unit-target")
        contract = {"run_id": "article_full_5000_002", "runner_schema": 8}
        with (
            patch.object(runner, "validate_authorized_profile"),
            patch.object(runner, "resolve_run_directory", return_value=target),
            patch.object(runner, "source_file_digests", return_value={"a": "b"}),
            patch.object(runner, "_build_contract", return_value=contract),
            patch.object(
                runner,
                "initialize_reused_run",
                side_effect=RuntimeError("stop after reuse initialization"),
            ) as initialize,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after reuse"):
                runner.main(
                    "article_full_5000_002",
                    "generation",
                    reuse_from_run_id="article_full_5000_001",
                )
        initialize.assert_called_once_with(
            target,
            source_run_id="article_full_5000_001",
            successor_contract=contract,
        )


if __name__ == "__main__":
    unittest.main()
