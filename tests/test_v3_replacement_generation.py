import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from closed_loop import v3_replacement_generation as replacement
from closed_loop.manuscript_v3 import StudyProfile, create_design
from closed_loop.model import N_COMPONENTS, N_STAGES


def _profile() -> StudyProfile:
    return StudyProfile(
        name="replacement_unit", development_count=3, test_count=2,
        robustness_count=1, layer_count=3,
        development_seed=101, test_seed=202, robustness_seed=303,
        parallel_workers=1, article_eligible=False,
        enforce_admission_gate=False,
    )


def _record(candidate: replacement._Candidate, accepted: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "candidate_round": candidate.round_index,
        "candidate_index": candidate.candidate_index,
        "candidate_ordinal": candidate.candidate_ordinal,
        "accepted": accepted,
        "attempt_status": "accepted" if accepted else "rejected",
        "error_type": "",
        "error_message": "",
        "elapsed_seconds": 1.0,
        "root_difference_inf": 1.0e-8,
        "branch_agreement": accepted,
        "branch_classification": "{}",
        "route_start_1": "test",
        "route_start_2": "test",
    }
    for name in (
        "minimum_state_start_1", "minimum_state_start_2",
        "state_negativity_start_1", "state_negativity_start_2",
        "rate_negativity_start_1", "rate_negativity_start_2",
        "mass_residual_start_1", "mass_residual_start_2",
        "largest_real_eigenvalue_start_1", "largest_real_eigenvalue_start_2",
        "stability_agreement_start_1", "stability_agreement_start_2",
        "feed_tss_start_1", "feed_tss_start_2",
        "external_solids_loss_start_1", "external_solids_loss_start_2",
    ):
        record[name] = 0.0
    return record


def _checkpoint(
    candidate: replacement._Candidate,
    profile: StudyProfile,
    contract_hash: str,
    *,
    accepted: bool,
    marker: float,
) -> None:
    state_size = N_STAGES * N_COMPONENTS + profile.layer_count
    replacement._write_attempt(
        candidate, contract_hash=contract_hash,
        target=np.full(profile.response_count, marker),
        first=np.full(state_size, marker), second=np.full(state_size, marker),
        record=_record(candidate, accepted),
    )


class ReplacementGenerationTests(unittest.TestCase):
    def test_rejection_reason_is_deterministic_and_retains_overlaps(self) -> None:
        record = replacement._annotate_rejection({
            "accepted": False, "attempt_status": "rejected",
            "mass_residual_start_1": 2.0e-8,
            "stability_agreement_start_1": 2.0e-6,
            "state_negativity_start_1": 2.0e-10,
            "feed_tss_start_1": 0.5,
            "root_difference_inf": 2.0e-6,
            "branch_agreement": False,
        })
        self.assertEqual(record["rejection_reason"], "mass_or_residual")
        self.assertEqual(
            record["rejection_reasons"],
            "mass_or_residual;stability;nonnegativity;domain;"
            "root_distance;branch_disagreement",
        )
        self.assertTrue(record["rejected_branch_disagreement"])

    def test_atomic_replace_retries_transient_windows_sharing_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.write_text("complete", encoding="utf-8")
            actual_replace = replacement.os.replace
            calls = 0

            def flaky_replace(first, second):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError(13, "sharing violation")
                return actual_replace(first, second)

            with patch.object(replacement.os, "replace", side_effect=flaky_replace), \
                    patch.object(replacement.time, "sleep") as sleeper:
                replacement._replace_with_retry(source, destination)
            self.assertEqual(calls, 3)
            self.assertEqual(sleeper.call_count, 2)
            self.assertEqual(destination.read_text(encoding="utf-8"), "complete")

    def test_continuation_is_row_major_open_and_replayable(self) -> None:
        first, state, draws = replacement._supplemental_coordinates(2, 987654321)
        second, replay_state, replay_draws = replacement._supplemental_coordinates(
            2, 987654321,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(draws, 54)
        self.assertEqual((state, draws), (replay_state, replay_draws))
        self.assertTrue(np.all(first > 0.0))
        self.assertTrue(np.all(first < 1.0))

    def test_reuses_base_attempts_and_fills_failed_slot_without_overwrite(self) -> None:
        profile = _profile()
        design = create_design(profile)
        decisions = design["development_decisions"]
        influents = design["development_influents"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "development"
            output.mkdir(parents=True)
            np.savez_compressed(output / "mechanistic_rows_v3.npz", legacy=[1.0])
            (output / "mechanistic_diagnostics.csv").write_text(
                "legacy,value\ntrue,1\n", encoding="utf-8",
            )
            legacy_hashes = {
                path: replacement._file_digest(path)
                for path in (
                    output / "mechanistic_rows_v3.npz",
                    output / "mechanistic_diagnostics.csv",
                )
            }
            base_contract = replacement._base_contract_hash(
                decisions, influents, profile,
            )
            base_candidates = [
                replacement._Candidate(
                    "development", 0, index, index, decisions[index],
                    influents[index], output / "rows" / f"row_{index:06d}.npz",
                )
                for index in range(3)
            ]
            for index, candidate in enumerate(base_candidates):
                _checkpoint(
                    candidate, profile, base_contract,
                    accepted=index != 1, marker=float(index + 1),
                )
            original_hashes = {
                item.checkpoint: replacement._file_digest(item.checkpoint)
                for item in base_candidates
            }

            generator = design["generators"]["development"]
            round_decisions, round_influents, _, _ = replacement._physical_supplemental(
                1, int(generator["final_state"]),
            )
            supplemental = replacement._Candidate(
                "development", 1, 0, 3, round_decisions[0], round_influents[0],
                output / "attempts/replacement/round_000001/candidate_000000.npz",
            )
            _checkpoint(
                supplemental, profile,
                replacement._replacement_contract_hash(profile, "development"),
                accepted=True, marker=9.0,
            )

            result = replacement.generate_mechanistic_block_with_replacements(
                decisions, influents, profile, output, block="development",
            )
            self.assertEqual(result.targets.shape, (3, profile.response_count))
            np.testing.assert_array_equal(result.targets[:, 0], [1.0, 9.0, 3.0])
            np.testing.assert_array_equal(result.decisions[0], decisions[0])
            np.testing.assert_array_equal(result.decisions[2], decisions[2])
            np.testing.assert_array_equal(result.decisions[1], round_decisions[0])
            self.assertEqual(
                result.provenance["source_candidate_round"].tolist(), [0, 1, 0],
            )
            self.assertEqual(len(result.attempts), 4)
            self.assertFalse(result.attempts.iloc[1]["accepted"])
            for path, digest in original_hashes.items():
                self.assertEqual(replacement._file_digest(path), digest)
            for path, digest in legacy_hashes.items():
                self.assertEqual(replacement._file_digest(path), digest)

            self.assertTrue((output / "mechanistic_accepted_v3.npz").is_file())
            self.assertTrue((output / "accepted_diagnostics.csv").is_file())
            self.assertTrue((output / "all_attempts.csv").is_file())
            self.assertTrue((output / "accepted_inputs.npz").is_file())
            self.assertTrue((output / "accepted_provenance.csv").is_file())
            self.assertTrue((output / "base_checkpoint_migration.csv").is_file())
            self.assertTrue((output / "mechanistic_rows_v3.npz").exists())
            self.assertTrue((output / "mechanistic_diagnostics.csv").exists())

            replayed = replacement.generate_mechanistic_block_with_replacements(
                decisions, influents, profile, output, block="development",
            )
            np.testing.assert_array_equal(replayed.targets, result.targets)
            for path, digest in original_hashes.items():
                self.assertEqual(replacement._file_digest(path), digest)
            for path, digest in legacy_hashes.items():
                self.assertEqual(replacement._file_digest(path), digest)

    def test_failed_supplement_is_retained_and_next_round_continues_stream(self) -> None:
        profile = _profile()
        design = create_design(profile)
        decisions = design["test_decisions"]
        influents = design["test_influents"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "test"
            base_contract = replacement._base_contract_hash(
                decisions, influents, profile,
            )
            candidates = [
                replacement._Candidate(
                    "test", 0, index, index, decisions[index], influents[index],
                    output / "rows" / f"row_{index:06d}.npz",
                )
                for index in range(2)
            ]
            _checkpoint(candidates[0], profile, base_contract, accepted=True, marker=1.0)
            _checkpoint(candidates[1], profile, base_contract, accepted=False, marker=2.0)

            generator = design["generators"]["test"]
            round_1_d, round_1_i, round_1_state, _ = replacement._physical_supplemental(
                1, int(generator["final_state"]),
            )
            first = replacement._Candidate(
                "test", 1, 0, 2, round_1_d[0], round_1_i[0],
                output / "attempts/replacement/round_000001/candidate_000000.npz",
            )
            contract = replacement._replacement_contract_hash(profile, "test")
            _checkpoint(first, profile, contract, accepted=False, marker=3.0)
            rejected_hash = replacement._file_digest(first.checkpoint)
            round_2_d, round_2_i, _, _ = replacement._physical_supplemental(
                1, round_1_state,
            )
            second = replacement._Candidate(
                "test", 2, 0, 3, round_2_d[0], round_2_i[0],
                output / "attempts/replacement/round_000002/candidate_000000.npz",
            )
            _checkpoint(second, profile, contract, accepted=True, marker=4.0)

            result = replacement.generate_mechanistic_block_with_replacements(
                decisions, influents, profile, output, block="test",
            )
            self.assertEqual(len(result.attempts), 4)
            self.assertEqual(
                result.provenance["source_candidate_round"].tolist(), [0, 2],
            )
            np.testing.assert_array_equal(result.targets[:, 0], [1.0, 4.0])
            summary = json.loads(
                (output / "replacement_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["supplemental_round_count"], 2)
            self.assertEqual(summary["supplemental_attempt_count"], 2)
            self.assertEqual(summary["supplemental_accepted_count"], 1)
            self.assertEqual(replacement._file_digest(first.checkpoint), rejected_hash)


if __name__ == "__main__":
    unittest.main()
