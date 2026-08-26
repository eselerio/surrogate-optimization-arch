"""Strict, resumable driver for an authorized article-v3 calculation.

The default article workload is 4,000 development inputs and 1,000 holdout
inputs. A user-authorized 50,000-input rerun uses the same frozen 80/20
split (40,000/10,000). This driver creates a source-bound result tree and never
reads a preflight artifact. Expensive stages finish with atomic manifests.

Optimization enters through :func:`run_optimization_stage`, which evaluates
one deterministic local attempt for each route in each of eleven article
cases and publishes convergence certification, exact common-reference replay,
physical-audit, and reporting checkpoints.  The former holdout-wide
smooth/reference equivalence sweep is retained only as legacy code and is not
executed by the article workflow.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
from time import perf_counter, perf_counter_ns, sleep
from typing import Any, Mapping

import casadi as ca
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from closed_loop.manuscript_v3 import (
    ARTICLE_FULL,
    DECISION_LOWER,
    DECISION_NAMES,
    DECISION_UPPER,
    RIDGE_GRID,
    AssessmentResult,
    StudyProfile,
    assess_raw_projected_mechanistic,
    clarifier_for,
    create_design,
    cross_validate_log_overflow_closure,
    cross_validate_ridge,
    overflow_tss_from_response,
    reduce_mechanistic_responses,
    violation_record,
)
from closed_loop.model import (
    COMPONENTS,
    NOMINAL_INFLUENT,
    ArticleOperatingPoint,
    INVARIANT_MATRIX,
    INFLUENT_LOWER,
    INFLUENT_UPPER,
    N_COMPONENTS,
    N_STAGES,
    TSS_VECTOR,
    assemble_target,
    branch_classification,
    diagnostics as mechanistic_diagnostics,
    generation_scale,
    solve_steady_state,
    unpack_state,
)
from closed_loop.projection import (
    LeastSquaresDiagnostics,
    LogOverflowTSSClosure,
    NetworkLayout,
    PhysicalProjector,
    QuadraticFeatureMap,
    QuadraticSurrogate,
    build_network_operators,
)
from closed_loop.v3_reporting import OBJECTIVE_COMPONENT_NAMES, write_reporting_tables
from closed_loop.v3_derivative_audit import audit_casadi_nlp_derivatives
from closed_loop.v3_replacement_generation import (
    MechanisticBlockResult,
    generate_mechanistic_block_with_replacements,
)
from closed_loop import v3_replacement_generation as replacement_generation
from closed_loop.v3_smooth import (
    CONTINUATION_SCHEDULE,
    DEFAULT_OBJECTIVE_WEIGHTS,
    DirectCase,
    DirectMultistartResult,
    DirectStartResult,
    SolverSettings,
    build_direct_nlp,
    branches_match as smooth_branches_match,
    classify_branches,
    compare_smooth_reference,
    engineering_feasible,
    engineering_quantities,
    fit_direct_assets,
    objective_components,
    ordered_normalized_starts as direct_normalized_starts,
    solve_direct_multistart,
    solve_fixed_input_two_start,
)
from closed_loop.v3_surrogate_nlp import (
    EXACT_QP_CENTER_START,
    EXACT_QP_SINGLE_START_PROTOCOL,
    FinalCandidateRecord,
    GAP_CONTINUATION,
    SurrogateCase,
    SurrogateCertificationSettings,
    SurrogateMultistartResult,
    SurrogateSolverSettings,
    SurrogateStartResult,
    build_surrogate_nlp,
    build_surrogate_assets,
    certify_surrogate_local_convergence,
    cold_reproject,
    solve_surrogate_exact_qp_local,
)
from closed_loop.v3_trust import calibrate_trust_diagnostics


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "article_v3"
LEGACY_RUN_ID = "article_full_5000_001"
DEFAULT_RUN_ID = "article_full_5000_002"
RUNNER_SCHEMA = 12
RESPONSE_SCHEMA = "clarifier_inventory_v1"
PROJECTION_SCHEMA = "system_wide_log_overflow_closure_v1"
ASSESSMENT_GATE_EXECUTION_POLICY = "advisory_continue"
DIRECT_SINGLE_CENTER_PROTOCOL = "smooth_direct_single_center_v1"
OPTIMIZATION_PROTOCOL = "single_center_local_exact_qp_no_minimum_srt_v2"
COMPARISON_PROTOCOL = "casewise_exact_common_reference_no_minimum_srt_v4"
TIMING_PROTOCOL = "robustness_casewise_aggregate_v1"
RUN_ID_PATTERN = re.compile(
    r"^article_full_(?:5000|10000|50000)_[A-Za-z0-9][A-Za-z0-9_-]*$"
)
AUTHORIZED_DATASET_TOTALS = (5_000, 50_000)
FROZEN_ACCEPTED_TOTAL = 16_714
FROZEN_DEVELOPMENT_COUNT = 13_371
FROZEN_TEST_COUNT = 3_343
FROZEN_PROFILE_NAME = "article_frozen_16714"
SAMPLED_ACCEPTED_TOTAL = 10_000
SAMPLED_DEVELOPMENT_COUNT = 8_000
SAMPLED_TEST_COUNT = 2_000
SAMPLED_PROFILE_NAME = "article_sampled_10000"
SAMPLED_ACCEPTED_SOURCE_RUN_ID = "article_full_50000_003"
SAMPLED_ACCEPTED_SEED = 20_260_826
FRESH_ROUTE_LOADER_FIX_MIGRATION_ID = "article-v3-fresh-route-loader-v1"
FRESH_ROUTE_LOADER_FIX_RUN_ID = "article_full_10000_sampled_002"
FRESH_ROUTE_LOADER_FIX_PREDECESSOR_SOURCE_DIGEST = (
    "2950773b403f27f66aa8768e6dc12ef740684eab3da3e037c61baff659f61795"
)
FRESH_ROUTE_LOADER_FIX_PREDECESSOR_CONTRACT_DIGEST = (
    "71b6d7b8a7bd5b46b3a63b30bb360669089a09c2339ffee3154d3f29db3bd80a"
)

SOURCE_FILES = (
    "scripts/run_article_v3_5000.py",
    "scripts/build_main_closed_loop_v3.py",
    "main_closed_loop.ipynb",
    "closed_loop/model.py",
    "closed_loop/manuscript_v3.py",
    "closed_loop/projection.py",
    "closed_loop/v3_smooth.py",
    "closed_loop/v3_surrogate_nlp.py",
    "closed_loop/v3_active_set.py",
    "closed_loop/v3_derivative_audit.py",
    "closed_loop/v3_trust.py",
    "closed_loop/v3_reporting.py",
    "closed_loop/v3_replacement_generation.py",
    "config/params_manuscript_v3.json",
    "article/wip_v3/manuscript.tex",
    "article/wip_v3/supplementary_material.tex",
    "pyproject.toml",
    "uv.lock",
)
DESIGN_ARRAYS = (
    "development_decisions",
    "development_influents",
    "test_decisions",
    "test_influents",
    "robustness_influents",
)

# A reduced-response fork may retain only files that define or audit the
# mechanistic generation.  In particular, do not copy arbitrary future files
# merely because they were placed below ``datasets``: fitted-response arrays
# and other downstream products must be rebuilt under schema 10.
REDUCED_FORK_DATASET_FILES = frozenset({
    "datasets/design.npz",
    "datasets/effective_design.npz",
    "datasets/effective_design_manifest.json",
    "datasets/frozen_accepted_complete.json",
    "datasets/sampled_accepted_complete.json",
    *(
        f"datasets/{block}/{name}"
        for block in ("development", "test")
        for name in (
            "mechanistic_accepted_v3.npz",
            "accepted_inputs.npz",
            "accepted_diagnostics.csv",
            "all_attempts.csv",
            "accepted_provenance.csv",
            "base_checkpoint_migration.csv",
            "replacement_summary.json",
            "accepted_coordinate_coverage.csv",
            "rejection_reason_summary.csv",
            "mechanistic_rows_v3.npz",
            "mechanistic_diagnostics.csv",
            "block_complete.json",
        )
    ),
})
REDUCED_FORK_DATASET_PREFIXES = tuple(
    f"datasets/{block}/{directory}/"
    for block in ("development", "test")
    for directory in ("rows", "source_rows", "attempts")
)
REDUCED_FORK_INPUT_FILES = (
    "inputs/generator_records.json",
    "inputs/partial_generation_fork.json",
    "inputs/frozen_accepted_partition.json",
    "inputs/frozen_accepted/source_design_50000.npz",
    "inputs/random_sampled_accepted_partition.json",
)


@dataclass(frozen=True)
class SourceContractMigrationAuthorization:
    """Narrow authorization for one already-started run's source migration."""

    migration_id: str
    run_id: str
    authorized_date: str
    reason: str
    predecessor_runner_schema: int
    successor_runner_schema: int
    predecessor_source_digest: str
    predecessor_contract_file_digest: str
    allowed_changed_source_files: frozenset[str]
    required_changed_source_files: frozenset[str]
    required_artifact_digests: Mapping[str, str]
    expected_accepted_rows: int
    expected_rejected_rows: int


@dataclass(frozen=True)
class AssessmentRecoveryMigrationAuthorization:
    """Pinned authorization to repair assessment without recomputing prior stages."""

    migration_id: str
    run_id: str
    authorized_date: str
    reason: str
    predecessor_runner_schema: int
    successor_runner_schema: int
    predecessor_source_digest: str
    predecessor_contract_file_digest: str
    required_prior_migration_ids: tuple[str, ...]
    predecessor_source_snapshot: str
    allowed_changed_source_files: frozenset[str]
    required_changed_source_files: frozenset[str]
    required_artifact_digests: Mapping[str, str]
    expected_effective_design_digest: str
    expected_ridge_input_digest: str


@dataclass(frozen=True)
class OptimizationProtocolMigrationAuthorization:
    """Pinned authorization to replace only the unfinished optimization stage."""

    migration_id: str
    run_id: str
    authorized_date: str
    reason: str
    predecessor_runner_schema: int
    successor_runner_schema: int
    predecessor_source_digest: str
    predecessor_contract_file_digest: str
    required_prior_migration_ids: tuple[str, ...]
    predecessor_source_snapshot: str
    allowed_changed_source_files: frozenset[str]
    required_changed_source_files: frozenset[str]
    required_artifact_digests: Mapping[str, str]
    expected_assessment_input_digest: str


@dataclass(frozen=True)
class CasewiseComparisonMigrationAuthorization:
    """Pinned migration retaining completed searches for casewise validation."""

    migration_id: str
    run_id: str
    authorized_date: str
    reason: str
    predecessor_runner_schema: int
    successor_runner_schema: int
    predecessor_source_digest: str
    predecessor_contract_file_digest: str
    required_prior_migration_ids: tuple[str, ...]
    predecessor_source_snapshot: str
    allowed_changed_source_files: frozenset[str]
    required_changed_source_files: frozenset[str]
    required_artifact_digests: Mapping[str, str]
    expected_case_marker_set_digest: str
    retired_casewise_snapshot: str | None = None


GENERATION_REPLACEMENT_MIGRATION = SourceContractMigrationAuthorization(
    migration_id="article-v3-generation-replacement-v1",
    run_id=LEGACY_RUN_ID,
    authorized_date="2026-08-23",
    reason=(
        "User-authorized replacement of rejected mechanistic generation attempts "
        "while retaining every independently verified accepted checkpoint."
    ),
    predecessor_runner_schema=2,
    successor_runner_schema=3,
    predecessor_source_digest=(
        "b1ea4a3dedc018acf461e0c9d6a4e9bf2e3124c41e1e7ea9f0dee16dc9765c87"
    ),
    predecessor_contract_file_digest=(
        "1435125e0538627d72149773cc2ac5c055dd6a1bd2da79d72836829225dbe8e5"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_replacement_generation.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_replacement_generation.py",
    }),
    required_artifact_digests={
        "inputs/generator_records.json": (
            "d8ac45c0f2cbba3cddd64ee171fbb9a0b922b246e95dce046ffb5a303355a669"
        ),
        "datasets/design.npz": (
            "e278a3bf56c4a8d1099eb88db2c3e86960027952b39547b7ac8f7f1b0ba95cb0"
        ),
        "datasets/development/mechanistic_rows_v3.npz": (
            "ac30ee5682884ea4aea21429c3016742b0fc825d5eaf6c09bdf7d26c875b454c"
        ),
        "datasets/development/mechanistic_diagnostics.csv": (
            "16d846f6f66154281ef977c925ae671a7ac1c80d423d0aac4436145daaba188b"
        ),
    },
    expected_accepted_rows=3_680,
    expected_rejected_rows=320,
)


ASSESSMENT_RECOVERY_MIGRATION = AssessmentRecoveryMigrationAuthorization(
    migration_id="article-v3-projection-audit-v1",
    run_id=LEGACY_RUN_ID,
    authorized_date="2026-08-23",
    reason=(
        "User-authorized repair of the projection audit and continuation after "
        "an advisory assessment gate, without recomputing completed generation "
        "or ridge fitting."
    ),
    predecessor_runner_schema=3,
    successor_runner_schema=4,
    predecessor_source_digest=(
        "4e599e0fd18ff3ba17c7e9af1c047e53067ff610b9f7bb787f1effc230f3d063"
    ),
    predecessor_contract_file_digest=(
        "118c06a8d4c9f418e50477138e5eca7c8c376ecbd4653de7cdc74bf83a941f87"
    ),
    required_prior_migration_ids=("article-v3-generation-replacement-v1",),
    predecessor_source_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-projection-audit-v1-predecessor-source"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/projection.py",
        "closed_loop/v3_trust.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/projection.py",
        "closed_loop/v3_trust.py",
    }),
    required_artifact_digests={
        "inputs/generator_records.json": (
            "d8ac45c0f2cbba3cddd64ee171fbb9a0b922b246e95dce046ffb5a303355a669"
        ),
        "datasets/design.npz": (
            "e278a3bf56c4a8d1099eb88db2c3e86960027952b39547b7ac8f7f1b0ba95cb0"
        ),
        "datasets/development/block_complete.json": (
            "3409b03537613d0d98166cee151ff6e6451ef16ebd26b1cf058d25d827455c06"
        ),
        "datasets/test/block_complete.json": (
            "5387fdac043b36ab4518e35a26b59ea35b77f8f7ef7db2ddb9c3ec2c8ecbe95f"
        ),
        "datasets/effective_design.npz": (
            "7716745352e918d67f8c88e65be730f335b9b079a28cf1852de19d5d7194ca96"
        ),
        "datasets/effective_design_manifest.json": (
            "33a7d7172b178f7b220c065e0003edb8bc10cc74dfc27e89970d71a20d02c255"
        ),
        "metrics/mechanistic_generation_summary.csv": (
            "138755780cfc643259ec740b291d820268021c84ea46b076e841d0069726ae53"
        ),
        "models/ridge_complete.json": (
            "8fb5374758492d8f159b8218f6621cca112fe1d243e4f66f8bde52ada8a91a5d"
        ),
        "models/ridge_surrogate.npz": (
            "9817ba3461d8d95c9e5098ea809f9be6cf0b65724dc9fec12f883ce30eabc632"
        ),
        "metrics/ridge_cross_validation.csv": (
            "29153e71ea86ec2892cab4ff9f58d264b6feb67a80627611bc965e83142c81a5"
        ),
        "metrics/ridge_fold_membership.csv": (
            "9155b59b038cf5dd702ea8053977addd742779fcb904d6dbd5fb13aff4686cd2"
        ),
    },
    expected_effective_design_digest=(
        "2f71ebc39dcbe9d70b2c927ae273bf4fb3700ee89f6dd85ccb18bed942207d9b"
    ),
    expected_ridge_input_digest=(
        "6466850f5d4ac2e651e641e2e19b00d9fe9c9e5412621fbe7aefb6d121d5028d"
    ),
)


SINGLE_START_EXACT_QP_MIGRATION = OptimizationProtocolMigrationAuthorization(
    migration_id="article-v3-direct-active-set-v1",
    run_id=LEGACY_RUN_ID,
    authorized_date="2026-08-23",
    reason=(
        "User-authorized replacement of the unfinished nine-start embedded-KKT "
        "optimization stage by one deterministic local attempt per route and "
        "case, with the surrogate solved by the seven-variable exact-QP route."
    ),
    predecessor_runner_schema=4,
    successor_runner_schema=5,
    predecessor_source_digest=(
        "5ca76a132a467fab76717d47c00e62e4901b51b0bbeefa6e144ed3264e843fc3"
    ),
    predecessor_contract_file_digest=(
        "c198eed5861079da4cc6b741264eee77bad035dc6ab4eaa39dfeebc018075501"
    ),
    required_prior_migration_ids=(
        "article-v3-generation-replacement-v1",
        "article-v3-projection-audit-v1",
    ),
    predecessor_source_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-direct-active-set-v1-predecessor-source"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_surrogate_nlp.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_surrogate_nlp.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_artifact_digests={
        "metrics/assessment_complete.json": (
            "2c0c9e37f539868e52b9ecd9f0f387f81f5a6e0d6d90ae9c46016bb74fc3fd3a"
        ),
        "metrics/admission_gate.json": (
            "3631f3fda28191dca854ed1cfe79fcf1114073217acbea0702d3a2db78c5f6eb"
        ),
        "metrics/untouched_prediction_metrics.csv": (
            "f446f0207b39fe4802951e94a75630620eae4ae390434679f812c12553e1aaaa"
        ),
        "metrics/physical_violations_assessment.csv": (
            "b9c5ce45431c80c99c003db89d163dba4c7a76098c1d57756ffa5ef6ce0fdc5b"
        ),
        "metrics/projection_qp_diagnostics.csv": (
            "dbb7e42a8a6c90d2bdc52883418316d3158d362b8a7e56a5a26c305dc034d9d1"
        ),
        "metrics/projection_feasibility_bound.csv": (
            "0781a621b42d69ca0c5fd983a156afc2afedd1a2f8d03a6a462d2ba77601ea6c"
        ),
        "predictions/untouched_test.npz": (
            "dbc11728e6d434d1d8b37b6623b9cd12f69deadfac2fe2fb82bdef7e05650b61"
        ),
        "metrics/trust_development_oof.csv": (
            "6658b47cd0c032e8434a82238d1ecae17afcb2acd17dd062b5bdfc297268a5dd"
        ),
        "metrics/trust_limits.json": (
            "73088c1cde7dddf6964256218b541f9fa035273bcac252fa025453930df56d4f"
        ),
        "models/trust_calibration.npz": (
            "f334c3d179f35d1ab2a748942854436cd88cc64e4e6190881bf15cd1d3409918"
        ),
    },
    expected_assessment_input_digest=(
        "49bc1639cfd85e0864dc07d16f33959e5b11c132ef7f1a95b9d89908f8d13323"
    ),
)


CASEWISE_COMMON_REFERENCE_MIGRATION = CasewiseComparisonMigrationAuthorization(
    migration_id="article-v3-casewise-common-reference-v1",
    run_id=LEGACY_RUN_ID,
    authorized_date="2026-08-24",
    reason=(
        "User-authorized retirement of the unfinished 1,000-row smooth/reference "
        "equivalence stage in favor of convergence-qualified, exact nonsmooth "
        "common-reference evaluation of the nominal and robustness decisions."
    ),
    predecessor_runner_schema=5,
    successor_runner_schema=6,
    predecessor_source_digest=(
        "85c1d10cb7c9ffac4291bff3f2e681670fb5f0fe7e6a0935ef898539f9efc445"
    ),
    predecessor_contract_file_digest=(
        "a4a761d1744bb58a21611ccf6029faaca6c1ef43b73d71d31848d81d3f2b8a1e"
    ),
    required_prior_migration_ids=(
        "article-v3-generation-replacement-v1",
        "article-v3-projection-audit-v1",
        "article-v3-direct-active-set-v1",
    ),
    predecessor_source_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-casewise-common-reference-v1-predecessor-source"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_smooth.py",
        "closed_loop/v3_surrogate_nlp.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_smooth.py",
        "closed_loop/v3_surrogate_nlp.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_artifact_digests={
        "metrics/assessment_complete.json": (
            "2c0c9e37f539868e52b9ecd9f0f387f81f5a6e0d6d90ae9c46016bb74fc3fd3a"
        ),
        "optimization/nominal/case_complete.json": (
            "826f8d9e4624749840508b661e15c143ed640ffd16f00df037b7ade4714a6d4f"
        ),
        "optimization/robustness_01/case_complete.json": (
            "116e0675cde324792d06e9ff5e9a2aea2ca36d0622dfcaea1d1cd2209854a560"
        ),
        "optimization/robustness_02/case_complete.json": (
            "cdc0f43e4df66239154d95666e476824bef5c4b90d39277b1cb57638a8364eab"
        ),
        "optimization/robustness_03/case_complete.json": (
            "91b80ee4cddcba698205d653aeeba9c7fae6b40f818f9ae5b5803357dbf18af9"
        ),
        "optimization/robustness_04/case_complete.json": (
            "f017043c80e9a5b799852926c420f5c67fe20e0e13b687eb49d451f68bcb86bb"
        ),
        "optimization/robustness_05/case_complete.json": (
            "2584eb58a045d8ba260613db200693e3b82999861d06437f1041bc99c6d9184c"
        ),
        "optimization/robustness_06/case_complete.json": (
            "564c13c8a31210b2e666fdaf0881d23bba859268bfd59f731580ce8b08eb3dfa"
        ),
        "optimization/robustness_07/case_complete.json": (
            "75505371a14f6d797131feecb1cd2ef5e24687cb34369060d91015aa694e0491"
        ),
        "optimization/robustness_08/case_complete.json": (
            "0b402375e75ff8dbb62647eb2b71bc56848caae5599c1f99b5c52ea693696735"
        ),
        "optimization/robustness_09/case_complete.json": (
            "f3b6856d3894e03393d3ba93d1d7c512d6ab9677d8b762b1ba497f2da77389a9"
        ),
        "optimization/robustness_10/case_complete.json": (
            "dee22ab590a32b96a9e43abd703bb7b897d1f3567985d38e3e188ad33377b45a"
        ),
    },
    expected_case_marker_set_digest=(
        "4f45fb0014b2c0de559bdd9f8530cfa5622d6aafa8c85856cd747ae0a8fb047f"
    ),
)


CONVERGENCE_POLL_REFINEMENT_MIGRATION = CasewiseComparisonMigrationAuthorization(
    migration_id="article-v3-convergence-poll-refinement-v1",
    run_id=LEGACY_RUN_ID,
    authorized_date="2026-08-24",
    reason=(
        "Outcome-aware correction after the first casewise pass showed that a "
        "full-rank feasible-direction requirement was mathematically invalid at "
        "constrained endpoints and the 120-evaluation poll cap was insufficient. "
        "The preliminary casewise outputs are archived; completed primary searches "
        "and every upstream scientific artifact remain unchanged."
    ),
    predecessor_runner_schema=6,
    successor_runner_schema=7,
    predecessor_source_digest=(
        "a598aaa531f6d5fdd0de477ff5c91699abc1c7d56f52f58b4a7f893c0fd9fc88"
    ),
    predecessor_contract_file_digest=(
        "9cfa88c0a2aa8703182d29b059b5993efd59d5f473d86b26ee0f9bf6978bad1e"
    ),
    required_prior_migration_ids=(
        "article-v3-generation-replacement-v1",
        "article-v3-projection-audit-v1",
        "article-v3-direct-active-set-v1",
        "article-v3-casewise-common-reference-v1",
    ),
    predecessor_source_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-convergence-poll-refinement-v1-predecessor-source"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_surrogate_nlp.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_surrogate_nlp.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_artifact_digests={
        **CASEWISE_COMMON_REFERENCE_MIGRATION.required_artifact_digests,
        "inputs/contract_migrations/article-v3-casewise-common-reference-v1.json": (
            "b9331954e41ef2b206c5e6342d6a40685f086594b4aa980d82b49b959f716470"
        ),
        "inputs/contract_migrations/"
        "article-v3-casewise-common-reference-v1-retained.json": (
            "1b722058973b6fa6820aa1bbd73f412af551e414c061d46f8d9d819d05c0b040"
        ),
    },
    expected_case_marker_set_digest=(
        "4f45fb0014b2c0de559bdd9f8530cfa5622d6aafa8c85856cd747ae0a8fb047f"
    ),
    retired_casewise_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-convergence-poll-refinement-v1-predecessor-casewise"
    ),
)


POLL_LINESEARCH_FORK_MIGRATION = CasewiseComparisonMigrationAuthorization(
    migration_id="article-v3-poll-linesearch-v1",
    run_id=LEGACY_RUN_ID,
    authorized_date="2026-08-24",
    reason=(
        "User-authorized rerun in a new self-contained result directory after "
        "the v2 fine-radius poll made 36 accepted moves and exhausted 2,500 "
        "exact-QP evaluations in robustness case 05. The v3 search adds exact-QP "
        "geometric ray acceleration while retaining the same final two-scale "
        "no-descent audit. All accepted datasets, fitted assets, assessment "
        "artifacts, and primary optimization searches are copied and hash-pinned."
    ),
    predecessor_runner_schema=7,
    successor_runner_schema=8,
    predecessor_source_digest=(
        "d12916000eab67f3d05c9bae4a61358e98411c6eadece4561b80264c1cf02b4d"
    ),
    predecessor_contract_file_digest=(
        "98ba7ce3bed9798dcf5615b5a8cbe9f2f87245a7892bd11750999faf11ac7805"
    ),
    required_prior_migration_ids=(
        "article-v3-generation-replacement-v1",
        "article-v3-projection-audit-v1",
        "article-v3-direct-active-set-v1",
        "article-v3-casewise-common-reference-v1",
        "article-v3-convergence-poll-refinement-v1",
    ),
    predecessor_source_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-poll-linesearch-v1-predecessor-source"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_surrogate_nlp.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_surrogate_nlp.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_artifact_digests={
        **CONVERGENCE_POLL_REFINEMENT_MIGRATION.required_artifact_digests,
        "inputs/contract_migrations/"
        "article-v3-convergence-poll-refinement-v1.json": (
            "fd1751ab404f9700856102a7ebaa66b2ba7a9656098c0f916be0ae36e3cb3274"
        ),
        "inputs/contract_migrations/"
        "article-v3-convergence-poll-refinement-v1-retained.json": (
            "f56c1fd469baec2534357e30d5aa2630460dbd2b117e010ab20ebe45ac44835e"
        ),
        "optimization/robustness_05/surrogate_local_convergence.json": (
            "5a87bbb423117a7288cf3dfb02d348986adf6f4445ee85c75bed43c210545bfd"
        ),
        "optimization/robustness_05/surrogate_local_convergence_complete.json": (
            "2875f44aad41aa7b58cc739caf90ce0b6e825c8acb3ad37cbeb2501cfb0fa946"
        ),
        "optimization/robustness_05/surrogate_certified.npz": (
            "105a8c9c7fe7a1054c85a15eea9cffa8ef9c7e87f642ff6daac95a2c6f47202d"
        ),
    },
    expected_case_marker_set_digest=(
        "4f45fb0014b2c0de559bdd9f8530cfa5622d6aafa8c85856cd747ae0a8fb047f"
    ),
    retired_casewise_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-poll-linesearch-v1-predecessor-casewise"
    ),
)


CASEWISE_TIMING_AGGREGATION_MIGRATION = CasewiseComparisonMigrationAuthorization(
    migration_id="article-v3-casewise-timing-v1",
    run_id=DEFAULT_RUN_ID,
    authorized_date="2026-08-24",
    reason=(
        "User-authorized removal of the unfinished 1,000-row repeated inference "
        "benchmark. Runtime is instead summarized from the ten completed "
        "robustness-case route, certification, recovery, and exact-replay records; "
        "all completed scientific casewise artifacts are retained unchanged."
    ),
    predecessor_runner_schema=8,
    successor_runner_schema=9,
    predecessor_source_digest=(
        "403b8d96b2b46693456b19d0a7566260748695f2284f949ef0259a39be782657"
    ),
    predecessor_contract_file_digest=(
        "5ce8065e93addd82ef6b92d18e32caaefb430b934393bc888524bd70429a2ab2"
    ),
    required_prior_migration_ids=(
        "article-v3-generation-replacement-v1",
        "article-v3-projection-audit-v1",
        "article-v3-direct-active-set-v1",
        "article-v3-casewise-common-reference-v1",
        "article-v3-convergence-poll-refinement-v1",
        "article-v3-poll-linesearch-v1",
    ),
    predecessor_source_snapshot=(
        "inputs/contract_migrations/"
        "article-v3-casewise-timing-v1-predecessor-source"
    ),
    allowed_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_changed_source_files=frozenset({
        "scripts/run_article_v3_5000.py",
        "closed_loop/v3_reporting.py",
        "scripts/build_main_closed_loop_v3.py",
        "main_closed_loop.ipynb",
        "config/params_manuscript_v3.json",
        "article/wip_v3/manuscript.tex",
        "article/wip_v3/supplementary_material.tex",
    }),
    required_artifact_digests={
        "metrics/case_common_reference_comparison.csv": (
            "c9a5ba276dad0a056a8677423ac8e611bc9badfb728eda92a90b6d0df08f7bb6"
        ),
        "metrics/selected_candidate_reference_evaluation.csv": (
            "0f6c2457b6a2272a9e32b0a5285e2a7447455c13b01ae149d567aeb17b40bc8d"
        ),
        "optimization/nominal/casewise_comparison_complete.json": (
            "2b192201efa51bfac9d84bf301b331603c8b706acd6144ac7ee390cc0cdec0de"
        ),
        "optimization/robustness_01/casewise_comparison_complete.json": (
            "97a17fc4e28d8220eb5845884224c9b420f4548befecc356f2c112b910304d6f"
        ),
        "optimization/robustness_02/casewise_comparison_complete.json": (
            "d5451187444440731879f5339339a9a8012d59dcf8fefc8fcd0926967c9015c1"
        ),
        "optimization/robustness_03/casewise_comparison_complete.json": (
            "1fe8a5ee1e6cc93bb3bf45af235860fae47a4bf2ff0cff22c6af11a44e976082"
        ),
        "optimization/robustness_04/casewise_comparison_complete.json": (
            "b699d7d353a1219f9363730168ebe287ff61b8eaeab60de82518ffdd18d47dfa"
        ),
        "optimization/robustness_05/casewise_comparison_complete.json": (
            "73af4b986e8e1dd80691f31892f624b5c71ecf6867c32dea123546c559c85d81"
        ),
        "optimization/robustness_06/casewise_comparison_complete.json": (
            "dd65659bd9be9bcef4f611b407924ebb34ae774e9192a3f2b9ab6fa0102f29c6"
        ),
        "optimization/robustness_07/casewise_comparison_complete.json": (
            "f48df365df9d3ea4e79053d58fc4046ecf0a1ec85c879576df9347c43638658e"
        ),
        "optimization/robustness_08/casewise_comparison_complete.json": (
            "3ac569aea33c12e85cbfeeb7ef6b4d1899936b235304948e8cd4056ed5772af5"
        ),
        "optimization/robustness_09/casewise_comparison_complete.json": (
            "2d47ba0b3074f8b75119441b55835d5d469eff2aebc6551dbefc04231045e372"
        ),
        "optimization/robustness_10/casewise_comparison_complete.json": (
            "ff23d396573667b908d7475b4e742819846295ff3b8750542b9f0a0de1301377"
        ),
    },
    expected_case_marker_set_digest=(
        "4f45fb0014b2c0de559bdd9f8530cfa5622d6aafa8c85856cd747ae0a8fb047f"
    ),
)


@dataclass(frozen=True)
class AnalysisBundle:
    passed: bool
    model: QuadraticSurrogate
    direct_assets: Any
    surrogate_assets: Any
    assessment: AssessmentResult | None
    gate: dict[str, Any]
    overflow_closure: LogOverflowTSSClosure | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Accepted development/test targets and their effective input design."""

    design: dict[str, object]
    development_targets: np.ndarray
    test_targets: np.ndarray

    def __iter__(self):
        """Preserve the historical two-target unpacking interface."""

        yield self.development_targets
        yield self.test_targets

    def __getitem__(self, index: int) -> np.ndarray:
        return (self.development_targets, self.test_targets)[index]


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Publish atomically despite transient Windows scanner/file-lock races."""

    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            transient = os.name == "nt" and getattr(exc, "winerror", None) in {5, 32}
            if not transient or attempt == 6:
                raise
            sleep(0.01 * (2**attempt))


def _json_ready(value: Any, *, nonfinite_to_none: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item, nonfinite_to_none=nonfinite_to_none)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, nonfinite_to_none=nonfinite_to_none) for item in value
        ]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist(), nonfinite_to_none=nonfinite_to_none)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            if nonfinite_to_none:
                return None
            raise ValueError("non-finite numbers are not permitted in JSON contracts")
        return result
    return value


def atomic_json(path: Path, value: Any, *, nonfinite_to_none: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_ready(value, nonfinite_to_none=nonfinite_to_none),
                stream, indent=2, sort_keys=True, allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file_digests() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"required article source is missing: {path}")
        result[relative] = file_digest(path)
    return result


def source_digest(files: Mapping[str, str] | None = None) -> str:
    manifest = dict(files or source_file_digests())
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def array_digest(**arrays: np.ndarray) -> str:
    digest = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _artifact_hashes(run: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    return {path.relative_to(run).as_posix(): file_digest(path) for path in paths}


def _artifacts_match(run: Path, expected: Mapping[str, str]) -> bool:
    if not expected:
        return False
    return all(
        (run / relative).is_file()
        and file_digest(run / relative) == expected_digest
        for relative, expected_digest in expected.items()
    )


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "pandas", "casadi", "osqp"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def resolve_run_directory(run_id: str, results_root: Path = RESULTS_ROOT) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise ValueError(
            "full-run id must match article_full_<5000-or-50000>_<identifier>; "
            "preflight names and path components are forbidden"
        )
    root = results_root.resolve()
    run = (root / run_id).resolve()
    if run.parent != root:
        raise ValueError("the full-run directory must be a direct child of its result root")
    return run


def profile_for_dataset_total(dataset_total: int) -> StudyProfile:
    """Return the frozen 80/20 article profile for an authorized row count."""

    if dataset_total not in AUTHORIZED_DATASET_TOTALS:
        raise ValueError(
            f"dataset total must be one of {AUTHORIZED_DATASET_TOTALS}, "
            f"not {dataset_total}"
        )
    if dataset_total == 5_000:
        return ARTICLE_FULL
    return replace(
        ARTICLE_FULL,
        name=f"article_full_{dataset_total}",
        development_count=dataset_total * 4 // 5,
        test_count=dataset_total // 5,
    )


def frozen_accepted_profile() -> StudyProfile:
    """Profile for the user-frozen accepted subset of the interrupted 50k run."""

    return replace(
        ARTICLE_FULL,
        name=FROZEN_PROFILE_NAME,
        development_count=FROZEN_DEVELOPMENT_COUNT,
        test_count=FROZEN_TEST_COUNT,
    )


def sampled_accepted_profile() -> StudyProfile:
    """Profile for a deterministic random sample of the accepted 50k rows."""

    return replace(
        ARTICLE_FULL,
        name=SAMPLED_PROFILE_NAME,
        development_count=SAMPLED_DEVELOPMENT_COUNT,
        test_count=SAMPLED_TEST_COUNT,
    )


def validate_authorized_profile(profile: StudyProfile) -> None:
    dataset_total = profile.development_count + profile.test_count
    frozen = profile.name == FROZEN_PROFILE_NAME
    sampled = profile.name == SAMPLED_PROFILE_NAME
    if dataset_total not in AUTHORIZED_DATASET_TOTALS and not frozen and not sampled:
        raise RuntimeError(
            f"article profile requests unauthorized dataset total {dataset_total}"
        )
    if frozen and (
        dataset_total != FROZEN_ACCEPTED_TOTAL
        or profile.development_count != FROZEN_DEVELOPMENT_COUNT
        or profile.test_count != FROZEN_TEST_COUNT
    ):
        raise RuntimeError("frozen accepted profile violates its fixed 80/20 split")
    if sampled and (
        dataset_total != SAMPLED_ACCEPTED_TOTAL
        or profile.development_count != SAMPLED_DEVELOPMENT_COUNT
        or profile.test_count != SAMPLED_TEST_COUNT
    ):
        raise RuntimeError("sampled accepted profile violates its fixed 80/20 split")
    expected = {
        "name": (
            FROZEN_PROFILE_NAME if frozen
            else SAMPLED_PROFILE_NAME if sampled
            else "article_full" if dataset_total == 5_000
            else f"article_full_{dataset_total}"
        ),
        "development_count": (
            FROZEN_DEVELOPMENT_COUNT if frozen
            else SAMPLED_DEVELOPMENT_COUNT if sampled
            else dataset_total * 4 // 5
        ),
        "test_count": (
            FROZEN_TEST_COUNT if frozen
            else SAMPLED_TEST_COUNT if sampled
            else dataset_total // 5
        ),
        "robustness_count": 10,
        "layer_count": 10,
        "development_seed": 100_042,
        "test_seed": 100_043,
        "robustness_seed": 314_159,
        "article_eligible": True,
        "enforce_admission_gate": True,
    }
    actual = asdict(profile)
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items() if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"article profile violates the authorized contract: {mismatches}")


def _expected_design_shapes(profile: StudyProfile) -> dict[str, tuple[int, int]]:
    return {
        "development_decisions": (profile.development_count, 7),
        "development_influents": (profile.development_count, 20),
        "test_decisions": (profile.test_count, 7),
        "test_influents": (profile.test_count, 20),
        "robustness_influents": (profile.robustness_count, 20),
    }


def validate_design(design: Mapping[str, object], profile: StudyProfile) -> None:
    for name, shape in _expected_design_shapes(profile).items():
        if name not in design:
            raise RuntimeError(f"fixed design is missing {name}")
        value = np.asarray(design[name], dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(f"fixed design {name} has invalid shape or values")
        lower, upper = (
            (DECISION_LOWER, DECISION_UPPER)
            if name.endswith("decisions") else (INFLUENT_LOWER, INFLUENT_UPPER)
        )
        if np.any(value < lower) or np.any(value > upper):
            raise RuntimeError(f"fixed design {name} lies outside its declared box")
    generators = design.get("generators")
    if not isinstance(generators, Mapping):
        raise RuntimeError("fixed design is missing generator records")
    expected_seeds = {
        "development": profile.development_seed,
        "test": profile.test_seed,
        "robustness": profile.robustness_seed,
    }
    for block, seed in expected_seeds.items():
        record = generators.get(block)
        if not isinstance(record, Mapping) or int(record.get("seed", -1)) != seed:
            raise RuntimeError(f"fixed design has an invalid {block} generator record")


def _design_digest(design: Mapping[str, object]) -> str:
    return array_digest(**{
        name: np.asarray(design[name], dtype="<f8") for name in DESIGN_ARRAYS
    })


def load_or_create_design(run: Path, profile: StudyProfile) -> dict[str, object]:
    expected = create_design(profile)
    validate_design(expected, profile)
    path = run / "datasets" / "design.npz"
    records_path = run / "inputs" / "generator_records.json"
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(DESIGN_ARRAYS):
                raise RuntimeError("existing design checkpoint has unexpected arrays")
            for name in DESIGN_ARRAYS:
                if not np.array_equal(stored[name], np.asarray(expected[name])):
                    raise RuntimeError("existing design checkpoint differs from fixed design")
    else:
        atomic_npz(path, **{name: np.asarray(expected[name]) for name in DESIGN_ARRAYS})
    expected_records = _json_ready(expected["generators"])
    if records_path.is_file():
        existing_records = json.loads(records_path.read_text(encoding="utf-8"))
        if existing_records != expected_records:
            raise RuntimeError("existing generator record differs from fixed design")
    else:
        atomic_json(records_path, expected_records)
    return expected


def _build_contract(
    run_id: str, profile: StudyProfile, source_files: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "runner_schema": RUNNER_SCHEMA,
        "run_id": run_id,
        "profile": asdict(profile),
        "fixed_dataset_total": profile.development_count + profile.test_count,
        "development_test_split": [profile.development_count, profile.test_count],
        "source_digest": source_digest(source_files),
        "source_files": dict(source_files),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_versions": _runtime_versions(),
        "assessment_gate_execution_policy": ASSESSMENT_GATE_EXECUTION_POLICY,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "validation_protocol": COMPARISON_PROTOCOL,
        "timing_protocol": TIMING_PROTOCOL,
        "response_schema": {
            "name": RESPONSE_SCHEMA,
            "mechanistic_response_count": profile.mechanistic_response_count,
            "surrogate_response_count": profile.surrogate_response_count,
            "shared_coordinate_count": (N_STAGES + 3) * N_COMPONENTS,
            "clarifier_inventory_formula": "sum(layer_volume_m3 * layer_tss_g_m3)",
            "clarifier_volume_m3": 6_000.0,
            "holdout_role": "frozen_post_selection_descriptive",
        },
        "projection_schema": PROJECTION_SCHEMA,
        "dataset_protocol": (
            "frozen_accepted_checkpoint_split_80_20_v1"
            if profile.name == FROZEN_PROFILE_NAME
            else "complete_declared_generation_with_replacements_v1"
        ),
        "preflight_artifacts_permitted": False,
        "full_run_admission_gate_bypass_permitted": False,
    }


def _canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def _validate_pinned_artifacts(
    run: Path, expected: Mapping[str, str],
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, required_digest in sorted(expected.items()):
        path = (run / relative).resolve()
        if run.resolve() not in path.parents or not path.is_file():
            raise RuntimeError(f"migration artifact is missing or outside the run: {relative}")
        observed_digest = file_digest(path)
        if observed_digest != required_digest:
            raise RuntimeError(
                f"migration refused because pinned artifact bytes changed: {relative}"
            )
        observed[relative] = observed_digest
    return observed


def _validate_retained_generation_checkpoints(
    run: Path, authorization: SourceContractMigrationAuthorization,
) -> dict[str, Any]:
    """Verify each retained accepted row against pinned aggregate artifacts."""

    pinned = _validate_pinned_artifacts(
        run, authorization.required_artifact_digests,
    )
    design_path = run / "datasets" / "design.npz"
    aggregate_path = run / "datasets" / "development" / "mechanistic_rows_v3.npz"
    diagnostics_path = (
        run / "datasets" / "development" / "mechanistic_diagnostics.csv"
    )
    try:
        with np.load(design_path, allow_pickle=False) as stored:
            decisions = np.asarray(stored["development_decisions"])
            influents = np.asarray(stored["development_influents"])
        with np.load(aggregate_path, allow_pickle=False) as stored:
            targets = np.asarray(stored["targets"])
            states_start_1 = np.asarray(stored["states_start_1"])
            states_start_2 = np.asarray(stored["states_start_2"])
        diagnostics = pd.read_csv(diagnostics_path)
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError("cannot inspect pinned generation artifacts") from exc
    row_count = len(decisions)
    if (
        len(influents) != row_count
        or len(targets) != row_count
        or len(states_start_1) != row_count
        or len(states_start_2) != row_count
        or len(diagnostics) != row_count
    ):
        raise RuntimeError("pinned generation artifacts disagree on row count")
    if not {"row", "accepted"}.issubset(diagnostics.columns):
        raise RuntimeError("pinned generation diagnostics omit row or accepted")
    rows = pd.to_numeric(diagnostics["row"], errors="coerce").to_numpy()
    if not np.array_equal(rows, np.arange(row_count)):
        raise RuntimeError("pinned generation diagnostics are not in fixed row order")
    accepted_values = diagnostics["accepted"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if accepted_values.isna().any():
        raise RuntimeError("pinned generation diagnostics have invalid acceptance flags")
    accepted_rows = np.flatnonzero(accepted_values.to_numpy(dtype=bool))
    rejected_rows = np.flatnonzero(~accepted_values.to_numpy(dtype=bool))
    if (
        len(accepted_rows) != authorization.expected_accepted_rows
        or len(rejected_rows) != authorization.expected_rejected_rows
    ):
        raise RuntimeError(
            "migration refused because accepted/rejected generation counts changed"
        )
    checkpoints: list[dict[str, Any]] = []
    row_contract_hashes: set[str] = set()
    for index in accepted_rows:
        relative = f"datasets/development/rows/row_{int(index):06d}.npz"
        path = run / relative
        if not path.is_file():
            raise RuntimeError(f"accepted generation checkpoint is missing: {relative}")
        try:
            with np.load(path, allow_pickle=False) as stored:
                record = json.loads(str(stored["record_json"].item()))
                valid = bool(
                    np.array_equal(stored["decision"], decisions[index])
                    and np.array_equal(stored["influent"], influents[index])
                    and np.array_equal(stored["target"], targets[index])
                    and np.array_equal(stored["state_start_1"], states_start_1[index])
                    and np.array_equal(stored["state_start_2"], states_start_2[index])
                    and int(record.get("row", -1)) == int(index)
                    and record.get("accepted") is True
                )
                row_contract_hash = str(stored["contract_hash"].item())
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot validate accepted generation checkpoint: {relative}"
            ) from exc
        if not valid or not row_contract_hash:
            raise RuntimeError(
                f"accepted generation checkpoint differs from pinned aggregates: {relative}"
            )
        row_contract_hashes.add(row_contract_hash)
        checkpoints.append({
            "original_candidate_index": int(index),
            "accepted_slot": int(index),
            "path": relative,
            "sha256": file_digest(path),
        })
    checkpoint_set_digest = _canonical_json_digest(checkpoints)
    return {
        "schema": 1,
        "block": "development",
        "original_row_count": row_count,
        "retained_accepted_count": len(accepted_rows),
        "excluded_rejected_count": len(rejected_rows),
        "excluded_original_candidate_indices": rejected_rows.astype(int).tolist(),
        "pinned_artifacts": pinned,
        "row_contract_hashes": sorted(row_contract_hashes),
        "checkpoint_set_digest": checkpoint_set_digest,
        "checkpoints": checkpoints,
    }


def _source_snapshot_manifest(
    run: Path, relative_directory: str, expected_files: Mapping[str, str],
) -> dict[str, Any]:
    """Validate and describe an exact, path-contained predecessor source snapshot."""

    root = (run / relative_directory).resolve()
    migrations_root = (run / "inputs" / "contract_migrations").resolve()
    if migrations_root not in root.parents or not root.is_dir():
        raise RuntimeError("predecessor source snapshot is missing or outside migrations")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    expected_names = set(map(str, expected_files))
    if actual != expected_names:
        raise RuntimeError(
            "predecessor source snapshot file set differs from its source manifest"
        )
    observed: dict[str, str] = {}
    for relative, expected_digest in sorted(expected_files.items()):
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(f"predecessor source snapshot is missing {relative}")
        digest = file_digest(path)
        if digest != expected_digest:
            raise RuntimeError(f"predecessor source snapshot changed: {relative}")
        observed[str(relative)] = digest
    return {
        "directory": relative_directory,
        "source_digest": source_digest(observed),
        "source_files": observed,
    }


def _validate_recorded_source_snapshot(
    run: Path, snapshot: Any,
) -> None:
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("recorded predecessor source snapshot is invalid")
    files = snapshot.get("source_files")
    if not isinstance(files, Mapping):
        raise RuntimeError("recorded predecessor source snapshot omits its manifest")
    observed = _source_snapshot_manifest(
        run, str(snapshot.get("directory", "")), files,
    )
    if observed != dict(snapshot):
        raise RuntimeError("recorded predecessor source snapshot is inconsistent")


def _artifact_archive_manifest(
    run: Path,
    relative_directory: str,
    *,
    require_live_match: bool,
    require_marker_closure: bool = False,
) -> dict[str, Any]:
    """Validate an immutable archive of superseded casewise outputs."""

    root = (run / relative_directory).resolve()
    migrations_root = (run / "inputs" / "contract_migrations").resolve()
    if migrations_root not in root.parents or not root.is_dir():
        raise RuntimeError("retired casewise archive is missing or outside migrations")
    files = {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }
    required = {
        "metrics/case_common_reference_comparison.csv",
        "metrics/selected_candidate_reference_evaluation.csv",
        *(
            f"optimization/{case_id}/{name}"
            for case_id in (
                "nominal", *(f"robustness_{index:02d}" for index in range(1, 11))
            )
            for name in (
                "surrogate_local_convergence.json",
                "surrogate_local_convergence_complete.json",
                "surrogate_casewise_reference_complete.json",
                "direct_casewise_reference_complete.json",
                "casewise_comparison_complete.json",
            )
        ),
    }
    missing = required - set(files)
    if missing:
        raise RuntimeError(
            f"retired casewise archive omits required artifacts: {sorted(missing)}"
        )
    if require_live_match:
        for relative, expected_digest in files.items():
            live = (run / relative).resolve()
            if (
                run.resolve() not in live.parents
                or not live.is_file()
                or file_digest(live) != expected_digest
            ):
                raise RuntimeError(
                    f"retired casewise archive differs from live predecessor: {relative}"
                )
    if require_marker_closure:
        _validate_artifact_archive_marker_closure(root)
    manifest = {
        "directory": relative_directory,
        "file_count": len(files),
        "files": files,
    }
    if require_marker_closure:
        manifest["marker_closure_verified"] = True
    return manifest


def _validated_marker_artifact(
    relative: Any, expected_digest: Any,
) -> tuple[str, str]:
    """Normalize one marker edge without permitting path traversal."""

    if not isinstance(relative, str):
        raise RuntimeError("unsafe artifact reference")
    parts = relative.split("/")
    if (
        not relative
        or "\\" in relative
        or relative.startswith("/")
        or re.match(r"^[A-Za-z]:", relative) is not None
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RuntimeError(f"unsafe artifact reference: {relative!r}")
    if not isinstance(expected_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_digest,
    ) is None:
        raise RuntimeError("invalid SHA-256 digest")
    return relative, expected_digest


def _archive_artifact_candidates(
    archive_root: Path,
    marker_parent: Path,
    relative: str,
    expected_digest: str,
) -> tuple[Path, Path, Path]:
    candidates = (
        (archive_root / relative).resolve(),
        (marker_parent / relative).resolve(),
        (archive_root / "_artifact_blobs" / expected_digest).resolve(),
    )
    if any(archive_root not in candidate.parents for candidate in candidates):
        raise RuntimeError(f"unsafe artifact reference: {relative!r}")
    return candidates


def _validate_artifact_archive_marker_closure(root: Path) -> None:
    """Require the transitive artifact closure of every completion marker."""

    archive_root = root.resolve()
    unresolved: list[str] = []
    queue = [
        (marker.resolve(), marker.parent.resolve(), False)
        for marker in sorted(archive_root.rglob("*complete.json"))
    ]
    seen_markers: set[tuple[str, bool, str]] = set()
    while queue:
        marker, logical_parent, content_addressed = queue.pop()
        marker_label = (
            marker.relative_to(archive_root).as_posix()
            if archive_root in marker.parents
            else str(marker)
        )
        try:
            marker_digest = file_digest(marker)
        except OSError as exc:
            unresolved.append(f"{marker_label}: {exc}")
            continue
        marker_key = (
            marker_digest,
            content_addressed,
            "_artifact_blobs"
            if content_addressed
            else logical_parent.relative_to(archive_root).as_posix(),
        )
        if marker_key in seen_markers:
            continue
        seen_markers.add(marker_key)
        try:
            payload = _load_json_object(marker, description="archived completion marker")
        except RuntimeError as exc:
            unresolved.append(f"{marker_label}: {exc}")
            continue
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue
        for raw_relative, raw_digest in artifacts.items():
            try:
                relative, expected_digest = _validated_marker_artifact(
                    raw_relative, raw_digest,
                )
                normal_root, normal_local, blob = _archive_artifact_candidates(
                    archive_root, logical_parent, relative, expected_digest,
                )
            except RuntimeError as exc:
                unresolved.append(f"{marker_label} -> {raw_relative}: {exc}")
                continue
            candidates = (
                (blob,) if content_addressed
                else (normal_root, normal_local, blob)
            )
            matched = next((
                candidate
                for candidate in candidates
                if candidate.is_file() and file_digest(candidate) == expected_digest
            ), None)
            if matched is None:
                unresolved.append(f"{marker_label} -> {relative}")
                continue
            if relative.endswith("complete.json"):
                queue.append((
                    matched,
                    matched.parent if matched != blob else logical_parent,
                    matched == blob,
                ))
    if unresolved:
        raise RuntimeError(
            "retired casewise archive has unresolved completion-marker "
            f"artifacts: {unresolved[:12]}"
        )


def _copy_artifact_archive_marker_closure(
    source_run: Path,
    target_run: Path,
    relative_directory: str,
) -> dict[str, str]:
    """Complete a copied archive from the source run without mutating it."""

    source_archive = (source_run / relative_directory).resolve()
    target_archive = (target_run / relative_directory).resolve()
    source_root = source_run.resolve()
    target_root = target_run.resolve()
    if (
        source_root not in source_archive.parents
        or target_root not in target_archive.parents
        or not source_archive.is_dir()
        or not target_archive.is_dir()
    ):
        raise RuntimeError("retired marker archive is missing or outside its run")
    copied: dict[str, str] = {}
    digest_index: dict[str, Path] | None = None
    queue = [
        (
            marker.resolve(),
            (
                target_archive
                / marker.relative_to(source_archive).parent
            ).resolve(),
            False,
        )
        for marker in sorted(source_archive.rglob("*complete.json"))
    ]
    seen_markers: set[tuple[str, bool, str]] = set()
    while queue:
        marker, logical_parent, content_addressed = queue.pop()
        marker_digest = file_digest(marker)
        marker_key = (
            marker_digest,
            content_addressed,
            "_artifact_blobs"
            if content_addressed
            else logical_parent.relative_to(target_archive).as_posix(),
        )
        if marker_key in seen_markers:
            continue
        seen_markers.add(marker_key)
        payload = _load_json_object(marker, description="archived completion marker")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue
        for raw_relative, raw_digest in artifacts.items():
            relative, expected_digest = _validated_marker_artifact(
                raw_relative, raw_digest,
            )
            normal_root, normal_local, blob = _archive_artifact_candidates(
                target_archive, logical_parent, relative, expected_digest,
            )
            archive_candidates = (
                (blob,) if content_addressed
                else (normal_root, normal_local, blob)
            )
            matched = next((
                candidate
                for candidate in archive_candidates
                if candidate.is_file() and file_digest(candidate) == expected_digest
            ), None)
            if matched is None:
                source_candidates = [normal_root, normal_local]
                source_candidates.extend((
                    (source_run / relative).resolve(),
                    (marker.parent / relative).resolve(),
                ))
                try:
                    live_parent = (
                        source_run
                        / logical_parent.relative_to(target_archive)
                    ).resolve()
                except ValueError as exc:
                    raise RuntimeError(
                        "retired marker logical parent escaped its archive"
                    ) from exc
                source_candidates.append((live_parent / relative).resolve())
                selected = next((
                    source
                    for source in source_candidates
                    if (
                        (source_root in source.parents or target_root in source.parents)
                        and source.is_file()
                        and file_digest(source) == expected_digest
                    )
                ), None)
                if selected is None:
                    if digest_index is None:
                        digest_index = {}
                        for source in sorted(source_root.rglob("*")):
                            if source.is_file():
                                digest_index.setdefault(file_digest(source), source)
                    selected = digest_index.get(expected_digest)
                if selected is None:
                    raise RuntimeError(
                        "cannot close retired marker artifact from source run: "
                        f"{relative}"
                    )
                blob.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(selected, blob)
                if file_digest(blob) != expected_digest:
                    raise RuntimeError("retired marker closure copy failed verification")
                copied[blob.relative_to(target_run).as_posix()] = expected_digest
                matched = blob
            if relative.endswith("complete.json"):
                queue.append((
                    matched,
                    matched.parent if matched != blob else logical_parent,
                    matched == blob,
                ))
    _validate_artifact_archive_marker_closure(target_archive)
    return copied


def _validate_recorded_artifact_archive(run: Path, archive: Any) -> None:
    if not isinstance(archive, Mapping):
        raise RuntimeError("recorded retired casewise archive is invalid")
    observed = _artifact_archive_manifest(
        run,
        str(archive.get("directory", "")),
        require_live_match=False,
        require_marker_closure=bool(archive.get("marker_closure_verified", False)),
    )
    if observed != dict(archive):
        raise RuntimeError("recorded retired casewise archive is inconsistent")


def _file_archive_manifest(run: Path, relative_directory: str) -> dict[str, Any]:
    """Hash every file in a path-contained archive directory."""

    root = (run / relative_directory).resolve()
    migrations_root = (run / "inputs" / "contract_migrations").resolve()
    if migrations_root not in root.parents or not root.is_dir():
        raise RuntimeError("retired file archive is missing or outside migrations")
    files = {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }
    if not files:
        raise RuntimeError("retired file archive is empty")
    return {
        "directory": relative_directory,
        "file_count": len(files),
        "files": files,
    }


def _validate_recorded_file_archive(run: Path, archive: Any) -> None:
    if not isinstance(archive, Mapping):
        raise RuntimeError("recorded retired file archive is invalid")
    observed = _file_archive_manifest(run, str(archive.get("directory", "")))
    if observed != dict(archive):
        raise RuntimeError("recorded retired file archive is inconsistent")


def _validate_migration_history(run: Path, contract: Mapping[str, Any]) -> None:
    history = contract.get("contract_migrations")
    if not isinstance(history, list) or not history:
        raise RuntimeError("migrated contract has no migration history")
    seen: set[str] = set()
    prior_successor_digest: str | None = None
    prior_successor_schema: int | None = None
    for position, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            raise RuntimeError("contract migration history contains a non-object entry")
        migration_id = str(entry.get("migration_id", ""))
        relative = str(entry.get("record", ""))
        expected_digest = str(entry.get("record_digest", ""))
        if not migration_id or migration_id in seen:
            raise RuntimeError("contract migration history has duplicate/empty identifiers")
        seen.add(migration_id)
        path = (run / relative).resolve()
        migrations_root = (run / "inputs" / "contract_migrations").resolve()
        if migrations_root not in path.parents or not path.is_file():
            raise RuntimeError("contract migration record is missing or outside its directory")
        if file_digest(path) != expected_digest:
            raise RuntimeError(f"contract migration record changed: {migration_id}")
        predecessor_relative = str(entry.get("predecessor_contract", ""))
        predecessor_digest = str(entry.get("predecessor_contract_digest", ""))
        predecessor_path = (run / predecessor_relative).resolve()
        if (
            migrations_root not in predecessor_path.parents
            or not predecessor_path.is_file()
            or file_digest(predecessor_path) != predecessor_digest
        ):
            raise RuntimeError(
                f"archived predecessor contract changed: {migration_id}"
            )
        record = _load_json_object(path, description="contract migration record")
        predecessor = record.get("predecessor")
        successor = record.get("successor")
        if (
            record.get("migration_id") != migration_id
            or not isinstance(predecessor, Mapping)
            or not isinstance(successor, Mapping)
            or predecessor.get("archived_contract") != predecessor_relative
            or predecessor.get("archived_contract_digest") != predecessor_digest
            or predecessor.get("contract_file_digest") != predecessor_digest
            or entry.get("predecessor_source_digest")
            != predecessor.get("source_digest")
            or entry.get("successor_source_digest")
            != successor.get("source_digest")
        ):
            raise RuntimeError(f"contract migration record is inconsistent: {migration_id}")
        archived = _load_json_object(
            predecessor_path, description="archived predecessor contract",
        )
        archived_history = archived.get("contract_migrations", [])
        predecessor_files = predecessor.get("source_files")
        if (
            not isinstance(predecessor_files, Mapping)
            or source_digest(predecessor_files) != predecessor.get("source_digest")
            or archived.get("source_digest") != predecessor.get("source_digest")
            or archived.get("runner_schema") != predecessor.get("runner_schema")
            or archived.get("source_files") != predecessor.get("source_files")
            or archived_history != history[:position]
        ):
            raise RuntimeError(
                f"archived predecessor contract breaks migration chain: {migration_id}"
            )
        if position and (
            predecessor.get("source_digest") != prior_successor_digest
            or predecessor.get("runner_schema") != prior_successor_schema
        ):
            raise RuntimeError(f"contract migration chain is discontinuous: {migration_id}")
        prior_successor_digest = str(successor.get("source_digest", ""))
        successor_files = successor.get("source_files")
        if (
            not isinstance(successor_files, Mapping)
            or source_digest(successor_files) != prior_successor_digest
        ):
            raise RuntimeError(
                f"contract migration successor manifest is inconsistent: {migration_id}"
            )
        try:
            prior_successor_schema = int(successor.get("runner_schema", -1))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"contract migration record has invalid schema: {migration_id}"
            ) from exc
        snapshot = record.get("predecessor_source_snapshot")
        if snapshot is not None:
            _validate_recorded_source_snapshot(run, snapshot)
        retired_archive = record.get("retired_casewise_snapshot")
        if retired_archive is not None:
            _validate_recorded_artifact_archive(run, retired_archive)
        retired_timing = record.get("retired_inference_timing_snapshot")
        if retired_timing is not None:
            _validate_recorded_file_archive(run, retired_timing)
        for manifest_key in (
            "retained_checkpoint_manifest",
            "retained_stage_manifest",
            "reused_artifact_manifest",
        ):
            manifest_reference = record.get(manifest_key)
            if manifest_reference is None:
                continue
            if not isinstance(manifest_reference, Mapping):
                raise RuntimeError(
                    f"contract migration has invalid {manifest_key}: {migration_id}"
                )
            manifest_path = (run / str(manifest_reference.get("path", ""))).resolve()
            if (
                migrations_root not in manifest_path.parents
                or not manifest_path.is_file()
                or file_digest(manifest_path) != manifest_reference.get("sha256")
            ):
                raise RuntimeError(
                    f"contract migration retained manifest changed: {migration_id}"
                )
    if (
        prior_successor_digest != contract.get("source_digest")
        or prior_successor_schema != int(contract.get("runner_schema", -1))
        or dict(successor_files) != contract.get("source_files")
    ):
        raise RuntimeError("contract migration history does not reach current contract")


def _migrate_source_contract(
    run: Path,
    previous: Mapping[str, Any],
    successor: Mapping[str, Any],
    authorization: SourceContractMigrationAuthorization,
) -> None:
    """Apply one explicit, allow-listed migration and leave immutable evidence."""

    contract_path = run / "inputs" / "contract.json"
    if str(previous.get("run_id")) != authorization.run_id:
        raise RuntimeError("source-contract migration is not authorized for this run id")
    if previous.get("contract_migrations"):
        raise RuntimeError("source-contract migration is one-time and was already used")
    if (
        int(previous.get("runner_schema", -1))
        != authorization.predecessor_runner_schema
        or int(successor.get("runner_schema", -1))
        != authorization.successor_runner_schema
        or previous.get("source_digest") != authorization.predecessor_source_digest
        or file_digest(contract_path) != authorization.predecessor_contract_file_digest
    ):
        raise RuntimeError("existing contract is not the pinned migration predecessor")
    old_files = previous.get("source_files")
    new_files = successor.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("source-contract migration requires source-file manifests")
    if source_digest(old_files) != previous.get("source_digest"):
        raise RuntimeError("predecessor source manifest is internally inconsistent")
    if source_digest(new_files) != successor.get("source_digest"):
        raise RuntimeError("successor source manifest is internally inconsistent")
    changed = {
        name for name in set(old_files) | set(new_files)
        if old_files.get(name) != new_files.get(name)
    }
    unauthorized = changed - authorization.allowed_changed_source_files
    missing_required = authorization.required_changed_source_files - changed
    if unauthorized or missing_required:
        raise RuntimeError(
            "source-contract migration refused arbitrary source drift; "
            f"unauthorized={sorted(unauthorized)}, missing_required={sorted(missing_required)}"
        )
    ignored = {"runner_schema", "source_digest", "source_files", "contract_migrations"}
    old_invariants = {key: value for key, value in previous.items() if key not in ignored}
    new_invariants = {key: value for key, value in successor.items() if key not in ignored}
    if old_invariants != new_invariants:
        raise RuntimeError("source-contract migration cannot change the run/profile contract")

    retention = _validate_retained_generation_checkpoints(run, authorization)
    migrations_directory = run / "inputs" / "contract_migrations"
    predecessor_archive = (
        migrations_directory / f"{authorization.migration_id}-predecessor-contract.json"
    )
    retention_path = migrations_directory / f"{authorization.migration_id}-retained.json"
    record_path = migrations_directory / f"{authorization.migration_id}.json"
    predecessor_bytes = contract_path.read_bytes()
    if sha256(predecessor_bytes).hexdigest() != authorization.predecessor_contract_file_digest:
        raise RuntimeError("predecessor contract bytes changed during migration validation")
    if predecessor_archive.is_file():
        if predecessor_archive.read_bytes() != predecessor_bytes:
            raise RuntimeError("archived predecessor contract bytes are inconsistent")
    else:
        atomic_bytes(predecessor_archive, predecessor_bytes)
    normalized_retention = _json_ready(retention)
    if retention_path.is_file():
        if _load_json_object(
            retention_path, description="retained-checkpoint manifest",
        ) != normalized_retention:
            raise RuntimeError("existing retained-checkpoint manifest is inconsistent")
    else:
        atomic_json(retention_path, normalized_retention)
    record_core = {
        "schema": 1,
        "migration_id": authorization.migration_id,
        "authorized_date": authorization.authorized_date,
        "reason": authorization.reason,
        "predecessor": {
            "runner_schema": previous["runner_schema"],
            "source_digest": previous["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": authorization.predecessor_contract_file_digest,
            "archived_contract": predecessor_archive.relative_to(run).as_posix(),
            "archived_contract_digest": file_digest(predecessor_archive),
        },
        "successor": {
            "runner_schema": successor["runner_schema"],
            "source_digest": successor["source_digest"],
            "source_files": dict(new_files),
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(changed)
        },
        "retained_checkpoint_manifest": {
            "path": retention_path.relative_to(run).as_posix(),
            "sha256": file_digest(retention_path),
            "checkpoint_set_digest": retention["checkpoint_set_digest"],
            "retained_accepted_count": retention["retained_accepted_count"],
            "excluded_rejected_count": retention["excluded_rejected_count"],
        },
    }
    if record_path.is_file():
        record = _load_json_object(record_path, description="contract migration record")
        without_time = dict(record)
        without_time.pop("applied_at_utc", None)
        if without_time != record_core or "applied_at_utc" not in record:
            raise RuntimeError("existing contract migration journal is inconsistent")
    else:
        record = {
            **record_core,
            "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(record_path, record)
    history_entry = {
        "migration_id": authorization.migration_id,
        "record": record_path.relative_to(run).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_archive.relative_to(run).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_archive),
        "predecessor_source_digest": previous["source_digest"],
        "successor_source_digest": successor["source_digest"],
    }
    migrated = dict(successor)
    migrated["contract_migrations"] = [history_entry]
    atomic_json(contract_path, migrated)
    _validate_migration_history(run, migrated)


def _validate_retained_assessment_checkpoints(
    run: Path, authorization: AssessmentRecoveryMigrationAuthorization,
) -> dict[str, Any]:
    """Prove generation and ridge checkpoints are complete before carrying them forward."""

    pinned = _validate_pinned_artifacts(run, authorization.required_artifact_digests)
    with np.load(run / "datasets" / "design.npz", allow_pickle=False) as stored:
        design = {name: np.asarray(stored[name]) for name in stored.files}
    design["generators"] = _load_json_object(
        run / "inputs" / "generator_records.json",
        description="generator records",
    )
    validate_design(design, ARTICLE_FULL)
    design_id = _design_digest(design)

    blocks: dict[str, MechanisticBlockResult] = {}
    for block, count in (
        ("development", ARTICLE_FULL.development_count),
        ("test", ARTICLE_FULL.test_count),
    ):
        loaded = _load_generation_checkpoint(
            run, block=block, count=count, profile=ARTICLE_FULL,
            source_id=authorization.predecessor_source_digest,
            design_id=design_id,
        )
        if loaded is None:
            raise RuntimeError(
                f"pinned {block} generation checkpoint cannot be independently loaded"
            )
        blocks[block] = loaded[0]

    effective_path = run / "datasets" / "effective_design.npz"
    with np.load(effective_path, allow_pickle=False) as stored:
        if set(stored.files) != set(DESIGN_ARRAYS):
            raise RuntimeError("pinned effective design has unexpected arrays")
        effective = {name: np.asarray(stored[name]) for name in DESIGN_ARRAYS}
    expected_effective = {
        **{name: np.asarray(design[name]) for name in DESIGN_ARRAYS},
        "development_decisions": blocks["development"].decisions,
        "development_influents": blocks["development"].influents,
        "test_decisions": blocks["test"].decisions,
        "test_influents": blocks["test"].influents,
    }
    if any(
        not np.array_equal(effective[name], expected_effective[name])
        for name in DESIGN_ARRAYS
    ):
        raise RuntimeError("pinned effective design differs from accepted inputs")
    effective_id = _design_digest(effective)
    if effective_id != authorization.expected_effective_design_digest:
        raise RuntimeError("pinned effective-design digest is not authorized")
    effective_manifest_path = run / "datasets" / "effective_design_manifest.json"
    effective_manifest = _load_json_object(
        effective_manifest_path, description="effective design manifest",
    )
    if (
        effective_manifest.get("source_digest")
        != authorization.predecessor_source_digest
        or effective_manifest.get("base_design_digest") != design_id
        or effective_manifest.get("effective_design_digest") != effective_id
        or effective_manifest.get("artifact")
        != {effective_path.relative_to(run).as_posix(): file_digest(effective_path)}
    ):
        raise RuntimeError("pinned effective design manifest is inconsistent")

    ridge_input_id = _ridge_input_digest(
        effective["development_decisions"],
        effective["development_influents"],
        blocks["development"].targets,
    )
    if ridge_input_id != authorization.expected_ridge_input_digest:
        raise RuntimeError("pinned ridge input digest is not authorized")
    ridge = _load_ridge(
        run,
        decisions=effective["development_decisions"],
        influents=effective["development_influents"],
        targets=blocks["development"].targets,
        input_id=ridge_input_id,
        source_id=authorization.predecessor_source_digest,
    )
    if ridge is None:
        raise RuntimeError("pinned ridge checkpoint cannot be independently loaded")

    stage_paths = {
        "generation/development": run / "datasets/development/block_complete.json",
        "generation/test": run / "datasets/test/block_complete.json",
        "generation/effective_design": effective_manifest_path,
        "ridge": run / "models/ridge_complete.json",
    }
    stages: dict[str, Any] = {}
    for stage, path in stage_paths.items():
        payload = _load_json_object(path, description=f"{stage} checkpoint")
        stages[stage] = {
            "checkpoint": path.relative_to(run).as_posix(),
            "checkpoint_sha256": file_digest(path),
            "artifact_source_digest": authorization.predecessor_source_digest,
            "artifacts": dict(payload.get("artifacts", {})),
        }
    stages["generation/effective_design"]["artifacts"] = dict(
        effective_manifest["artifact"]
    )
    generation_summary_path = run / "metrics" / "mechanistic_generation_summary.csv"
    generation_summary = pd.read_csv(generation_summary_path)
    if (
        list(generation_summary.get("block", [])) != ["development", "test"]
        or list(generation_summary.get("accepted_rows", [])) != [4_000, 1_000]
    ):
        raise RuntimeError("pinned generation summary is inconsistent")
    stages["generation/summary"] = {
        "checkpoint": generation_summary_path.relative_to(run).as_posix(),
        "checkpoint_sha256": file_digest(generation_summary_path),
        "artifact_source_digest": authorization.predecessor_source_digest,
        "artifacts": {},
    }
    return {
        "schema": 1,
        "source_digest": authorization.predecessor_source_digest,
        "effective_design_digest": effective_id,
        "ridge_input_digest": ridge_input_id,
        "pinned_artifacts": pinned,
        "stages": stages,
    }


def _migrate_assessment_recovery_contract(
    run: Path,
    previous: Mapping[str, Any],
    successor: Mapping[str, Any],
    authorization: AssessmentRecoveryMigrationAuthorization,
) -> None:
    """Append the audited projection/gate-policy migration to an existing run."""

    contract_path = run / "inputs" / "contract.json"
    history = previous.get("contract_migrations")
    if str(previous.get("run_id")) != authorization.run_id:
        raise RuntimeError("assessment-recovery migration is not authorized for this run id")
    if not isinstance(history, list):
        raise RuntimeError("assessment-recovery migration requires prior migration history")
    history_ids = tuple(str(entry.get("migration_id", "")) for entry in history)
    if history_ids != authorization.required_prior_migration_ids:
        raise RuntimeError("assessment-recovery migration has unexpected prior history")
    _validate_migration_history(run, previous)
    if (
        int(previous.get("runner_schema", -1))
        != authorization.predecessor_runner_schema
        or int(successor.get("runner_schema", -1))
        != authorization.successor_runner_schema
        or previous.get("source_digest") != authorization.predecessor_source_digest
        or file_digest(contract_path) != authorization.predecessor_contract_file_digest
    ):
        raise RuntimeError("existing contract is not the pinned assessment predecessor")
    old_files = previous.get("source_files")
    new_files = successor.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("assessment-recovery migration requires source manifests")
    if source_digest(old_files) != previous.get("source_digest"):
        raise RuntimeError("predecessor source manifest is internally inconsistent")
    if source_digest(new_files) != successor.get("source_digest"):
        raise RuntimeError("successor source manifest is internally inconsistent")
    changed = {
        name for name in set(old_files) | set(new_files)
        if old_files.get(name) != new_files.get(name)
    }
    unauthorized = changed - authorization.allowed_changed_source_files
    missing_required = authorization.required_changed_source_files - changed
    if unauthorized or missing_required:
        raise RuntimeError(
            "assessment-recovery migration refused arbitrary source drift; "
            f"unauthorized={sorted(unauthorized)}, missing_required={sorted(missing_required)}"
        )
    ignored = {
        "runner_schema", "source_digest", "source_files", "contract_migrations",
        "assessment_gate_execution_policy",
    }
    old_invariants = {key: value for key, value in previous.items() if key not in ignored}
    new_invariants = {key: value for key, value in successor.items() if key not in ignored}
    if old_invariants != new_invariants:
        raise RuntimeError("assessment-recovery migration cannot change run/profile data")
    if (
        "assessment_gate_execution_policy" in previous
        or successor.get("assessment_gate_execution_policy")
        != ASSESSMENT_GATE_EXECUTION_POLICY
    ):
        raise RuntimeError("assessment-recovery gate-policy transition is invalid")

    source_snapshot = _source_snapshot_manifest(
        run, authorization.predecessor_source_snapshot, old_files,
    )
    if source_snapshot["source_digest"] != authorization.predecessor_source_digest:
        raise RuntimeError("predecessor source snapshot has the wrong aggregate digest")
    retention = _validate_retained_assessment_checkpoints(run, authorization)
    migrations_directory = run / "inputs" / "contract_migrations"
    predecessor_archive = (
        migrations_directory / f"{authorization.migration_id}-predecessor-contract.json"
    )
    retention_path = migrations_directory / f"{authorization.migration_id}-retained.json"
    record_path = migrations_directory / f"{authorization.migration_id}.json"
    predecessor_bytes = contract_path.read_bytes()
    if sha256(predecessor_bytes).hexdigest() != authorization.predecessor_contract_file_digest:
        raise RuntimeError("predecessor contract bytes changed during migration validation")
    if predecessor_archive.is_file():
        if predecessor_archive.read_bytes() != predecessor_bytes:
            raise RuntimeError("archived predecessor contract bytes are inconsistent")
    else:
        atomic_bytes(predecessor_archive, predecessor_bytes)
    normalized_retention = _json_ready(retention)
    if retention_path.is_file():
        if _load_json_object(
            retention_path, description="retained-stage manifest",
        ) != normalized_retention:
            raise RuntimeError("existing retained-stage manifest is inconsistent")
    else:
        atomic_json(retention_path, normalized_retention)
    record_core = {
        "schema": 1,
        "migration_id": authorization.migration_id,
        "authorized_date": authorization.authorized_date,
        "reason": authorization.reason,
        "predecessor": {
            "runner_schema": previous["runner_schema"],
            "source_digest": previous["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": authorization.predecessor_contract_file_digest,
            "archived_contract": predecessor_archive.relative_to(run).as_posix(),
            "archived_contract_digest": file_digest(predecessor_archive),
        },
        "predecessor_source_snapshot": source_snapshot,
        "successor": {
            "runner_schema": successor["runner_schema"],
            "source_digest": successor["source_digest"],
            "source_files": dict(new_files),
            "assessment_gate_execution_policy": ASSESSMENT_GATE_EXECUTION_POLICY,
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(changed)
        },
        "retained_stage_manifest": {
            "path": retention_path.relative_to(run).as_posix(),
            "sha256": file_digest(retention_path),
            "source_digest": retention["source_digest"],
            "effective_design_digest": retention["effective_design_digest"],
            "ridge_input_digest": retention["ridge_input_digest"],
        },
    }
    if record_path.is_file():
        record = _load_json_object(record_path, description="contract migration record")
        without_time = dict(record)
        without_time.pop("applied_at_utc", None)
        if without_time != record_core or "applied_at_utc" not in record:
            raise RuntimeError("existing assessment migration journal is inconsistent")
    else:
        record = {**record_core, "applied_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(record_path, record)
    history_entry = {
        "migration_id": authorization.migration_id,
        "record": record_path.relative_to(run).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_archive.relative_to(run).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_archive),
        "predecessor_source_digest": previous["source_digest"],
        "successor_source_digest": successor["source_digest"],
    }
    migrated = dict(successor)
    migrated["contract_migrations"] = [*history, history_entry]
    atomic_json(contract_path, migrated)
    _validate_migration_history(run, migrated)


def _validate_retained_optimization_protocol_checkpoints(
    run: Path,
    previous: Mapping[str, Any],
    authorization: OptimizationProtocolMigrationAuthorization,
) -> dict[str, Any]:
    """Prove all completed pre-optimization stages before carrying them forward."""

    pinned = _validate_pinned_artifacts(run, authorization.required_artifact_digests)
    history = previous.get("contract_migrations")
    if not isinstance(history, list) or not history:
        raise RuntimeError("optimization migration requires prior migration history")
    latest = history[-1]
    latest_record = _load_json_object(
        run / str(latest.get("record", "")),
        description="latest predecessor migration record",
    )
    retained_reference = latest_record.get("retained_stage_manifest")
    if not isinstance(retained_reference, Mapping):
        raise RuntimeError("predecessor migration omits its retained-stage manifest")
    retained_path = (run / str(retained_reference.get("path", ""))).resolve()
    migrations_root = (run / "inputs" / "contract_migrations").resolve()
    if (
        migrations_root not in retained_path.parents
        or not retained_path.is_file()
        or file_digest(retained_path) != retained_reference.get("sha256")
    ):
        raise RuntimeError("predecessor retained-stage manifest changed")
    prior_retention = _load_json_object(
        retained_path, description="predecessor retained-stage manifest",
    )
    prior_stages = prior_retention.get("stages")
    if not isinstance(prior_stages, Mapping):
        raise RuntimeError("predecessor retained-stage manifest omits stages")
    required_prior_stages = (
        "generation/development",
        "generation/test",
        "generation/effective_design",
        "generation/summary",
        "ridge",
    )
    stages: dict[str, Any] = {}
    run_root = run.resolve()
    for stage in required_prior_stages:
        stage_record = prior_stages.get(stage)
        if not isinstance(stage_record, Mapping):
            raise RuntimeError(f"predecessor retained manifest omits {stage}")
        checkpoint = (run / str(stage_record.get("checkpoint", ""))).resolve()
        if (
            run_root not in checkpoint.parents
            or not checkpoint.is_file()
            or file_digest(checkpoint) != stage_record.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"retained {stage} checkpoint changed")
        artifacts = stage_record.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(f"retained {stage} artifact manifest is invalid")
        if artifacts and not _artifacts_match(run, artifacts):
            raise RuntimeError(f"retained {stage} artifacts changed")
        artifact_source = str(stage_record.get("artifact_source_digest", ""))
        if not artifact_source:
            raise RuntimeError(f"retained {stage} omits its source binding")
        stages[stage] = dict(stage_record)

    marker_path = run / "metrics" / "assessment_complete.json"
    marker = _load_json_object(marker_path, description="assessment checkpoint")
    marker_artifacts = marker.get("artifacts")
    if (
        marker.get("source_digest") != authorization.predecessor_source_digest
        or marker.get("input_digest")
        != authorization.expected_assessment_input_digest
        or not isinstance(marker.get("passed"), bool)
        or not isinstance(marker_artifacts, Mapping)
        or not _artifacts_match(run, marker_artifacts)
    ):
        raise RuntimeError("completed assessment checkpoint is not safely reusable")
    gate = _load_json_object(
        run / "metrics" / "admission_gate.json", description="assessment gate",
    )
    if (
        not isinstance(gate.get("passed"), bool)
        or gate.get("passed") != marker.get("passed")
        or gate.get("execution_policy") != ASSESSMENT_GATE_EXECUTION_POLICY
        or gate.get("optimization_permitted")
        != assessment_gate_allows_optimization(bool(gate.get("passed")))
    ):
        raise RuntimeError("completed assessment gate is inconsistent")
    stages["assessment"] = {
        "checkpoint": marker_path.relative_to(run).as_posix(),
        "checkpoint_sha256": file_digest(marker_path),
        "artifact_source_digest": authorization.predecessor_source_digest,
        "artifacts": dict(marker_artifacts),
    }
    return {
        "schema": 2,
        "predecessor_source_digest": authorization.predecessor_source_digest,
        "effective_design_digest": prior_retention.get("effective_design_digest"),
        "ridge_input_digest": prior_retention.get("ridge_input_digest"),
        "assessment_input_digest": authorization.expected_assessment_input_digest,
        "pinned_artifacts": pinned,
        "stages": stages,
    }


def _migrate_optimization_protocol_contract(
    run: Path,
    previous: Mapping[str, Any],
    successor: Mapping[str, Any],
    authorization: OptimizationProtocolMigrationAuthorization,
) -> None:
    """Append the single-start exact-QP protocol migration in place."""

    contract_path = run / "inputs" / "contract.json"
    history = previous.get("contract_migrations")
    if str(previous.get("run_id")) != authorization.run_id:
        raise RuntimeError("optimization migration is not authorized for this run id")
    if not isinstance(history, list):
        raise RuntimeError("optimization migration requires prior migration history")
    history_ids = tuple(str(entry.get("migration_id", "")) for entry in history)
    if history_ids != authorization.required_prior_migration_ids:
        raise RuntimeError("optimization migration has unexpected prior history")
    _validate_migration_history(run, previous)
    if (
        int(previous.get("runner_schema", -1))
        != authorization.predecessor_runner_schema
        or int(successor.get("runner_schema", -1))
        != authorization.successor_runner_schema
        or previous.get("source_digest") != authorization.predecessor_source_digest
        or file_digest(contract_path) != authorization.predecessor_contract_file_digest
    ):
        raise RuntimeError("existing contract is not the pinned optimization predecessor")
    old_files = previous.get("source_files")
    new_files = successor.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("optimization migration requires source manifests")
    if source_digest(old_files) != previous.get("source_digest"):
        raise RuntimeError("predecessor source manifest is internally inconsistent")
    if source_digest(new_files) != successor.get("source_digest"):
        raise RuntimeError("successor source manifest is internally inconsistent")
    changed = {
        name for name in set(old_files) | set(new_files)
        if old_files.get(name) != new_files.get(name)
    }
    unauthorized = changed - authorization.allowed_changed_source_files
    missing_required = authorization.required_changed_source_files - changed
    if unauthorized or missing_required:
        raise RuntimeError(
            "optimization migration refused arbitrary source drift; "
            f"unauthorized={sorted(unauthorized)}, missing_required={sorted(missing_required)}"
        )
    ignored = {
        "runner_schema", "source_digest", "source_files", "contract_migrations",
        "optimization_protocol",
    }
    old_invariants = {key: value for key, value in previous.items() if key not in ignored}
    new_invariants = {key: value for key, value in successor.items() if key not in ignored}
    if old_invariants != new_invariants:
        raise RuntimeError("optimization migration cannot change run/profile data")
    if (
        "optimization_protocol" in previous
        or successor.get("optimization_protocol") != OPTIMIZATION_PROTOCOL
    ):
        raise RuntimeError("optimization-protocol transition is invalid")

    source_snapshot = _source_snapshot_manifest(
        run, authorization.predecessor_source_snapshot, old_files,
    )
    if source_snapshot["source_digest"] != authorization.predecessor_source_digest:
        raise RuntimeError("predecessor source snapshot has the wrong aggregate digest")
    retention = _validate_retained_optimization_protocol_checkpoints(
        run, previous, authorization,
    )
    migrations_directory = run / "inputs" / "contract_migrations"
    predecessor_archive = (
        migrations_directory / f"{authorization.migration_id}-predecessor-contract.json"
    )
    retention_path = migrations_directory / f"{authorization.migration_id}-retained.json"
    record_path = migrations_directory / f"{authorization.migration_id}.json"
    predecessor_bytes = contract_path.read_bytes()
    if sha256(predecessor_bytes).hexdigest() != authorization.predecessor_contract_file_digest:
        raise RuntimeError("predecessor contract bytes changed during migration validation")
    if predecessor_archive.is_file():
        if predecessor_archive.read_bytes() != predecessor_bytes:
            raise RuntimeError("archived predecessor contract bytes are inconsistent")
    else:
        atomic_bytes(predecessor_archive, predecessor_bytes)
    normalized_retention = _json_ready(retention)
    if retention_path.is_file():
        if _load_json_object(
            retention_path, description="retained-stage manifest",
        ) != normalized_retention:
            raise RuntimeError("existing optimization retention manifest is inconsistent")
    else:
        atomic_json(retention_path, normalized_retention)
    record_core = {
        "schema": 1,
        "migration_id": authorization.migration_id,
        "authorized_date": authorization.authorized_date,
        "reason": authorization.reason,
        "predecessor": {
            "runner_schema": previous["runner_schema"],
            "source_digest": previous["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": authorization.predecessor_contract_file_digest,
            "archived_contract": predecessor_archive.relative_to(run).as_posix(),
            "archived_contract_digest": file_digest(predecessor_archive),
        },
        "predecessor_source_snapshot": source_snapshot,
        "successor": {
            "runner_schema": successor["runner_schema"],
            "source_digest": successor["source_digest"],
            "source_files": dict(new_files),
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(changed)
        },
        "retained_stage_manifest": {
            "path": retention_path.relative_to(run).as_posix(),
            "sha256": file_digest(retention_path),
            "predecessor_source_digest": retention["predecessor_source_digest"],
            "assessment_input_digest": retention["assessment_input_digest"],
        },
    }
    if record_path.is_file():
        record = _load_json_object(record_path, description="contract migration record")
        without_time = dict(record)
        without_time.pop("applied_at_utc", None)
        if without_time != record_core or "applied_at_utc" not in record:
            raise RuntimeError("existing optimization migration journal is inconsistent")
    else:
        record = {**record_core, "applied_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(record_path, record)
    history_entry = {
        "migration_id": authorization.migration_id,
        "record": record_path.relative_to(run).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_archive.relative_to(run).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_archive),
        "predecessor_source_digest": previous["source_digest"],
        "successor_source_digest": successor["source_digest"],
    }
    migrated = dict(successor)
    migrated["contract_migrations"] = [*history, history_entry]
    atomic_json(contract_path, migrated)
    _validate_migration_history(run, migrated)


def _validate_retained_casewise_comparison_checkpoints(
    run: Path,
    previous: Mapping[str, Any],
    authorization: CasewiseComparisonMigrationAuthorization,
) -> dict[str, Any]:
    """Prove that all completed searches can be reused without recomputation."""

    pinned = _validate_pinned_artifacts(run, authorization.required_artifact_digests)
    history = previous.get("contract_migrations")
    if not isinstance(history, list) or not history:
        raise RuntimeError("casewise migration requires prior migration history")
    latest = history[-1]
    latest_record = _load_json_object(
        run / str(latest.get("record", "")),
        description="predecessor migration record",
    )
    retained_reference = latest_record.get("retained_stage_manifest")
    if not isinstance(retained_reference, Mapping):
        raise RuntimeError("predecessor migration omits its retained-stage manifest")
    retained_path = (run / str(retained_reference.get("path", ""))).resolve()
    migrations_root = (run / "inputs" / "contract_migrations").resolve()
    if (
        migrations_root not in retained_path.parents
        or not retained_path.is_file()
        or file_digest(retained_path) != retained_reference.get("sha256")
    ):
        raise RuntimeError("predecessor retained-stage manifest changed")
    prior_retention = _load_json_object(
        retained_path, description="predecessor retained-stage manifest",
    )
    prior_stages = prior_retention.get("stages")
    if not isinstance(prior_stages, Mapping):
        raise RuntimeError("predecessor retained-stage manifest omits stages")
    stages = {str(key): dict(value) for key, value in prior_stages.items()}

    cases = ("nominal", *(f"robustness_{index:02d}" for index in range(1, 11)))
    marker_records: list[dict[str, Any]] = []
    for case_id in cases:
        case_directory = run / "optimization" / case_id
        marker_path = case_directory / "case_complete.json"
        marker = _load_json_object(marker_path, description=f"{case_id} case marker")
        if (
            marker.get("stage") != "optimization_case"
            or marker.get("case") != case_id
            or int(marker.get("selected_decision_count", -1)) not in (1, 2)
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            raise RuntimeError(f"completed optimization case changed: {case_id}")
        routes = marker.get("routes")
        if not isinstance(routes, list) or len(routes) != 2:
            raise RuntimeError(f"completed optimization case omits routes: {case_id}")
        route_lookup = {str(item.get("route")): item for item in routes}
        if set(route_lookup) != {"surrogate", "direct"}:
            raise RuntimeError(f"completed optimization case has invalid routes: {case_id}")
        if route_lookup["surrogate"].get("protocol") != EXACT_QP_SINGLE_START_PROTOCOL:
            raise RuntimeError(f"retained surrogate protocol changed: {case_id}")
        if route_lookup["direct"].get("protocol") != DIRECT_SINGLE_CENTER_PROTOCOL:
            raise RuntimeError(f"retained direct protocol changed: {case_id}")
        for route, protocol in (
            ("surrogate", EXACT_QP_SINGLE_START_PROTOCOL),
            ("direct", DIRECT_SINGLE_CENTER_PROTOCOL),
        ):
            route_marker = _load_json_object(
                case_directory / f"{route}_complete.json",
                description=f"{case_id} {route} completion marker",
            )
            if (
                route_marker.get("protocol") != protocol
                or int(route_marker.get("start_count", -1)) != 1
                or not _artifacts_match(case_directory, route_marker.get("artifacts", {}))
            ):
                raise RuntimeError(f"retained {case_id} {route} route changed")
        relative_marker = marker_path.relative_to(run).as_posix()
        marker_records.append({
            "case": case_id,
            "sha256": file_digest(marker_path),
            "artifact_count": len(marker["artifacts"]),
            "artifacts_match": True,
        })
        stage_name = f"optimization/{case_id}"
        existing_stage = stages.get(stage_name)
        if isinstance(existing_stage, Mapping):
            if (
                existing_stage.get("checkpoint") != relative_marker
                or existing_stage.get("checkpoint_sha256") != file_digest(marker_path)
                or existing_stage.get("artifacts") != dict(marker["artifacts"])
            ):
                raise RuntimeError(f"retained optimization stage changed: {case_id}")
            stages[stage_name] = dict(existing_stage)
        else:
            stages[stage_name] = {
                "checkpoint": relative_marker,
                "checkpoint_sha256": file_digest(marker_path),
                "artifact_source_digest": authorization.predecessor_source_digest,
                "artifacts": dict(marker["artifacts"]),
            }
    marker_set_digest = _canonical_json_digest(marker_records)
    if marker_set_digest != authorization.expected_case_marker_set_digest:
        raise RuntimeError(
            "completed optimization case-marker set differs from the pinned predecessor"
        )
    return {
        "schema": 3,
        "predecessor_source_digest": authorization.predecessor_source_digest,
        "effective_design_digest": prior_retention.get("effective_design_digest"),
        "ridge_input_digest": prior_retention.get("ridge_input_digest"),
        "assessment_input_digest": prior_retention.get("assessment_input_digest"),
        "case_marker_set_digest": marker_set_digest,
        "pinned_artifacts": pinned,
        "retired_unfinished_stage": "validation/untouched_test_equivalence",
        "stages": stages,
    }


def _migrate_casewise_comparison_contract(
    run: Path,
    previous: Mapping[str, Any],
    successor: Mapping[str, Any],
    authorization: CasewiseComparisonMigrationAuthorization,
) -> None:
    """Append the casewise exact-reference comparison migration in place."""

    contract_path = run / "inputs" / "contract.json"
    history = previous.get("contract_migrations")
    if str(previous.get("run_id")) != authorization.run_id:
        raise RuntimeError("casewise migration is not authorized for this run id")
    if not isinstance(history, list):
        raise RuntimeError("casewise migration requires prior migration history")
    history_ids = tuple(str(entry.get("migration_id", "")) for entry in history)
    if history_ids != authorization.required_prior_migration_ids:
        raise RuntimeError("casewise migration has unexpected prior history")
    _validate_migration_history(run, previous)
    if (
        int(previous.get("runner_schema", -1))
        != authorization.predecessor_runner_schema
        or int(successor.get("runner_schema", -1))
        != authorization.successor_runner_schema
        or previous.get("source_digest") != authorization.predecessor_source_digest
        or file_digest(contract_path) != authorization.predecessor_contract_file_digest
    ):
        raise RuntimeError("existing contract is not the pinned casewise predecessor")
    old_files = previous.get("source_files")
    new_files = successor.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("casewise migration requires source manifests")
    if source_digest(old_files) != previous.get("source_digest"):
        raise RuntimeError("predecessor source manifest is internally inconsistent")
    if source_digest(new_files) != successor.get("source_digest"):
        raise RuntimeError("successor source manifest is internally inconsistent")
    changed = {
        name for name in set(old_files) | set(new_files)
        if old_files.get(name) != new_files.get(name)
    }
    unauthorized = changed - authorization.allowed_changed_source_files
    missing_required = authorization.required_changed_source_files - changed
    if unauthorized or missing_required:
        raise RuntimeError(
            "casewise migration refused arbitrary source drift; "
            f"unauthorized={sorted(unauthorized)}, missing_required={sorted(missing_required)}"
        )
    ignored = {
        "runner_schema", "source_digest", "source_files", "contract_migrations",
        "validation_protocol", "timing_protocol",
    }
    old_invariants = {key: value for key, value in previous.items() if key not in ignored}
    new_invariants = {key: value for key, value in successor.items() if key not in ignored}
    if old_invariants != new_invariants:
        raise RuntimeError("casewise migration cannot change run/profile data")
    if (
        authorization.migration_id
        == CASEWISE_TIMING_AGGREGATION_MIGRATION.migration_id
    ):
        expected_predecessor_protocol = COMPARISON_PROTOCOL
        target_protocol = COMPARISON_PROTOCOL
        if (
            previous.get("timing_protocol") is not None
            or successor.get("timing_protocol") != TIMING_PROTOCOL
        ):
            raise RuntimeError("casewise timing-protocol transition is invalid")
    elif (
        authorization.migration_id
        == CONVERGENCE_POLL_REFINEMENT_MIGRATION.migration_id
    ):
        expected_predecessor_protocol = "casewise_exact_common_reference_v1"
        target_protocol = "casewise_exact_common_reference_v2"
    else:
        expected_predecessor_protocol = None
        target_protocol = "casewise_exact_common_reference_v1"
    if (
        previous.get("validation_protocol") != expected_predecessor_protocol
        or successor.get("validation_protocol") != target_protocol
    ):
        raise RuntimeError("casewise validation-protocol transition is invalid")

    source_snapshot = _source_snapshot_manifest(
        run, authorization.predecessor_source_snapshot, old_files,
    )
    if source_snapshot["source_digest"] != authorization.predecessor_source_digest:
        raise RuntimeError("predecessor source snapshot has the wrong aggregate digest")
    retention = _validate_retained_casewise_comparison_checkpoints(
        run, previous, authorization,
    )
    if (
        authorization.migration_id
        == CASEWISE_TIMING_AGGREGATION_MIGRATION.migration_id
    ):
        retention["retired_unfinished_stage"] = "inference_timing"
        retention["retained_casewise_source_digest"] = previous["source_digest"]
    retired_casewise_snapshot = (
        None
        if authorization.retired_casewise_snapshot is None
        else _artifact_archive_manifest(
            run,
            authorization.retired_casewise_snapshot,
            require_live_match=True,
        )
    )
    retired_timing_snapshot = None
    live_timing_directory = run / "timing"
    if (
        authorization.migration_id
        == CASEWISE_TIMING_AGGREGATION_MIGRATION.migration_id
    ):
        retired_timing_relative = (
            "inputs/contract_migrations/"
            "article-v3-casewise-timing-v1-retired-inference-timing"
        )
        retired_timing_directory = run / retired_timing_relative
        if live_timing_directory.is_dir() and not retired_timing_directory.exists():
            shutil.copytree(live_timing_directory, retired_timing_directory)
        if retired_timing_directory.is_dir():
            retired_timing_snapshot = _file_archive_manifest(
                run, retired_timing_relative,
            )
    migrations_directory = run / "inputs" / "contract_migrations"
    predecessor_archive = (
        migrations_directory / f"{authorization.migration_id}-predecessor-contract.json"
    )
    retention_path = migrations_directory / f"{authorization.migration_id}-retained.json"
    record_path = migrations_directory / f"{authorization.migration_id}.json"
    predecessor_bytes = contract_path.read_bytes()
    if sha256(predecessor_bytes).hexdigest() != authorization.predecessor_contract_file_digest:
        raise RuntimeError("predecessor contract bytes changed during migration validation")
    if predecessor_archive.is_file():
        if predecessor_archive.read_bytes() != predecessor_bytes:
            raise RuntimeError("archived predecessor contract bytes are inconsistent")
    else:
        atomic_bytes(predecessor_archive, predecessor_bytes)
    normalized_retention = _json_ready(retention)
    if retention_path.is_file():
        if _load_json_object(
            retention_path, description="retained-stage manifest",
        ) != normalized_retention:
            raise RuntimeError("existing casewise retention manifest is inconsistent")
    else:
        atomic_json(retention_path, normalized_retention)
    record_core = {
        "schema": 1,
        "migration_id": authorization.migration_id,
        "authorized_date": authorization.authorized_date,
        "reason": authorization.reason,
        "predecessor": {
            "runner_schema": previous["runner_schema"],
            "source_digest": previous["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": authorization.predecessor_contract_file_digest,
            "archived_contract": predecessor_archive.relative_to(run).as_posix(),
            "archived_contract_digest": file_digest(predecessor_archive),
        },
        "predecessor_source_snapshot": source_snapshot,
        "retired_casewise_snapshot": retired_casewise_snapshot,
        **(
            {"retired_inference_timing_snapshot": retired_timing_snapshot}
            if retired_timing_snapshot is not None else {}
        ),
        "successor": {
            "runner_schema": successor["runner_schema"],
            "source_digest": successor["source_digest"],
            "source_files": dict(new_files),
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "validation_protocol": target_protocol,
            **(
                {"timing_protocol": successor.get("timing_protocol")}
                if authorization.migration_id
                == CASEWISE_TIMING_AGGREGATION_MIGRATION.migration_id
                else {}
            ),
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(changed)
        },
        "retained_stage_manifest": {
            "path": retention_path.relative_to(run).as_posix(),
            "sha256": file_digest(retention_path),
            "predecessor_source_digest": retention["predecessor_source_digest"],
            "assessment_input_digest": retention["assessment_input_digest"],
            "case_marker_set_digest": retention["case_marker_set_digest"],
        },
    }
    if record_path.is_file():
        record = _load_json_object(record_path, description="contract migration record")
        without_time = dict(record)
        without_time.pop("applied_at_utc", None)
        if without_time != record_core or "applied_at_utc" not in record:
            raise RuntimeError("existing casewise migration journal is inconsistent")
    else:
        record = {**record_core, "applied_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(record_path, record)
    history_entry = {
        "migration_id": authorization.migration_id,
        "record": record_path.relative_to(run).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_archive.relative_to(run).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_archive),
        "predecessor_source_digest": previous["source_digest"],
        "successor_source_digest": successor["source_digest"],
    }
    migrated = dict(successor)
    migrated["contract_migrations"] = [*history, history_entry]
    atomic_json(contract_path, migrated)
    _validate_migration_history(run, migrated)
    if (
        authorization.migration_id
        == CASEWISE_TIMING_AGGREGATION_MIGRATION.migration_id
        and live_timing_directory.is_dir()
    ):
        shutil.rmtree(live_timing_directory)


def _migrate_fresh_route_loader_fix_contract(
    run: Path,
    previous: Mapping[str, Any],
    successor: Mapping[str, Any],
) -> None:
    """Journal the fresh-route loader fix while retaining completed assessment.

    This migration is intentionally pinned to the sampled 10k run that exposed
    the bug.  It permits only the driver source file to change and independently
    binds every checkpoint needed to resume immediately before optimization.
    """

    contract_path = run / "inputs" / "contract.json"
    if str(previous.get("run_id")) != FRESH_ROUTE_LOADER_FIX_RUN_ID:
        raise RuntimeError("fresh-route loader migration is not authorized for this run")
    if previous.get("contract_migrations"):
        raise RuntimeError("fresh-route loader migration requires an unmigrated run")
    if (
        int(previous.get("runner_schema", -1)) != RUNNER_SCHEMA
        or int(successor.get("runner_schema", -1)) != RUNNER_SCHEMA
        or previous.get("source_digest")
        != FRESH_ROUTE_LOADER_FIX_PREDECESSOR_SOURCE_DIGEST
        or file_digest(contract_path)
        != FRESH_ROUTE_LOADER_FIX_PREDECESSOR_CONTRACT_DIGEST
    ):
        raise RuntimeError("run is not the pinned fresh-route loader predecessor")
    old_files = previous.get("source_files")
    new_files = successor.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("fresh-route loader migration requires source manifests")
    if (
        source_digest(old_files) != previous.get("source_digest")
        or source_digest(new_files) != successor.get("source_digest")
    ):
        raise RuntimeError("fresh-route loader source manifest is inconsistent")
    changed = {
        name for name in set(old_files) | set(new_files)
        if old_files.get(name) != new_files.get(name)
    }
    if changed != {"scripts/run_article_v3_5000.py"}:
        raise RuntimeError(
            "fresh-route loader migration permits only its driver correction; "
            f"changed={sorted(changed)}"
        )
    ignored = {"source_digest", "source_files", "contract_migrations"}
    if (
        {key: value for key, value in previous.items() if key not in ignored}
        != {key: value for key, value in successor.items() if key not in ignored}
    ):
        raise RuntimeError("fresh-route loader migration cannot change run data")

    migration_directory = run / "inputs" / "contract_migrations"
    snapshot_relative = (
        f"inputs/contract_migrations/{FRESH_ROUTE_LOADER_FIX_MIGRATION_ID}"
        "-predecessor-source"
    )
    source_snapshot = _source_snapshot_manifest(run, snapshot_relative, old_files)
    if source_snapshot["source_digest"] != previous["source_digest"]:
        raise RuntimeError("fresh-route loader predecessor snapshot has wrong digest")

    stage_specs = {
        "generation/development": run / "datasets/development/block_complete.json",
        "generation/test": run / "datasets/test/block_complete.json",
        "generation/effective_design": run / "datasets/effective_design_manifest.json",
        "generation/summary": run / "metrics/mechanistic_generation_summary.csv",
        "ridge": run / "models/ridge_complete.json",
        "assessment": run / "metrics/assessment_complete.json",
    }
    stages: dict[str, Any] = {}
    for stage, checkpoint in stage_specs.items():
        if not checkpoint.is_file():
            raise RuntimeError(f"fresh-route loader migration is missing {stage}")
        artifacts: Mapping[str, Any] = {}
        if checkpoint.suffix == ".json":
            marker = _load_json_object(checkpoint, description=f"retained {stage}")
            if marker.get("source_digest") != previous["source_digest"]:
                raise RuntimeError(f"retained {stage} has the wrong source binding")
            marker_artifacts = marker.get(
                "artifact" if stage == "generation/effective_design" else "artifacts",
                {},
            )
            if not isinstance(marker_artifacts, Mapping) or not _artifacts_match(
                run, marker_artifacts,
            ):
                raise RuntimeError(f"retained {stage} artifacts changed")
            artifacts = marker_artifacts
        stages[stage] = {
            "checkpoint": checkpoint.relative_to(run).as_posix(),
            "checkpoint_sha256": file_digest(checkpoint),
            "artifact_source_digest": previous["source_digest"],
            "artifacts": dict(artifacts),
        }
    retention = {
        "schema": 1,
        "reason": "resume after correcting fresh-run retained-route detection",
        "predecessor_source_digest": previous["source_digest"],
        "stages": stages,
    }

    predecessor_path = (
        migration_directory
        / f"{FRESH_ROUTE_LOADER_FIX_MIGRATION_ID}-predecessor-contract.json"
    )
    retention_path = (
        migration_directory / f"{FRESH_ROUTE_LOADER_FIX_MIGRATION_ID}-retained.json"
    )
    record_path = migration_directory / f"{FRESH_ROUTE_LOADER_FIX_MIGRATION_ID}.json"
    predecessor_bytes = contract_path.read_bytes()
    if sha256(predecessor_bytes).hexdigest() != FRESH_ROUTE_LOADER_FIX_PREDECESSOR_CONTRACT_DIGEST:
        raise RuntimeError("fresh-route loader predecessor contract changed")
    atomic_bytes(predecessor_path, predecessor_bytes)
    atomic_json(retention_path, retention)
    record = {
        "schema": 1,
        "migration_id": FRESH_ROUTE_LOADER_FIX_MIGRATION_ID,
        "authorized_date": "2026-08-26",
        "reason": (
            "User-authorized correction of fresh-run retained-route detection and "
            "continuation from completed sampled-data assessment checkpoints."
        ),
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor": {
            "runner_schema": previous["runner_schema"],
            "source_digest": previous["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": file_digest(predecessor_path),
            "archived_contract": predecessor_path.relative_to(run).as_posix(),
            "archived_contract_digest": file_digest(predecessor_path),
        },
        "predecessor_source_snapshot": source_snapshot,
        "successor": {
            "runner_schema": successor["runner_schema"],
            "source_digest": successor["source_digest"],
            "source_files": dict(new_files),
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(changed)
        },
        "retained_stage_manifest": {
            "path": retention_path.relative_to(run).as_posix(),
            "sha256": file_digest(retention_path),
            "predecessor_source_digest": previous["source_digest"],
        },
    }
    atomic_json(record_path, record)
    history_entry = {
        "migration_id": FRESH_ROUTE_LOADER_FIX_MIGRATION_ID,
        "record": record_path.relative_to(run).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_path.relative_to(run).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_path),
        "predecessor_source_digest": previous["source_digest"],
        "successor_source_digest": successor["source_digest"],
    }
    migrated = dict(successor)
    migrated["contract_migrations"] = [history_entry]
    atomic_json(contract_path, migrated)
    _validate_migration_history(run, migrated)


def establish_contract(
    run: Path,
    contract: Mapping[str, Any],
    *,
    authorize_generation_replacement_migration: bool = False,
    authorize_assessment_recovery_migration: bool = False,
    authorize_single_start_exact_qp_migration: bool = False,
    authorize_casewise_common_reference_migration: bool = False,
    authorize_convergence_poll_refinement_migration: bool = False,
    authorize_casewise_timing_migration: bool = False,
    authorize_fresh_route_loader_fix_migration: bool = False,
) -> None:
    if sum(map(bool, (
        authorize_generation_replacement_migration,
        authorize_assessment_recovery_migration,
        authorize_single_start_exact_qp_migration,
        authorize_casewise_common_reference_migration,
        authorize_convergence_poll_refinement_migration,
        authorize_casewise_timing_migration,
        authorize_fresh_route_loader_fix_migration,
    ))) > 1:
        raise ValueError("authorize exactly one source-contract migration at a time")
    path = run / "inputs" / "contract.json"
    normalized = _json_ready(contract)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous == normalized:
            return
        previous_without_history = dict(previous)
        previous_without_history.pop("contract_migrations", None)
        if previous_without_history == normalized:
            _validate_migration_history(run, previous)
            return
        if authorize_fresh_route_loader_fix_migration:
            _migrate_fresh_route_loader_fix_contract(run, previous, normalized)
            return
        if authorize_generation_replacement_migration:
            _migrate_source_contract(
                run, previous, normalized, GENERATION_REPLACEMENT_MIGRATION,
            )
            return
        if authorize_assessment_recovery_migration:
            _migrate_assessment_recovery_contract(
                run, previous, normalized, ASSESSMENT_RECOVERY_MIGRATION,
            )
            return
        if authorize_single_start_exact_qp_migration:
            _migrate_optimization_protocol_contract(
                run, previous, normalized, SINGLE_START_EXACT_QP_MIGRATION,
            )
            return
        if authorize_casewise_common_reference_migration:
            _migrate_casewise_comparison_contract(
                run, previous, normalized, CASEWISE_COMMON_REFERENCE_MIGRATION,
            )
            return
        if authorize_convergence_poll_refinement_migration:
            _migrate_casewise_comparison_contract(
                run,
                previous,
                normalized,
                CONVERGENCE_POLL_REFINEMENT_MIGRATION,
            )
            return
        if authorize_casewise_timing_migration:
            _migrate_casewise_comparison_contract(
                run,
                previous,
                normalized,
                CASEWISE_TIMING_AGGREGATION_MIGRATION,
            )
            return
        raise RuntimeError(
            "existing full-run contract differs; choose a new run id or provide the "
            "explicit authorized migration flag"
        )
    else:
        atomic_json(path, normalized)


def _validate_fork_reuse_manifest(
    run: Path,
    reference: Mapping[str, Any],
    *,
    verify_files: bool,
) -> dict[str, Any]:
    path = (run / str(reference.get("path", ""))).resolve()
    migrations_root = (run / "inputs" / "contract_migrations").resolve()
    if (
        migrations_root not in path.parents
        or not path.is_file()
        or file_digest(path) != reference.get("sha256")
    ):
        raise RuntimeError("run-reuse manifest is missing or changed")
    manifest = _load_json_object(path, description="run-reuse manifest")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("run-reuse manifest omits its copied-file ledger")
    if manifest.get("file_set_digest") != _canonical_json_digest(files):
        raise RuntimeError("run-reuse copied-file ledger digest is inconsistent")
    if verify_files:
        for relative, expected_digest in files.items():
            candidate = (run / str(relative)).resolve()
            if (
                run.resolve() not in candidate.parents
                or not candidate.is_file()
                or file_digest(candidate) != expected_digest
            ):
                raise RuntimeError(f"reused artifact changed or is missing: {relative}")
    return manifest


def _validate_existing_reused_run(
    run: Path,
    *,
    source_run_id: str,
    expected_contract: Mapping[str, Any],
) -> None:
    contract = _load_json_object(
        run / "inputs" / "contract.json", description="reused run contract",
    )
    without_history = dict(contract)
    without_history.pop("contract_migrations", None)
    if without_history != dict(expected_contract):
        raise RuntimeError("existing rerun directory has a different current contract")
    _validate_migration_history(run, contract)
    latest = contract["contract_migrations"][-1]
    record = _load_json_object(
        run / str(latest["record"]), description="latest run-fork record",
    )
    fork = record.get("run_fork")
    if (
        not isinstance(fork, Mapping)
        or fork.get("source_run_id") != source_run_id
        or fork.get("target_run_id") != run.name
    ):
        raise RuntimeError("existing rerun directory has different lineage")
    reference = record.get("reused_artifact_manifest")
    if not isinstance(reference, Mapping):
        raise RuntimeError("existing rerun contract omits its reuse manifest")
    manifest = _validate_fork_reuse_manifest(run, reference, verify_files=True)
    if (
        manifest.get("source_run_id") != source_run_id
        or manifest.get("target_run_id") != run.name
        or int(reference.get("file_count", -1)) != int(manifest.get("file_count", -2))
        or reference.get("file_set_digest") != manifest.get("file_set_digest")
    ):
        raise RuntimeError("existing rerun reuse-manifest metadata is inconsistent")


def _copy_reusable_files(
    source_run: Path,
    target_run: Path,
    retention: Mapping[str, Any],
) -> dict[str, str]:
    """Byte-copy immutable upstream and primary-search artifacts into a rerun."""

    relative_paths: set[str] = set()
    for top_level in ("datasets", "inputs/contract_migrations"):
        root = source_run / top_level
        if not root.is_dir():
            raise RuntimeError(f"reuse source omits required directory: {top_level}")
        relative_paths.update(
            path.relative_to(source_run).as_posix()
            for path in root.rglob("*") if path.is_file()
        )
    relative_paths.add("inputs/generator_records.json")
    stages = retention.get("stages")
    if not isinstance(stages, Mapping) or not stages:
        raise RuntimeError("run reuse requires a nonempty retained-stage manifest")
    for stage_name, stage in stages.items():
        if not isinstance(stage, Mapping):
            raise RuntimeError(f"retained stage is invalid: {stage_name}")
        checkpoint = str(stage.get("checkpoint", ""))
        artifacts = stage.get("artifacts", {})
        if not checkpoint or not isinstance(artifacts, Mapping):
            raise RuntimeError(f"retained stage is incomplete: {stage_name}")
        relative_paths.add(checkpoint)
        relative_paths.update(map(str, artifacts))

    source_root = source_run.resolve()
    target_root = target_run.resolve()
    copied: dict[str, str] = {}
    for relative in sorted(relative_paths):
        source = (source_run / relative).resolve()
        target = (target_run / relative).resolve()
        if (
            source_root not in source.parents
            or target_root not in target.parents
            or not source.is_file()
        ):
            raise RuntimeError(f"unsafe or missing reusable artifact: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        expected_digest = file_digest(source)
        if file_digest(target) != expected_digest:
            raise RuntimeError(f"reused artifact copy verification failed: {relative}")
        copied[relative] = expected_digest
    return copied


def _initialize_reduced_response_fork(
    run: Path,
    *,
    source_run: Path,
    source_contract: Mapping[str, Any],
    successor_contract: Mapping[str, Any],
    retain_ridge: bool = False,
) -> None:
    """Fork immutable generation evidence and, optionally, the unchanged ridge fit.

    Layer-resolved mechanistic checkpoints remain authoritative source data.
    When ``retain_ridge`` is false, every model is excluded.  When true, only
    the byte-identical Extended-ICSOR response fit is retained; closure,
    projection, assessment, optimization, replay, timing, and reports are
    intentionally excluded.
    """

    source_root = source_run.resolve()
    run.parent.mkdir(parents=True, exist_ok=True)
    temporary_owner = tempfile.TemporaryDirectory(prefix=".r-", dir=run.parent)
    temporary = Path(temporary_owner.name).resolve()
    copied: dict[str, str] = {}
    relative_paths: set[str] = set()
    datasets = source_run / "datasets"
    if not datasets.is_dir():
        raise RuntimeError("reduced-response fork source omits the datasets directory")
    for path in datasets.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_run).as_posix()
        if (
            relative in REDUCED_FORK_DATASET_FILES
            or any(
                relative.startswith(prefix)
                for prefix in REDUCED_FORK_DATASET_PREFIXES
            )
        ):
            relative_paths.add(relative)
    migrations = source_run / "inputs" / "contract_migrations"
    if migrations.is_dir():
        relative_paths.update(
            path.relative_to(source_run).as_posix()
            for path in migrations.rglob("*") if path.is_file()
        )
    for relative in REDUCED_FORK_INPUT_FILES:
        if (source_run / relative).is_file():
            relative_paths.add(relative)
    generation_summary = source_run / "metrics" / "mechanistic_generation_summary.csv"
    if generation_summary.is_file():
        relative_paths.add(generation_summary.relative_to(source_run).as_posix())
    if retain_ridge:
        relative_paths.update({
            "models/ridge_complete.json",
            "models/ridge_surrogate.npz",
            "metrics/ridge_cross_validation.csv",
            "metrics/ridge_fold_membership.csv",
        })
    for relative in sorted(relative_paths):
        source = (source_run / relative).resolve()
        target = (temporary / relative).resolve()
        if source_root not in source.parents or not source.is_file():
            raise RuntimeError(f"unsafe or missing generation artifact: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = file_digest(source)
        if file_digest(target) != digest:
            raise RuntimeError(f"generation-only copy verification failed: {relative}")
        copied[relative] = digest

    migrations_directory = temporary / "inputs" / "contract_migrations"
    migrations_directory.mkdir(parents=True, exist_ok=True)
    migration_id = f"article-v3-reduced-response-{run.name}"
    predecessor_path = migrations_directory / f"{migration_id}-predecessor-contract.json"
    atomic_bytes(predecessor_path, (source_run / "inputs" / "contract.json").read_bytes())

    stages: dict[str, Any] = {}
    for block in ("development", "test"):
        relative = f"datasets/{block}/block_complete.json"
        checkpoint = temporary / relative
        marker = _load_json_object(
            checkpoint, description=f"retained {block} generation marker",
        )
        if not _artifacts_match(temporary, marker.get("artifacts", {})):
            raise RuntimeError(f"retained {block} generation artifacts changed")
        stages[f"generation/{block}"] = {
            "artifact_source_digest": str(marker.get("source_digest", "")),
            "checkpoint": relative,
            "checkpoint_sha256": file_digest(checkpoint),
            "artifacts": dict(marker.get("artifacts", {})),
        }
    frozen_relative = "datasets/frozen_accepted_complete.json"
    frozen_checkpoint = temporary / frozen_relative
    if frozen_checkpoint.is_file():
        frozen_marker = _load_json_object(
            frozen_checkpoint, description="retained frozen accepted-data marker",
        )
        if not _artifacts_match(temporary, frozen_marker.get("artifacts", {})):
            raise RuntimeError("retained frozen accepted-data artifacts changed")
        stages["generation/frozen_accepted"] = {
            "artifact_source_digest": str(
                frozen_marker.get("source_digest", source_contract["source_digest"])
            ),
            "checkpoint": frozen_relative,
            "checkpoint_sha256": file_digest(frozen_checkpoint),
            "artifacts": dict(frozen_marker.get("artifacts", {})),
        }
    summary_relative = "metrics/mechanistic_generation_summary.csv"
    if (temporary / summary_relative).is_file():
        generation_source_ids = {
            str(stages[f"generation/{block}"]["artifact_source_digest"])
            for block in ("development", "test")
        }
        if len(generation_source_ids) != 1 or "" in generation_source_ids:
            raise RuntimeError(
                "retained generation blocks do not share one source binding"
            )
        stages["generation/summary"] = {
            "artifact_source_digest": next(iter(generation_source_ids)),
            "checkpoint": summary_relative,
            "checkpoint_sha256": file_digest(temporary / summary_relative),
            "artifacts": {},
        }
    effective_relative = "datasets/effective_design_manifest.json"
    if (temporary / effective_relative).is_file():
        effective_marker = _load_json_object(
            temporary / effective_relative,
            description="retained effective-design marker",
        )
        effective_source_id = str(effective_marker.get("source_digest", ""))
        if not effective_source_id:
            raise RuntimeError("retained effective-design marker omits its source binding")
        stages["generation/effective_design"] = {
            "artifact_source_digest": effective_source_id,
            "checkpoint": effective_relative,
            "checkpoint_sha256": file_digest(temporary / effective_relative),
            "artifacts": {},
        }
    if retain_ridge:
        ridge_relative = "models/ridge_complete.json"
        ridge_checkpoint = temporary / ridge_relative
        ridge_marker = _load_json_object(
            ridge_checkpoint, description="retained Extended-ICSOR ridge marker",
        )
        if not _artifacts_match(temporary, ridge_marker.get("artifacts", {})):
            raise RuntimeError("retained Extended-ICSOR ridge artifacts changed")
        stages["ridge"] = {
            "artifact_source_digest": str(ridge_marker.get("source_digest", "")),
            "checkpoint": ridge_relative,
            "checkpoint_sha256": file_digest(ridge_checkpoint),
            "artifacts": dict(ridge_marker.get("artifacts", {})),
        }
    retained = {
        "schema": 1,
        "policy": (
            "generation_and_unchanged_extended_icsor_fit_for_log_closure_v1"
            if retain_ridge
            else "generation_only_for_clarifier_inventory_response_v1"
        ),
        "predecessor_source_digest": source_contract["source_digest"],
        "stages": stages,
    }
    retained_path = migrations_directory / f"{migration_id}-retained.json"
    atomic_json(retained_path, retained)
    reuse_manifest = {
        "schema": 1,
        "copy_mode": (
            "independent_byte_copy_generation_and_ridge"
            if retain_ridge else "independent_byte_copy_generation_only"
        ),
        "source_run_id": source_run.name,
        "target_run_id": run.name,
        "source_contract_sha256": file_digest(source_run / "inputs" / "contract.json"),
        "file_count": len(copied),
        "file_set_digest": _canonical_json_digest(copied),
        "files": copied,
    }
    reuse_path = migrations_directory / f"{migration_id}-reused-files.json"
    atomic_json(reuse_path, reuse_manifest)
    old_files = source_contract.get("source_files")
    new_files = successor_contract.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("response-schema fork requires complete source manifests")
    record = {
        "schema": 1,
        "migration_id": migration_id,
        "authorized_date": "2026-08-25",
        "reason": (
            "User-authorized addition of the log-overflow-TSS projection closure "
            "while preserving the exact sampled dataset and unchanged trained "
            "Extended-ICSOR response model."
            if retain_ridge else
            "User-authorized replacement of layer-wise surrogate outputs by one "
            "clarifier-solids inventory while preserving full mechanistic checkpoints."
        ),
        "run_fork": {
            "source_run_id": source_run.name,
            "target_run_id": run.name,
            "self_contained": True,
            "recomputed_scope": (
                "closure_fit_projection_assessment_optimization_replay_timing_reporting"
                if retain_ridge else
                "response_transform_fit_assessment_optimization_replay_timing_reporting"
            ),
        },
        "predecessor": {
            "run_id": source_run.name,
            "runner_schema": source_contract["runner_schema"],
            "source_digest": source_contract["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": file_digest(predecessor_path),
            "archived_contract": predecessor_path.relative_to(temporary).as_posix(),
            "archived_contract_digest": file_digest(predecessor_path),
        },
        "successor": {
            "run_id": run.name,
            "runner_schema": successor_contract["runner_schema"],
            "source_digest": successor_contract["source_digest"],
            "source_files": dict(new_files),
            "response_schema": RESPONSE_SCHEMA,
            "projection_schema": successor_contract.get("projection_schema"),
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(set(old_files) | set(new_files))
            if old_files.get(name) != new_files.get(name)
        },
        "retained_stage_manifest": {
            "path": retained_path.relative_to(temporary).as_posix(),
            "sha256": file_digest(retained_path),
        },
        "reused_artifact_manifest": {
            "path": reuse_path.relative_to(temporary).as_posix(),
            "sha256": file_digest(reuse_path),
            "file_count": len(copied),
            "file_set_digest": reuse_manifest["file_set_digest"],
        },
        "superseded_artifact_scopes": (
            [
                "log_overflow_closure_model", "projection", "predictions",
                "assessment_metrics", "optimization", "selected_control_replays",
                "timing", "report",
            ]
            if retain_ridge else
            [
                "models", "predictions", "assessment_metrics", "optimization",
                "selected_control_replays", "timing", "report",
            ]
        ),
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record_path = migrations_directory / f"{migration_id}.json"
    atomic_json(record_path, record)
    history_entry = {
        "migration_id": migration_id,
        "record": record_path.relative_to(temporary).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_path.relative_to(temporary).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_path),
        "predecessor_source_digest": source_contract["source_digest"],
        "successor_source_digest": successor_contract["source_digest"],
    }
    migrated = dict(successor_contract)
    migrated["contract_migrations"] = [
        *source_contract.get("contract_migrations", []), history_entry,
    ]
    atomic_json(temporary / "inputs" / "contract.json", migrated)
    _validate_migration_history(temporary, migrated)
    _validate_fork_reuse_manifest(
        temporary, record["reused_artifact_manifest"], verify_files=True,
    )
    _replace_with_retry(temporary, run)
    temporary_owner.cleanup()


def _validate_all_retained_stages(run: Path, retention: Mapping[str, Any]) -> None:
    stages = retention.get("stages")
    if not isinstance(stages, Mapping) or not stages:
        raise RuntimeError("retained-stage manifest is empty")
    for stage_name, stage in stages.items():
        if not isinstance(stage, Mapping):
            raise RuntimeError(f"retained stage is invalid: {stage_name}")
        checkpoint = (run / str(stage.get("checkpoint", ""))).resolve()
        if (
            run.resolve() not in checkpoint.parents
            or not checkpoint.is_file()
            or file_digest(checkpoint) != stage.get("checkpoint_sha256")
            or (
                bool(stage.get("artifacts", {}))
                and not _artifacts_match(run, stage.get("artifacts", {}))
            )
        ):
            raise RuntimeError(f"retained stage changed before reuse: {stage_name}")


def initialize_reused_run(
    run: Path,
    *,
    source_run_id: str,
    successor_contract: Mapping[str, Any],
) -> None:
    """Create a self-contained rerun folder from immutable prior checkpoints."""

    if successor_contract.get("run_id") != run.name:
        raise RuntimeError("successor contract run id does not match the rerun directory")
    source_run = resolve_run_directory(source_run_id)
    if source_run == run:
        raise ValueError("reuse source and target run directories must differ")
    if run.exists():
        _validate_existing_reused_run(
            run,
            source_run_id=source_run_id,
            expected_contract=successor_contract,
        )
        return
    source_contract_path = source_run / "inputs" / "contract.json"
    source_contract = _load_json_object(
        source_contract_path, description="reuse-source contract",
    )
    if source_contract.get("run_id") != source_run_id:
        raise RuntimeError("reuse-source contract has the wrong run id")
    _validate_migration_history(source_run, source_contract)
    old_files = source_contract.get("source_files")
    new_files = successor_contract.get("source_files")
    if not isinstance(old_files, Mapping) or not isinstance(new_files, Mapping):
        raise RuntimeError("run reuse requires complete source manifests")
    changed = {
        name for name in set(old_files) | set(new_files)
        if old_files.get(name) != new_files.get(name)
    }
    is_reduced_response_fork = bool(
        int(source_contract.get("runner_schema", -1)) == 9
        and int(successor_contract.get("runner_schema", -1)) == RUNNER_SCHEMA
        and successor_contract.get("response_schema", {}).get("name")
        == RESPONSE_SCHEMA
        and source_contract.get("profile") == successor_contract.get("profile")
        and source_contract.get("fixed_dataset_total")
        == successor_contract.get("fixed_dataset_total")
        and source_contract.get("development_test_split")
        == successor_contract.get("development_test_split")
        and source_contract.get("dataset_protocol")
        == successor_contract.get("dataset_protocol")
    )
    is_log_overflow_closure_fork = bool(
        int(source_contract.get("runner_schema", -1)) == 10
        and int(successor_contract.get("runner_schema", -1)) == RUNNER_SCHEMA
        and source_contract.get("response_schema")
        == successor_contract.get("response_schema")
        and source_contract.get("profile") == successor_contract.get("profile")
        and source_contract.get("fixed_dataset_total")
        == successor_contract.get("fixed_dataset_total")
        and source_contract.get("development_test_split")
        == successor_contract.get("development_test_split")
        and source_contract.get("dataset_protocol")
        == successor_contract.get("dataset_protocol")
        and successor_contract.get("projection_schema") == PROJECTION_SCHEMA
    )
    if is_reduced_response_fork or is_log_overflow_closure_fork:
        _initialize_reduced_response_fork(
            run,
            source_run=source_run,
            source_contract=source_contract,
            successor_contract=successor_contract,
            retain_ridge=is_log_overflow_closure_fork,
        )
        return
    is_pinned_v2_to_v3 = bool(
        source_run_id == POLL_LINESEARCH_FORK_MIGRATION.run_id
        and run.name == DEFAULT_RUN_ID
        and int(source_contract.get("runner_schema", -1))
        == POLL_LINESEARCH_FORK_MIGRATION.predecessor_runner_schema
        and source_contract.get("source_digest")
        == POLL_LINESEARCH_FORK_MIGRATION.predecessor_source_digest
        and file_digest(source_contract_path)
        == POLL_LINESEARCH_FORK_MIGRATION.predecessor_contract_file_digest
    )
    is_same_source_rerun = bool(
        int(source_contract.get("runner_schema", -1)) == RUNNER_SCHEMA
        and source_contract.get("source_digest") == successor_contract.get("source_digest")
        and source_contract.get("validation_protocol") == COMPARISON_PROTOCOL
        and not changed
    )
    if not (is_pinned_v2_to_v3 or is_same_source_rerun):
        raise RuntimeError(
            "reuse source is neither the pinned v2 predecessor nor a current-contract run"
        )
    if is_pinned_v2_to_v3:
        unauthorized = changed - POLL_LINESEARCH_FORK_MIGRATION.allowed_changed_source_files
        missing = POLL_LINESEARCH_FORK_MIGRATION.required_changed_source_files - changed
        if unauthorized or missing:
            raise RuntimeError(
                "run-fork migration refused source drift; "
                f"unauthorized={sorted(unauthorized)}, missing_required={sorted(missing)}"
            )
        if tuple(
            str(entry.get("migration_id", ""))
            for entry in source_contract.get("contract_migrations", [])
        ) != POLL_LINESEARCH_FORK_MIGRATION.required_prior_migration_ids:
            raise RuntimeError("pinned run-fork predecessor has unexpected history")
        if source_contract.get("validation_protocol") != "casewise_exact_common_reference_v2":
            raise RuntimeError("pinned run-fork predecessor has the wrong protocol")
        migration_id = POLL_LINESEARCH_FORK_MIGRATION.migration_id
        reason = POLL_LINESEARCH_FORK_MIGRATION.reason
        source_snapshot_relative = POLL_LINESEARCH_FORK_MIGRATION.predecessor_source_snapshot
        retired_snapshot_relative = POLL_LINESEARCH_FORK_MIGRATION.retired_casewise_snapshot
        source_snapshot = _source_snapshot_manifest(
            source_run, source_snapshot_relative, old_files,
        )
        if source_snapshot["source_digest"] != source_contract["source_digest"]:
            raise RuntimeError("pinned rerun source snapshot has the wrong digest")
        if retired_snapshot_relative is None:
            raise RuntimeError("pinned rerun omits its predecessor result archive")
        _artifact_archive_manifest(
            source_run, retired_snapshot_relative, require_live_match=True,
        )
        trigger = _load_json_object(
            source_run / "optimization" / "robustness_05"
            / "surrogate_local_convergence.json",
            description="v2 poll-budget trigger",
        )
        trigger_certificate = trigger.get("certificate")
        if (
            trigger.get("status") != "poll_budget_limited"
            or not isinstance(trigger_certificate, Mapping)
            or trigger_certificate.get("protocol")
            != "exact_qp_two_scale_feasible_poll_v2"
            or int(trigger_certificate.get("evaluations", -1)) != 2_500
            or int(trigger_certificate.get("accepted_improvements", -1)) != 36
            or float(trigger_certificate.get("initial_objective", float("nan")))
            != 1.0993998220835606
            or float(trigger_certificate.get("final_objective", float("nan")))
            != 1.0992203629252697
        ):
            raise RuntimeError("pinned v2 poll-budget trigger evidence changed")
    else:
        migration_id = f"article-v3-run-reuse-{run.name}"
        if migration_id in {
            str(entry.get("migration_id", ""))
            for entry in source_contract.get("contract_migrations", [])
        }:
            raise RuntimeError("run-reuse migration identifier already exists")
        reason = (
            "User-authorized rerun in a new self-contained directory using "
            "byte-identical upstream and primary-search checkpoints."
        )
        source_snapshot_relative = (
            f"inputs/contract_migrations/{migration_id}-predecessor-source"
        )
        retired_snapshot_relative = None

    invariant_keys = (
        "profile", "fixed_dataset_total", "development_test_split", "python",
        "platform", "runtime_versions", "assessment_gate_execution_policy",
        "optimization_protocol", "preflight_artifacts_permitted",
        "full_run_admission_gate_bypass_permitted",
    )
    if any(source_contract.get(key) != successor_contract.get(key) for key in invariant_keys):
        raise RuntimeError("run reuse cannot change the scientific profile or runtime")

    retention_authorization = replace(
        POLL_LINESEARCH_FORK_MIGRATION,
        run_id=source_run_id,
        predecessor_runner_schema=int(source_contract["runner_schema"]),
        predecessor_source_digest=str(source_contract["source_digest"]),
        predecessor_contract_file_digest=file_digest(source_contract_path),
        required_artifact_digests=(
            POLL_LINESEARCH_FORK_MIGRATION.required_artifact_digests
            if is_pinned_v2_to_v3
            else {
                path: digest
                for path, digest in (
                    POLL_LINESEARCH_FORK_MIGRATION.required_artifact_digests.items()
                )
                if not path.startswith(
                    "optimization/robustness_05/surrogate_local_convergence"
                )
                and path
                != "optimization/robustness_05/surrogate_certified.npz"
            }
        ),
    )
    retention = _validate_retained_casewise_comparison_checkpoints(
        source_run, source_contract, retention_authorization,
    )
    _validate_all_retained_stages(source_run, retention)

    run.parent.mkdir(parents=True, exist_ok=True)
    temporary_owner = tempfile.TemporaryDirectory(
        # Keep the sibling staging name shorter than the final run id.  Some
        # retained attempt paths are already close to legacy Windows MAX_PATH;
        # a descriptive temporary prefix would make otherwise valid copies
        # fail before the final atomic rename.
        prefix=".r-", dir=run.parent,
    )
    temporary = Path(temporary_owner.name).resolve()
    if temporary.parent != run.parent.resolve():
        raise RuntimeError("rerun staging directory escaped the results directory")
    copied = _copy_reusable_files(source_run, temporary, retention)
    if retired_snapshot_relative is not None:
        copied.update(_copy_artifact_archive_marker_closure(
            source_run, temporary, retired_snapshot_relative,
        ))
    migrations_directory = temporary / "inputs" / "contract_migrations"
    migrations_directory.mkdir(parents=True, exist_ok=True)

    if is_same_source_rerun:
        snapshot_root = temporary / source_snapshot_relative
        for relative, expected_digest in old_files.items():
            source = (ROOT / str(relative)).resolve()
            target = (snapshot_root / str(relative)).resolve()
            if ROOT.resolve() not in source.parents or not source.is_file():
                raise RuntimeError(f"cannot snapshot current source file: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if file_digest(target) != expected_digest:
                raise RuntimeError(f"current source differs from source-run contract: {relative}")
    source_snapshot = _source_snapshot_manifest(
        temporary, source_snapshot_relative, old_files,
    )
    retired_snapshot = (
        None
        if retired_snapshot_relative is None
        else _artifact_archive_manifest(
            temporary,
            retired_snapshot_relative,
            require_live_match=False,
            require_marker_closure=True,
        )
    )

    retention_path = migrations_directory / f"{migration_id}-retained.json"
    atomic_json(retention_path, retention)
    reuse_path = migrations_directory / f"{migration_id}-reused-files.json"
    reuse_manifest = {
        "schema": 1,
        "copy_mode": "independent_byte_copy",
        "source_run_id": source_run_id,
        "target_run_id": run.name,
        "source_contract_sha256": file_digest(source_contract_path),
        "file_count": len(copied),
        "file_set_digest": _canonical_json_digest(copied),
        "files": copied,
    }
    atomic_json(reuse_path, reuse_manifest)
    predecessor_archive = migrations_directory / f"{migration_id}-predecessor-contract.json"
    atomic_bytes(predecessor_archive, source_contract_path.read_bytes())

    record_path = migrations_directory / f"{migration_id}.json"
    record = {
        "schema": 2,
        "migration_id": migration_id,
        "authorized_date": "2026-08-24",
        "reason": reason,
        "run_fork": {
            "source_run_id": source_run_id,
            "target_run_id": run.name,
            "self_contained": True,
            "recomputed_scope": "casewise_certification_reference_timing_reporting",
        },
        "predecessor": {
            "run_id": source_run_id,
            "runner_schema": source_contract["runner_schema"],
            "source_digest": source_contract["source_digest"],
            "source_files": dict(old_files),
            "contract_file_digest": file_digest(source_contract_path),
            "archived_contract": predecessor_archive.relative_to(temporary).as_posix(),
            "archived_contract_digest": file_digest(predecessor_archive),
        },
        "predecessor_source_snapshot": source_snapshot,
        "retired_casewise_snapshot": retired_snapshot,
        "successor": {
            "run_id": run.name,
            "runner_schema": successor_contract["runner_schema"],
            "source_digest": successor_contract["source_digest"],
            "source_files": dict(new_files),
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "validation_protocol": COMPARISON_PROTOCOL,
        },
        "changed_source_files": {
            name: {"old": old_files.get(name), "new": new_files.get(name)}
            for name in sorted(changed)
        },
        "retained_stage_manifest": {
            "path": retention_path.relative_to(temporary).as_posix(),
            "sha256": file_digest(retention_path),
            "predecessor_source_digest": retention["predecessor_source_digest"],
            "assessment_input_digest": retention["assessment_input_digest"],
            "case_marker_set_digest": retention["case_marker_set_digest"],
        },
        "reused_artifact_manifest": {
            "path": reuse_path.relative_to(temporary).as_posix(),
            "sha256": file_digest(reuse_path),
            "file_count": len(copied),
            "file_set_digest": reuse_manifest["file_set_digest"],
        },
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(record_path, record)
    history_entry = {
        "migration_id": migration_id,
        "record": record_path.relative_to(temporary).as_posix(),
        "record_digest": file_digest(record_path),
        "predecessor_contract": predecessor_archive.relative_to(temporary).as_posix(),
        "predecessor_contract_digest": file_digest(predecessor_archive),
        "predecessor_source_digest": source_contract["source_digest"],
        "successor_source_digest": successor_contract["source_digest"],
    }
    migrated = dict(successor_contract)
    migrated["contract_migrations"] = [
        *source_contract["contract_migrations"], history_entry,
    ]
    atomic_json(temporary / "inputs" / "contract.json", migrated)
    _validate_migration_history(temporary, migrated)
    _validate_fork_reuse_manifest(
        temporary, record["reused_artifact_manifest"], verify_files=True,
    )
    _replace_with_retry(temporary, run)
    temporary_owner.cleanup()


def _checkpoint_source_is_authorized(
    run: Path,
    *,
    stage: str,
    checkpoint: Path,
    observed_source_id: str,
    current_source_id: str,
) -> bool:
    """Allow a carried-forward checkpoint only through its exact migration pin."""

    if observed_source_id == current_source_id:
        return True
    try:
        contract = _load_json_object(
            run / "inputs" / "contract.json", description="run contract",
        )
        if contract.get("source_digest") != current_source_id:
            return False
        _validate_migration_history(run, contract)
        history = contract["contract_migrations"]
        latest = history[-1]
        record = _load_json_object(
            run / str(latest["record"]), description="latest migration record",
        )
        if record.get("successor", {}).get("source_digest") != current_source_id:
            return False
        retained_reference = record.get("retained_stage_manifest")
        if not isinstance(retained_reference, Mapping):
            return False
        retained_path = (run / str(retained_reference.get("path", ""))).resolve()
        migrations_root = (run / "inputs" / "contract_migrations").resolve()
        if (
            migrations_root not in retained_path.parents
            or not retained_path.is_file()
            or file_digest(retained_path) != retained_reference.get("sha256")
        ):
            return False
        retained = _load_json_object(
            retained_path, description="retained-stage manifest",
        )
        stage_record = retained.get("stages", {}).get(stage)
        if not isinstance(stage_record, Mapping):
            return False
        relative = checkpoint.resolve().relative_to(run.resolve()).as_posix()
        return bool(
            stage_record.get("artifact_source_digest") == observed_source_id
            and stage_record.get("checkpoint") == relative
            and stage_record.get("checkpoint_sha256") == file_digest(checkpoint)
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


def _casewise_artifact_source_id(run: Path, current_source_id: str) -> str:
    """Return the retained casewise binding after the timing-only migration."""

    if not (run / "inputs" / "contract.json").is_file():
        return current_source_id
    contract = _load_json_object(
        run / "inputs" / "contract.json", description="run contract",
    )
    history = contract.get("contract_migrations")
    if not isinstance(history, list) or not history:
        return current_source_id
    latest = history[-1]
    if latest.get("migration_id") != CASEWISE_TIMING_AGGREGATION_MIGRATION.migration_id:
        return current_source_id
    _validate_migration_history(run, contract)
    record = _load_json_object(
        run / str(latest["record"]), description="casewise timing migration",
    )
    retained_reference = record.get("retained_stage_manifest")
    if not isinstance(retained_reference, Mapping):
        raise RuntimeError("casewise timing migration omits retained artifacts")
    retained_path = (run / str(retained_reference.get("path", ""))).resolve()
    if (
        run.resolve() not in retained_path.parents
        or not retained_path.is_file()
        or file_digest(retained_path) != retained_reference.get("sha256")
    ):
        raise RuntimeError("casewise timing retention manifest changed")
    retained = _load_json_object(
        retained_path, description="casewise timing retention manifest",
    )
    predecessor_source_id = str(
        retained.get("retained_casewise_source_digest", "")
    )
    if (
        predecessor_source_id
        != record.get("predecessor", {}).get("source_digest")
        or record.get("successor", {}).get("source_digest") != current_source_id
    ):
        raise RuntimeError("casewise timing source lineage is inconsistent")
    pinned = retained.get("pinned_artifacts")
    if not isinstance(pinned, Mapping):
        raise RuntimeError("casewise timing migration omits its pinned ledger")
    _validate_pinned_artifacts(run, pinned)
    return predecessor_source_id


def assert_source_unchanged(expected: Mapping[str, str]) -> None:
    current = source_file_digests()
    if current != dict(expected):
        changed = sorted(set(current) | set(expected))
        changed = [name for name in changed if current.get(name) != expected.get(name)]
        raise RuntimeError(
            "article source changed during the run; do not mix artifacts. "
            f"Changed files: {', '.join(changed)}"
        )


def _validate_generation_block(
    targets: np.ndarray,
    diagnostics: pd.DataFrame,
    *,
    block: str,
    count: int,
    profile: StudyProfile,
) -> None:
    if targets.shape != (count, profile.mechanistic_response_count) or not np.all(np.isfinite(targets)):
        raise RuntimeError(f"{block} target block is incomplete or non-finite")
    required = {
        "row", "accepted", "root_difference_inf", "branch_agreement",
        "mass_residual_start_1", "mass_residual_start_2",
        "state_negativity_start_1", "state_negativity_start_2",
        "rate_negativity_start_1", "rate_negativity_start_2",
        "largest_real_eigenvalue_start_1", "largest_real_eigenvalue_start_2",
        "stability_agreement_start_1", "stability_agreement_start_2",
        "feed_tss_start_1", "feed_tss_start_2",
        "external_solids_loss_start_1", "external_solids_loss_start_2",
    }
    missing = required - set(diagnostics.columns)
    if missing:
        raise RuntimeError(f"{block} diagnostics omit columns: {sorted(missing)}")
    rows = np.asarray(diagnostics["row"], dtype=int)
    if len(diagnostics) != count or not np.array_equal(np.sort(rows), np.arange(count)):
        raise RuntimeError(f"{block} diagnostics do not cover every fixed row exactly once")
    accepted = diagnostics["accepted"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    branches = diagnostics["branch_agreement"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if accepted.isna().any() or not bool(accepted.all()):
        raise RuntimeError(f"{block} contains an unaccepted fixed mechanistic row")
    if branches.isna().any() or not bool(branches.all()):
        raise RuntimeError(f"{block} contains a two-start branch disagreement")
    numeric_columns = list(required - {"row", "accepted", "branch_agreement"})
    numeric = diagnostics[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.all(np.isfinite(numeric.to_numpy())):
        raise RuntimeError(f"{block} mechanistic audits contain non-finite values")
    checks = (
        ("root_difference_inf", np.less_equal, 1.0e-6),
        ("mass_residual_start_1", np.less_equal, 1.0e-8),
        ("mass_residual_start_2", np.less_equal, 1.0e-8),
        ("state_negativity_start_1", np.less_equal, 1.0e-10),
        ("state_negativity_start_2", np.less_equal, 1.0e-10),
        ("rate_negativity_start_1", np.less_equal, 1.0e-12),
        ("rate_negativity_start_2", np.less_equal, 1.0e-12),
        ("largest_real_eigenvalue_start_1", np.less_equal, -1.0e-8),
        ("largest_real_eigenvalue_start_2", np.less_equal, -1.0e-8),
        ("stability_agreement_start_1", np.less_equal, 1.0e-6),
        ("stability_agreement_start_2", np.less_equal, 1.0e-6),
        ("feed_tss_start_1", np.greater_equal, 1.0),
        ("feed_tss_start_2", np.greater_equal, 1.0),
        ("external_solids_loss_start_1", np.greater_equal, 1.0),
        ("external_solids_loss_start_2", np.greater_equal, 1.0),
    )
    failures = [name for name, comparison, limit in checks if not bool(
        np.all(comparison(numeric[name].to_numpy(), limit))
    )]
    if failures:
        raise RuntimeError(f"{block} failed mechanistic generation gates: {failures}")


def _load_generation_checkpoint(
    run: Path,
    *,
    block: str,
    count: int,
    profile: StudyProfile,
    source_id: str,
    design_id: str,
) -> tuple[MechanisticBlockResult, float] | None:
    marker_path = run / "datasets" / block / "block_complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            not _checkpoint_source_is_authorized(
                run, stage=f"generation/{block}", checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("design_digest") != design_id
            or marker.get("block") != block
            or int(marker.get("row_count", -1)) != count
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return None
        output = run / "datasets" / block
        target_path = output / "mechanistic_accepted_v3.npz"
        input_path = output / "accepted_inputs.npz"
        with np.load(target_path, allow_pickle=False) as stored:
            targets = np.asarray(stored["targets"], dtype=float)
        with np.load(input_path, allow_pickle=False) as stored:
            decisions = np.asarray(stored["decisions"], dtype=float)
            influents = np.asarray(stored["influents"], dtype=float)
            source_candidate_id = np.asarray(stored["source_candidate_id"], dtype=str)
        diagnostics = pd.read_csv(output / "accepted_diagnostics.csv")
        attempts = pd.read_csv(output / "all_attempts.csv")
        provenance = pd.read_csv(output / "accepted_provenance.csv")
        _validate_generation_block(
            targets, diagnostics, block=block, count=count, profile=profile,
        )
        if (
            decisions.shape != (count, 7)
            or influents.shape != (count, 20)
            or not np.all(np.isfinite(decisions))
            or not np.all(np.isfinite(influents))
            or np.any(decisions < DECISION_LOWER)
            or np.any(decisions > DECISION_UPPER)
            or np.any(influents < INFLUENT_LOWER)
            or np.any(influents > INFLUENT_UPPER)
            or len(provenance) != count
            or len(source_candidate_id) != count
            or not np.array_equal(
                source_candidate_id,
                provenance["source_candidate_id"].to_numpy(dtype=str),
            )
            or marker.get("effective_input_digest")
            != array_digest(
                decisions=np.asarray(decisions, dtype="<f8"),
                influents=np.asarray(influents, dtype="<f8"),
            )
        ):
            return None
        _validate_attempt_checkpoint_hashes(output, attempts)
        return MechanisticBlockResult(
            decisions=decisions, influents=influents, targets=targets,
            diagnostics=diagnostics, attempts=attempts, provenance=provenance,
        ), float(marker["elapsed_seconds"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _validate_attempt_checkpoint_hashes(
    output: Path, attempts: pd.DataFrame,
) -> None:
    required = {"checkpoint_path", "checkpoint_sha256", "candidate_id", "accepted"}
    if required - set(attempts.columns) or attempts.empty:
        raise RuntimeError("generation attempt ledger is incomplete")
    if attempts["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("generation attempt ledger contains duplicate candidates")
    root = output.resolve()
    for row in attempts.itertuples(index=False):
        path = (output / str(row.checkpoint_path)).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError("generation attempt checkpoint is missing or outside its block")
        if file_digest(path) != str(row.checkpoint_sha256):
            raise RuntimeError(f"generation attempt checkpoint changed: {path.name}")


def _boolean_series(values: pd.Series, *, description: str) -> pd.Series:
    converted = values.astype(str).str.lower().map({"true": True, "false": False})
    if converted.isna().any():
        raise RuntimeError(f"{description} contains invalid Boolean values")
    return converted.astype(bool)


def _write_generation_audits(
    output: Path, result: MechanisticBlockResult,
) -> None:
    physical = np.column_stack((result.decisions, result.influents))
    lower = np.concatenate((DECISION_LOWER, INFLUENT_LOWER))
    upper = np.concatenate((DECISION_UPPER, INFLUENT_UPPER))
    names = tuple(DECISION_NAMES) + tuple(COMPONENTS)
    coverage = pd.DataFrame([
        {
            "coordinate_index": index,
            "coordinate_group": "decision" if index < 7 else "influent",
            "coordinate": names[index],
            "declared_lower": lower[index],
            "declared_upper": upper[index],
            "accepted_minimum": float(np.min(physical[:, index])),
            "accepted_maximum": float(np.max(physical[:, index])),
            "accepted_span_fraction": float(
                np.ptp(physical[:, index]) / (upper[index] - lower[index])
            ),
            "outside_declared_box_count": int(np.count_nonzero(
                (physical[:, index] < lower[index])
                | (physical[:, index] > upper[index])
            )),
        }
        for index in range(physical.shape[1])
    ])
    attempts = result.attempts.copy()
    if "rejection_reason" not in attempts.columns:
        raise RuntimeError("generation attempt ledger omits rejection_reason")
    accepted = _boolean_series(attempts["accepted"], description="attempt ledger")
    reasons = attempts["rejection_reason"].astype(str)
    if bool(((accepted & reasons.ne("accepted")) | (~accepted & reasons.eq("accepted"))).any()):
        raise RuntimeError("generation rejection reasons disagree with acceptance flags")
    rejected = attempts.loc[attempts["rejection_reason"] != "accepted"]
    if rejected.empty:
        rejection_summary = pd.DataFrame(columns=(
            "rejection_reason", "attempt_count", "fraction_of_all_attempts",
        ))
    else:
        rejection_summary = (
            rejected.groupby("rejection_reason", dropna=False).size()
            .rename("attempt_count").reset_index()
            .sort_values("rejection_reason", kind="stable").reset_index(drop=True)
        )
        rejection_summary["fraction_of_all_attempts"] = (
            rejection_summary["attempt_count"] / len(attempts)
        )
    atomic_dataframe(output / "accepted_coordinate_coverage.csv", coverage)
    atomic_dataframe(output / "rejection_reason_summary.csv", rejection_summary)


def _generation_publication_paths(output: Path) -> tuple[Path, ...]:
    names = (
        "mechanistic_accepted_v3.npz", "accepted_inputs.npz",
        "accepted_diagnostics.csv", "all_attempts.csv",
        "accepted_provenance.csv", "base_checkpoint_migration.csv",
        "replacement_summary.json", "accepted_coordinate_coverage.csv",
        "rejection_reason_summary.csv",
    )
    fixed = tuple(output / name for name in names)
    summary_path = output / "replacement_summary.json"
    summary = _load_json_object(summary_path, description="replacement summary")
    round_count = int(summary.get("supplemental_round_count", -1))
    if round_count < 0:
        raise RuntimeError("replacement summary has an invalid round count")
    expected_manifests = tuple(
        output / "attempts" / "replacement" / f"round_{index:06d}" / "manifest.json"
        for index in range(1, round_count + 1)
    )
    actual_manifests = tuple(sorted(
        (output / "attempts" / "replacement").glob("round_*/manifest.json")
    ))
    if actual_manifests != expected_manifests:
        raise RuntimeError("supplemental round-manifest sequence is incomplete or unexpected")
    return fixed + expected_manifests


def _run_generation_block(
    run: Path,
    design: Mapping[str, object],
    *,
    block: str,
    profile: StudyProfile,
    source_files: Mapping[str, str],
    design_id: str,
) -> tuple[MechanisticBlockResult, float, bool]:
    count = profile.development_count if block == "development" else profile.test_count
    source_id = source_digest(source_files)
    checkpoint = _load_generation_checkpoint(
        run, block=block, count=count, profile=profile,
        source_id=source_id, design_id=design_id,
    )
    if checkpoint is not None:
        return (*checkpoint, True)
    started = perf_counter()
    result = generate_mechanistic_block_with_replacements(
        np.asarray(design[f"{block}_decisions"]),
        np.asarray(design[f"{block}_influents"]),
        profile,
        run / "datasets" / block,
        block=block,
    )
    elapsed = perf_counter() - started
    _validate_generation_block(
        result.targets, result.diagnostics,
        block=block, count=count, profile=profile,
    )
    _validate_attempt_checkpoint_hashes(run / "datasets" / block, result.attempts)
    _write_generation_audits(run / "datasets" / block, result)
    assert_source_unchanged(source_files)
    paths = _generation_publication_paths(run / "datasets" / block)
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"{block} generator did not publish its required artifacts")
    atomic_json(run / "datasets" / block / "block_complete.json", {
        "stage": "mechanistic_generation",
        "block": block,
        "source_digest": source_id,
        "design_digest": design_id,
        "row_count": count,
        "accepted_count": count,
        "target_shape": list(result.targets.shape),
        "attempt_count": len(result.attempts),
        "rejected_attempt_count": int(
            (~_boolean_series(
                result.attempts["accepted"], description="attempt ledger",
            )).sum()
        ),
        "replacement_slot_count": int(
            _boolean_series(
                result.provenance["replaced_base_candidate"],
                description="provenance ledger",
            ).sum()
        ),
        "effective_input_digest": array_digest(
            decisions=np.asarray(result.decisions, dtype="<f8"),
            influents=np.asarray(result.influents, dtype="<f8"),
        ),
        "elapsed_seconds": elapsed,
        "artifacts": _artifact_hashes(run, paths),
    })
    return result, elapsed, False


def run_generation(
    run: Path,
    design: Mapping[str, object],
    *,
    profile: StudyProfile,
    source_files: Mapping[str, str],
) -> GenerationResult:
    design_id = _design_digest(design)
    blocks: dict[str, tuple[MechanisticBlockResult, float, bool]] = {}
    for block in ("development", "test"):
        blocks[block] = _run_generation_block(
            run, design, block=block, profile=profile,
            source_files=source_files, design_id=design_id,
        )
    summary = pd.DataFrame([
        {
            "block": block,
            "fixed_candidate_rows": len(result[0].diagnostics),
            "required_accepted_rows": len(result[0].diagnostics),
            "accepted_rows": int(result[0].diagnostics["accepted"].astype(
                str
            ).str.lower().eq("true").sum()),
            "total_attempts": len(result[0].attempts),
            "excluded_rejected_attempts": int(
                (~_boolean_series(
                    result[0].attempts["accepted"], description="attempt ledger",
                )).sum()
            ),
            "replacement_slots": int(
                _boolean_series(
                    result[0].provenance["replaced_base_candidate"],
                    description="provenance ledger",
                ).sum()
            ),
            "elapsed_seconds": result[1],
            "reused_complete_checkpoint": result[2],
        }
        for block, result in blocks.items()
    ])
    summary_path = run / "metrics" / "mechanistic_generation_summary.csv"
    source_id = source_digest(source_files)
    marker_source_ids = {
        str(_load_json_object(
            run / "datasets" / block / "block_complete.json",
            description=f"{block} generation marker",
        ).get("source_digest", ""))
        for block in blocks
    }
    carried_forward = bool(
        all(result[2] for result in blocks.values())
        and marker_source_ids != {source_id}
    )
    if carried_forward:
        if len(marker_source_ids) != 1 or not _checkpoint_source_is_authorized(
            run, stage="generation/summary", checkpoint=summary_path,
            observed_source_id=next(iter(marker_source_ids)),
            current_source_id=source_id,
        ):
            raise RuntimeError("generation summary is not authorized for carry-forward")
    else:
        atomic_dataframe(summary_path, summary)
    effective_design = dict(design)
    for block, result in blocks.items():
        effective_design[f"{block}_decisions"] = result[0].decisions
        effective_design[f"{block}_influents"] = result[0].influents
    effective_path = run / "datasets" / "effective_design.npz"
    effective_arrays = {
        name: np.asarray(effective_design[name]) for name in DESIGN_ARRAYS
    }
    if effective_path.is_file():
        with np.load(effective_path, allow_pickle=False) as stored:
            if set(stored.files) != set(DESIGN_ARRAYS) or any(
                not np.array_equal(stored[name], effective_arrays[name])
                for name in DESIGN_ARRAYS
            ):
                raise RuntimeError("existing effective design differs from accepted inputs")
    else:
        atomic_npz(effective_path, **effective_arrays)
    effective_id = _design_digest(effective_design)
    effective_manifest = {
        "schema": 1,
        "base_design_digest": design_id,
        "effective_design_digest": effective_id,
        "source_digest": source_id,
        "artifact": {
            effective_path.relative_to(run).as_posix(): file_digest(effective_path),
        },
        "blocks": {
            block: {
                "accepted_input_artifact": (
                    run / "datasets" / block / "accepted_inputs.npz"
                ).relative_to(run).as_posix(),
                "accepted_input_artifact_sha256": file_digest(
                    run / "datasets" / block / "accepted_inputs.npz"
                ),
                "replacement_slots": int(
                    _boolean_series(
                        result[0].provenance["replaced_base_candidate"],
                        description="provenance ledger",
                    ).sum()
                ),
            }
            for block, result in blocks.items()
        },
    }
    effective_manifest_path = run / "datasets" / "effective_design_manifest.json"
    if effective_manifest_path.is_file():
        existing_manifest = _load_json_object(
            effective_manifest_path, description="effective design manifest",
        )
        expected_manifest = _json_ready(effective_manifest)
        existing_source_id = str(existing_manifest.get("source_digest", ""))
        existing_without_source = dict(existing_manifest)
        expected_without_source = dict(expected_manifest)
        existing_without_source.pop("source_digest", None)
        expected_without_source.pop("source_digest", None)
        if (
            existing_without_source != expected_without_source
            or not _checkpoint_source_is_authorized(
                run, stage="generation/effective_design",
                checkpoint=effective_manifest_path,
                observed_source_id=existing_source_id,
                current_source_id=source_id,
            )
        ):
            raise RuntimeError("existing effective design manifest differs")
    else:
        atomic_json(effective_manifest_path, effective_manifest)
    assert_source_unchanged(source_files)
    return GenerationResult(
        design=effective_design,
        development_targets=blocks["development"][0].targets,
        test_targets=blocks["test"][0].targets,
    )


def sample_accepted_generation(
    run: Path,
    *,
    profile: StudyProfile,
    source_files: Mapping[str, str],
) -> GenerationResult:
    """Materialize a self-contained random 10k subset of accepted source rows.

    The population is the 16,714 accepted rows frozen from the interrupted 50k
    generation.  Sampling is without replacement and happens before the fixed
    80/20 development/holdout partition.  No new mechanistic solve is performed.
    """

    validate_authorized_profile(profile)
    if profile.name != SAMPLED_PROFILE_NAME:
        raise RuntimeError("accepted-row sampling requires the sampled profile")
    marker_path = run / "datasets" / "sampled_accepted_complete.json"
    records_path = run / "inputs" / "generator_records.json"
    design_path = run / "datasets" / "design.npz"
    if marker_path.is_file():
        with np.load(design_path, allow_pickle=False) as stored:
            design = {name: np.asarray(stored[name]) for name in DESIGN_ARRAYS}
        design["generators"] = _load_json_object(
            records_path, description="sampled generator records",
        )
        validate_design(design, profile)
        return run_generation(run, design, profile=profile, source_files=source_files)

    source_run = resolve_run_directory(SAMPLED_ACCEPTED_SOURCE_RUN_ID)
    source_marker = source_run / "datasets" / "frozen_accepted_complete.json"
    source_partition = source_run / "inputs" / "frozen_accepted_partition.json"
    if not source_marker.is_file() or not source_partition.is_file():
        raise RuntimeError("sample source does not contain the frozen accepted dataset")
    source_generators = _load_json_object(
        source_run / "inputs" / "generator_records.json",
        description="sample source generator records",
    )
    source_marker_data = _load_json_object(
        source_marker, description="sample source frozen-dataset marker",
    )
    if int(source_marker_data.get("accepted_count", -1)) != FROZEN_ACCEPTED_TOTAL:
        raise RuntimeError("sample source has an unexpected accepted-row population")

    population: list[dict[str, Any]] = []
    source_design_path = source_run / "datasets" / "effective_design.npz"
    with np.load(source_design_path, allow_pickle=False) as stored:
        if set(stored.files) != set(DESIGN_ARRAYS):
            raise RuntimeError("sample source effective design has unexpected arrays")
        robustness_influents = np.asarray(stored["robustness_influents"], dtype=float)
    for block in ("development", "test"):
        output = source_run / "datasets" / block
        with np.load(output / "accepted_inputs.npz", allow_pickle=False) as stored:
            decisions = np.asarray(stored["decisions"], dtype=float)
            influents = np.asarray(stored["influents"], dtype=float)
            source_candidate_id = np.asarray(stored["source_candidate_id"], dtype=str)
            source_candidate_round = np.asarray(stored["source_candidate_round"], dtype=int)
            source_candidate_index = np.asarray(stored["source_candidate_index"], dtype=int)
            source_candidate_ordinal = np.asarray(stored["source_candidate_ordinal"], dtype=int)
        with np.load(output / "mechanistic_accepted_v3.npz", allow_pickle=False) as stored:
            targets = np.asarray(stored["targets"], dtype=float)
            states_start_1 = np.asarray(stored["states_start_1"], dtype=float)
            states_start_2 = np.asarray(stored["states_start_2"], dtype=float)
        diagnostics = pd.read_csv(output / "accepted_diagnostics.csv")
        attempts = pd.read_csv(output / "all_attempts.csv")
        if not (
            len(decisions) == len(influents) == len(targets) == len(diagnostics)
            == len(source_candidate_id)
        ):
            raise RuntimeError(f"sample source {block} accepted artifacts disagree")
        attempts_by_candidate = attempts.set_index("candidate_id", drop=False)
        if attempts_by_candidate.index.duplicated().any():
            raise RuntimeError(f"sample source {block} attempt ledger has duplicate IDs")
        for row in range(len(decisions)):
            candidate_id = str(source_candidate_id[row])
            if candidate_id not in attempts_by_candidate.index:
                raise RuntimeError(f"sample source {block} omits accepted checkpoint")
            attempt = attempts_by_candidate.loc[candidate_id].to_dict()
            checkpoint = (output / str(attempt["checkpoint_path"])).resolve()
            if not checkpoint.is_file() or file_digest(checkpoint) != str(
                attempt["checkpoint_sha256"]
            ):
                raise RuntimeError(f"sample source checkpoint changed: {candidate_id}")
            population.append({
                "decision": decisions[row],
                "influent": influents[row],
                "target": targets[row],
                "state_start_1": states_start_1[row],
                "state_start_2": states_start_2[row],
                "diagnostic": diagnostics.iloc[row].to_dict(),
                "attempt": attempt,
                "checkpoint": checkpoint,
                "source_candidate_id": candidate_id,
                "source_candidate_round": int(source_candidate_round[row]),
                "source_candidate_index": int(source_candidate_index[row]),
                "source_candidate_ordinal": int(source_candidate_ordinal[row]),
            })
    population.sort(key=lambda item: item["source_candidate_index"])
    if len(population) != FROZEN_ACCEPTED_TOTAL:
        raise RuntimeError("sample source accepted-row population is incomplete")
    source_indices = [item["source_candidate_index"] for item in population]
    if len(set(source_indices)) != len(source_indices):
        raise RuntimeError("sample source accepted-row population has duplicate indices")

    rng = np.random.default_rng(SAMPLED_ACCEPTED_SEED)
    population_order = np.asarray(rng.permutation(len(population)), dtype=int)
    selected_population_slots = population_order[:SAMPLED_ACCEPTED_TOTAL]
    development_items = [population[index] for index in selected_population_slots[:SAMPLED_DEVELOPMENT_COUNT]]
    test_items = [population[index] for index in selected_population_slots[SAMPLED_DEVELOPMENT_COUNT:]]

    sampled_design: dict[str, object] = {
        "development_decisions": np.vstack([item["decision"] for item in development_items]),
        "development_influents": np.vstack([item["influent"] for item in development_items]),
        "test_decisions": np.vstack([item["decision"] for item in test_items]),
        "test_influents": np.vstack([item["influent"] for item in test_items]),
        "robustness_influents": robustness_influents,
        "generators": {
            **source_generators,
            "sampling": {
                "algorithm": "numpy.default_rng.permutation",
                "seed": SAMPLED_ACCEPTED_SEED,
                "population_size": len(population),
                "selected_size": SAMPLED_ACCEPTED_TOTAL,
                "replacement": False,
            },
        },
    }
    validate_design(sampled_design, profile)
    atomic_npz(design_path, **{
        name: np.asarray(sampled_design[name]) for name in DESIGN_ARRAYS
    })
    atomic_json(records_path, _json_ready(sampled_design["generators"]))
    source_id = source_digest(source_files)
    design_id = _design_digest(sampled_design)

    def publish_block(block: str, selected: list[dict[str, Any]]) -> None:
        output = run / "datasets" / block
        checkpoint_directory = output / "source_rows"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        decisions = np.vstack([item["decision"] for item in selected])
        influents = np.vstack([item["influent"] for item in selected])
        targets = np.vstack([item["target"] for item in selected])
        states_start_1 = np.vstack([item["state_start_1"] for item in selected])
        states_start_2 = np.vstack([item["state_start_2"] for item in selected])
        diagnostics_records: list[dict[str, Any]] = []
        attempts_records: list[dict[str, Any]] = []
        provenance_records: list[dict[str, Any]] = []
        migration_records: list[dict[str, Any]] = []
        for slot, item in enumerate(selected):
            source_index = int(item["source_candidate_index"])
            checkpoint = item["checkpoint"]
            destination = checkpoint_directory / f"row_{source_index:06d}.npz"
            if destination.is_file():
                if file_digest(destination) != file_digest(checkpoint):
                    raise RuntimeError("sampled checkpoint destination changed")
            else:
                shutil.copy2(checkpoint, destination)
            diagnostic = dict(item["diagnostic"])
            diagnostic["row"] = slot
            diagnostics_records.append(diagnostic)
            attempt = dict(item["attempt"])
            attempt["checkpoint_path"] = destination.relative_to(output).as_posix()
            attempt["checkpoint_sha256"] = file_digest(destination)
            attempts_records.append(attempt)
            provenance = {
                "accepted_slot": slot,
                "base_candidate_id": f"{block}:sampled:c{slot:06d}",
                "source_candidate_id": item["source_candidate_id"],
                "source_candidate_round": item["source_candidate_round"],
                "source_candidate_index": source_index,
                "source_candidate_ordinal": item["source_candidate_ordinal"],
                "replaced_base_candidate": False,
            }
            provenance_records.append(provenance)
            migration_records.append({
                "candidate_id": item["source_candidate_id"],
                "candidate_index": source_index,
                "checkpoint_path": destination.relative_to(output).as_posix(),
                "checkpoint_sha256": attempt["checkpoint_sha256"],
                "original_contract_hash": "frozen_accepted_checkpoint_split_80_20_v1",
                "preexisting_checkpoint": True,
                "accepted": True,
                "preserved_without_rewrite": True,
            })
        diagnostics = pd.DataFrame(diagnostics_records)
        attempts = pd.DataFrame(attempts_records)
        provenance = pd.DataFrame(provenance_records)
        if attempts["candidate_id"].astype(str).duplicated().any():
            raise RuntimeError("sampled accepted rows have duplicate candidate IDs")
        atomic_npz(
            output / "accepted_inputs.npz",
            decisions=decisions,
            influents=influents,
            source_candidate_id=provenance["source_candidate_id"].to_numpy(str),
            source_candidate_round=provenance["source_candidate_round"].to_numpy(int),
            source_candidate_index=provenance["source_candidate_index"].to_numpy(int),
            source_candidate_ordinal=provenance["source_candidate_ordinal"].to_numpy(int),
        )
        atomic_npz(
            output / "mechanistic_accepted_v3.npz",
            contract_hash=np.asarray("random_sampled_accepted_checkpoint_split_80_20_v1"),
            targets=targets,
            states_start_1=states_start_1,
            states_start_2=states_start_2,
        )
        atomic_dataframe(output / "accepted_diagnostics.csv", diagnostics)
        atomic_dataframe(output / "all_attempts.csv", attempts)
        atomic_dataframe(output / "accepted_provenance.csv", provenance)
        atomic_dataframe(output / "base_checkpoint_migration.csv", pd.DataFrame(migration_records))
        atomic_json(output / "replacement_summary.json", {
            "schema": replacement_generation.REPLACEMENT_SCHEMA,
            "block": block,
            "requested_accepted_count": len(selected),
            "accepted_count": len(selected),
            "base_attempt_count": len(selected),
            "base_accepted_count": len(selected),
            "supplemental_attempt_count": 0,
            "supplemental_accepted_count": 0,
            "supplemental_round_count": 0,
            "random_sampled_from_frozen_accepted_run": SAMPLED_ACCEPTED_SOURCE_RUN_ID,
            "sample_seed": SAMPLED_ACCEPTED_SEED,
        })
        result = MechanisticBlockResult(
            decisions=decisions, influents=influents, targets=targets,
            diagnostics=diagnostics, attempts=attempts, provenance=provenance,
        )
        _validate_generation_block(
            targets, diagnostics, block=block, count=len(selected), profile=profile,
        )
        _validate_attempt_checkpoint_hashes(output, attempts)
        _write_generation_audits(output, result)
        publication_paths = _generation_publication_paths(output)
        atomic_json(output / "block_complete.json", {
            "stage": "random_sampled_accepted_mechanistic_dataset",
            "block": block,
            "source_digest": source_id,
            "design_digest": design_id,
            "row_count": len(selected),
            "accepted_count": len(selected),
            "target_shape": list(targets.shape),
            "attempt_count": len(attempts),
            "rejected_attempt_count": 0,
            "replacement_slot_count": 0,
            "effective_input_digest": array_digest(
                decisions=np.asarray(decisions, dtype="<f8"),
                influents=np.asarray(influents, dtype="<f8"),
            ),
            "elapsed_seconds": float(np.nansum(pd.to_numeric(
                attempts.get("elapsed_seconds", pd.Series(dtype=float)), errors="coerce",
            ))),
            "artifacts": _artifact_hashes(run, publication_paths),
        })

    publish_block("development", development_items)
    publish_block("test", test_items)
    partition_path = run / "inputs" / "random_sampled_accepted_partition.json"
    selected_records = [
        {
            "sampled_slot": slot,
            "block": "development" if slot < SAMPLED_DEVELOPMENT_COUNT else "test",
            "block_slot": slot if slot < SAMPLED_DEVELOPMENT_COUNT else slot - SAMPLED_DEVELOPMENT_COUNT,
            "population_slot": int(population_slot),
            "source_candidate_id": population[int(population_slot)]["source_candidate_id"],
            "source_candidate_index": population[int(population_slot)]["source_candidate_index"],
        }
        for slot, population_slot in enumerate(selected_population_slots)
    ]
    atomic_json(partition_path, {
        "schema": 1,
        "protocol": "random_sampled_accepted_checkpoint_split_80_20_v1",
        "source_run_id": SAMPLED_ACCEPTED_SOURCE_RUN_ID,
        "target_run_id": run.name,
        "source_population_accepted_count": len(population),
        "sampled_accepted_count": SAMPLED_ACCEPTED_TOTAL,
        "development_count": SAMPLED_DEVELOPMENT_COUNT,
        "test_count": SAMPLED_TEST_COUNT,
        "sampling_algorithm": "numpy.default_rng.permutation",
        "sampling_seed": SAMPLED_ACCEPTED_SEED,
        "sampling_without_replacement": True,
        "new_mechanistic_solves": 0,
        "selected_rows": selected_records,
        "source_artifacts": {
            "frozen_accepted_complete.json": file_digest(source_marker),
            "frozen_accepted_partition.json": file_digest(source_partition),
            "effective_design.npz": file_digest(source_design_path),
        },
    })
    atomic_json(marker_path, {
        "stage": "random_sampled_accepted_dataset",
        "source_digest": source_id,
        "design_digest": design_id,
        "accepted_count": SAMPLED_ACCEPTED_TOTAL,
        "development_count": SAMPLED_DEVELOPMENT_COUNT,
        "test_count": SAMPLED_TEST_COUNT,
        "artifacts": _artifact_hashes(run, (
            design_path, partition_path,
            run / "datasets" / "development" / "block_complete.json",
            run / "datasets" / "test" / "block_complete.json",
        )),
    })
    assert_source_unchanged(source_files)
    return run_generation(run, sampled_design, profile=profile, source_files=source_files)


def freeze_accepted_generation(
    run: Path,
    *,
    profile: StudyProfile,
    source_files: Mapping[str, str],
) -> GenerationResult:
    """Freeze accepted interrupted-run checkpoints into an 80/20 dataset.

    The source checkpoints are validated against the original deterministic
    50,000-row design.  Accepted candidates are ordered by their original
    candidate index before the first 13,371 are assigned to development and
    the remaining 3,343 to the holdout block.  No new mechanistic solve is
    performed by this operation.
    """

    validate_authorized_profile(profile)
    if profile.name != FROZEN_PROFILE_NAME:
        raise RuntimeError("accepted-checkpoint freezing requires the frozen profile")
    marker_path = run / "datasets" / "frozen_accepted_complete.json"
    records_path = run / "inputs" / "generator_records.json"
    design_path = run / "datasets" / "design.npz"
    if marker_path.is_file():
        with np.load(design_path, allow_pickle=False) as stored:
            design = {name: np.asarray(stored[name]) for name in DESIGN_ARRAYS}
        design["generators"] = _load_json_object(
            records_path, description="frozen generator records",
        )
        validate_design(design, profile)
        return run_generation(
            run, design, profile=profile, source_files=source_files,
        )

    source_profile = profile_for_dataset_total(50_000)
    source_design = create_design(source_profile)
    validate_design(source_design, source_profile)
    if not design_path.is_file():
        raise RuntimeError("frozen run is missing its original 50,000-row design")
    with np.load(design_path, allow_pickle=False) as stored:
        if set(stored.files) != set(DESIGN_ARRAYS) or any(
            not np.array_equal(stored[name], np.asarray(source_design[name]))
            for name in DESIGN_ARRAYS
        ):
            raise RuntimeError("frozen source design differs from the declared 50k design")
    source_archive = (
        run / "inputs" / "frozen_accepted" / "source_design_50000.npz"
    )
    source_archive.parent.mkdir(parents=True, exist_ok=True)
    if source_archive.is_file():
        if file_digest(source_archive) != file_digest(design_path):
            raise RuntimeError("archived 50k source design changed")
    else:
        shutil.copy2(design_path, source_archive)

    rows_directory = run / "datasets" / "development" / "rows"
    paths = sorted(rows_directory.glob("row_*.npz"))
    if len(paths) != 18_211:
        raise RuntimeError(
            f"frozen checkpoint inventory must contain 18,211 rows, found {len(paths)}"
        )
    base_contract = replacement_generation._base_contract_hash(
        np.asarray(source_design["development_decisions"]),
        np.asarray(source_design["development_influents"]),
        source_profile,
    )
    accepted_items: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
    rejected_items: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
    for path in paths:
        match = re.fullmatch(r"row_(\d{6})\.npz", path.name)
        if match is None:
            raise RuntimeError(f"unexpected frozen checkpoint name: {path.name}")
        index = int(match.group(1))
        candidate = replacement_generation._Candidate(
            block="development", round_index=0, candidate_index=index,
            candidate_ordinal=index,
            decision=np.asarray(source_design["development_decisions"])[index],
            influent=np.asarray(source_design["development_influents"])[index],
            checkpoint=path,
        )
        loaded = replacement_generation._load_attempt(
            candidate,
            expected_contract_hash=base_contract,
            state_size=5 * 20 + source_profile.layer_count,
            response_count=source_profile.mechanistic_response_count,
        )
        if loaded is None:
            raise RuntimeError(f"frozen checkpoint disappeared: {path}")
        item = (candidate, *loaded)
        (accepted_items if bool(loaded[3]["accepted"]) else rejected_items).append(item)
    accepted_items.sort(key=lambda item: item[0].candidate_index)
    rejected_items.sort(key=lambda item: item[0].candidate_index)
    if len(accepted_items) != FROZEN_ACCEPTED_TOTAL or len(rejected_items) != 1_497:
        raise RuntimeError(
            "frozen acceptance inventory changed: "
            f"accepted={len(accepted_items)}, rejected={len(rejected_items)}"
        )

    development_items = accepted_items[:FROZEN_DEVELOPMENT_COUNT]
    test_items = accepted_items[FROZEN_DEVELOPMENT_COUNT:]
    if len(test_items) != FROZEN_TEST_COUNT:
        raise RuntimeError("frozen 80/20 partition has the wrong holdout size")
    frozen_design: dict[str, object] = {
        "development_decisions": np.vstack([item[0].decision for item in development_items]),
        "development_influents": np.vstack([item[0].influent for item in development_items]),
        "test_decisions": np.vstack([item[0].decision for item in test_items]),
        "test_influents": np.vstack([item[0].influent for item in test_items]),
        "robustness_influents": np.asarray(source_design["robustness_influents"]),
        "generators": source_design["generators"],
    }
    validate_design(frozen_design, profile)
    atomic_npz(
        design_path,
        **{name: np.asarray(frozen_design[name]) for name in DESIGN_ARRAYS},
    )
    if records_path.is_file():
        existing_records = _load_json_object(
            records_path, description="source generator records",
        )
        if existing_records != _json_ready(source_design["generators"]):
            raise RuntimeError("source generator records changed before freezing")
    else:
        atomic_json(records_path, _json_ready(source_design["generators"]))

    source_id = source_digest(source_files)
    design_id = _design_digest(frozen_design)

    def publish_block(
        block: str,
        selected: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]],
        attempts_source: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]],
    ) -> None:
        output = run / "datasets" / block
        output.mkdir(parents=True, exist_ok=True)
        copied_directory = output / "source_rows"
        if block == "test":
            copied_directory.mkdir(parents=True, exist_ok=True)
        selected_slots = {
            item[0].candidate_id: slot for slot, item in enumerate(selected)
        }
        decisions = np.vstack([item[0].decision for item in selected])
        influents = np.vstack([item[0].influent for item in selected])
        targets = np.vstack([item[1] for item in selected])
        states_start_1 = np.vstack([item[2] for item in selected])
        states_start_2 = np.vstack([item[3] for item in selected])
        diagnostics_records: list[dict[str, Any]] = []
        provenance_records: list[dict[str, Any]] = []
        for slot, (candidate, _target, _first, _second, record) in enumerate(selected):
            diagnostic = dict(record)
            diagnostic.update({
                "row": slot,
                "accepted_slot": slot,
                "source_candidate_id": candidate.candidate_id,
                "source_candidate_round": candidate.round_index,
                "source_candidate_index": candidate.candidate_index,
                "source_candidate_ordinal": candidate.candidate_ordinal,
            })
            diagnostics_records.append(diagnostic)
            provenance_records.append({
                "accepted_slot": slot,
                "base_candidate_id": f"{block}:frozen:c{slot:06d}",
                "source_candidate_id": candidate.candidate_id,
                "source_candidate_round": candidate.round_index,
                "source_candidate_index": candidate.candidate_index,
                "source_candidate_ordinal": candidate.candidate_ordinal,
                "replaced_base_candidate": False,
            })
        diagnostics = pd.DataFrame(diagnostics_records)
        provenance = pd.DataFrame(provenance_records)
        attempt_records: list[dict[str, Any]] = []
        migration_records: list[dict[str, Any]] = []
        for candidate, _target, _first, _second, record in attempts_source:
            checkpoint = candidate.checkpoint
            if block == "test":
                destination = copied_directory / checkpoint.name
                if destination.is_file():
                    if file_digest(destination) != file_digest(checkpoint):
                        raise RuntimeError("copied frozen holdout checkpoint changed")
                else:
                    shutil.copy2(checkpoint, destination)
                checkpoint = destination
            item = dict(record)
            item.update({
                "checkpoint_path": checkpoint.relative_to(output).as_posix(),
                "checkpoint_sha256": file_digest(checkpoint),
                "selected_for_accepted_block": candidate.candidate_id in selected_slots,
                "accepted_slot": selected_slots.get(candidate.candidate_id, np.nan),
            })
            attempt_records.append(item)
            migration_records.append({
                "candidate_id": candidate.candidate_id,
                "candidate_index": candidate.candidate_index,
                "checkpoint_path": checkpoint.relative_to(output).as_posix(),
                "checkpoint_sha256": file_digest(checkpoint),
                "original_contract_hash": base_contract,
                "preexisting_checkpoint": True,
                "accepted": bool(record["accepted"]),
                "preserved_without_rewrite": True,
            })
        attempts = pd.DataFrame(attempt_records).sort_values(
            ["candidate_ordinal"], kind="stable",
        ).reset_index(drop=True)
        atomic_npz(
            output / "accepted_inputs.npz",
            decisions=decisions,
            influents=influents,
            source_candidate_id=provenance["source_candidate_id"].to_numpy(str),
            source_candidate_round=provenance["source_candidate_round"].to_numpy(int),
            source_candidate_index=provenance["source_candidate_index"].to_numpy(int),
            source_candidate_ordinal=provenance["source_candidate_ordinal"].to_numpy(int),
        )
        atomic_npz(
            output / "mechanistic_accepted_v3.npz",
            contract_hash=np.asarray("frozen_accepted_checkpoint_split_80_20_v1"),
            targets=targets,
            states_start_1=states_start_1,
            states_start_2=states_start_2,
        )
        atomic_dataframe(output / "accepted_diagnostics.csv", diagnostics)
        atomic_dataframe(output / "all_attempts.csv", attempts)
        atomic_dataframe(output / "accepted_provenance.csv", provenance)
        atomic_dataframe(output / "base_checkpoint_migration.csv", pd.DataFrame(migration_records))
        atomic_json(output / "replacement_summary.json", {
            "schema": replacement_generation.REPLACEMENT_SCHEMA,
            "block": block,
            "requested_accepted_count": len(selected),
            "accepted_count": len(selected),
            "base_attempt_count": len(attempts),
            "base_accepted_count": len(selected),
            "supplemental_attempt_count": 0,
            "supplemental_accepted_count": 0,
            "supplemental_round_count": 0,
            "initial_seed": source_profile.development_seed,
            "initial_final_state": int(source_design["generators"]["development"]["final_state"]),
            "initial_draw_count": int(source_design["generators"]["development"]["draw_count"]),
            "replacement_final_state": int(source_design["generators"]["development"]["final_state"]),
            "replacement_draw_count": 0,
            "total_stream_draw_count": int(source_design["generators"]["development"]["draw_count"]),
            "frozen_from_interrupted_run": True,
        })
        result = MechanisticBlockResult(
            decisions=decisions, influents=influents, targets=targets,
            diagnostics=diagnostics, attempts=attempts, provenance=provenance,
        )
        _validate_generation_block(
            targets, diagnostics, block=block, count=len(selected), profile=profile,
        )
        _validate_attempt_checkpoint_hashes(output, attempts)
        _write_generation_audits(output, result)
        publication_paths = _generation_publication_paths(output)
        elapsed = float(np.nansum(pd.to_numeric(
            attempts.get("elapsed_seconds", pd.Series(dtype=float)), errors="coerce",
        )))
        atomic_json(output / "block_complete.json", {
            "stage": "frozen_accepted_mechanistic_dataset",
            "block": block,
            "source_digest": source_id,
            "design_digest": design_id,
            "row_count": len(selected),
            "accepted_count": len(selected),
            "target_shape": list(targets.shape),
            "attempt_count": len(attempts),
            "rejected_attempt_count": int((~_boolean_series(
                attempts["accepted"], description="frozen attempt ledger",
            )).sum()),
            "replacement_slot_count": 0,
            "effective_input_digest": array_digest(
                decisions=np.asarray(decisions, dtype="<f8"),
                influents=np.asarray(influents, dtype="<f8"),
            ),
            "elapsed_seconds": elapsed,
            "artifacts": _artifact_hashes(run, publication_paths),
        })

    publish_block(
        "development", development_items,
        sorted(development_items + rejected_items, key=lambda item: item[0].candidate_index),
    )
    publish_block("test", test_items, test_items)
    partition_path = run / "inputs" / "frozen_accepted_partition.json"
    atomic_json(partition_path, {
        "schema": 1,
        "protocol": "frozen_accepted_checkpoint_split_80_20_v1",
        "source_run_id": "article_full_50000_001",
        "target_run_id": run.name,
        "completed_source_attempts": len(paths),
        "accepted_source_attempts": len(accepted_items),
        "rejected_source_attempts": len(rejected_items),
        "ordering": "ascending original development candidate index",
        "development_count": len(development_items),
        "test_count": len(test_items),
        "new_mechanistic_solves": 0,
        "rejected_rows_used_in_analysis": 0,
    })
    atomic_json(marker_path, {
        "stage": "frozen_accepted_dataset",
        "source_digest": source_id,
        "design_digest": design_id,
        "accepted_count": len(accepted_items),
        "development_count": len(development_items),
        "test_count": len(test_items),
        "artifacts": _artifact_hashes(run, (
            design_path, source_archive, partition_path,
            run / "datasets" / "development" / "block_complete.json",
            run / "datasets" / "test" / "block_complete.json",
        )),
    })
    assert_source_unchanged(source_files)
    return run_generation(
        run, frozen_design, profile=profile, source_files=source_files,
    )


def _ridge_input_digest(
    decisions: np.ndarray, influents: np.ndarray, targets: np.ndarray,
) -> str:
    return array_digest(
        development_decisions=np.asarray(decisions, dtype="<f8"),
        development_influents=np.asarray(influents, dtype="<f8"),
        development_targets=np.asarray(targets, dtype="<f8"),
    )


def _validate_ridge_scores(
    scores: pd.DataFrame, fold_membership: np.ndarray, row_count: int,
) -> None:
    required = {"fold", "gamma", "raw_nrmse", "selected"}
    if required - set(scores.columns):
        raise RuntimeError("ridge score checkpoint omits required columns")
    if len(scores) != 5 * len(RIDGE_GRID):
        raise RuntimeError("ridge score checkpoint does not contain the full 5-fold grid")
    if set(np.asarray(scores["fold"], dtype=int)) != {1, 2, 3, 4, 5}:
        raise RuntimeError("ridge score checkpoint has invalid fold identifiers")
    gamma = np.asarray(scores["gamma"], dtype=float)
    nrmse = np.asarray(scores["raw_nrmse"], dtype=float)
    if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(nrmse)):
        raise RuntimeError("ridge score checkpoint contains non-finite values")
    for candidate in RIDGE_GRID:
        if np.count_nonzero(np.isclose(gamma, candidate, rtol=1e-12, atol=0.0)) != 5:
            raise RuntimeError("ridge score checkpoint does not cover each penalty five times")
    selected = scores["selected"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if selected.isna().any() or int(selected.sum()) != 5:
        raise RuntimeError("ridge score checkpoint has an invalid selected penalty")
    chosen = gamma[selected.to_numpy()]
    if not np.allclose(chosen, chosen[0], rtol=0.0, atol=0.0):
        raise RuntimeError("ridge score checkpoint selects more than one penalty")
    membership = np.asarray(fold_membership, dtype=int)
    if membership.shape != (row_count,) or set(membership) != {1, 2, 3, 4, 5}:
        raise RuntimeError("ridge fold membership checkpoint is invalid")
    counts = np.bincount(membership, minlength=6)[1:]
    if int(counts.max() - counts.min()) > 1:
        raise RuntimeError("ridge folds do not form the declared balanced partition")


def save_ridge(
    run: Path, result: Any, *, input_id: str, source_id: str,
) -> None:
    model = result.model
    scores_path = run / "metrics" / "ridge_cross_validation.csv"
    fold_path = run / "metrics" / "ridge_fold_membership.csv"
    bundle_path = run / "models" / "ridge_surrogate.npz"
    atomic_dataframe(scores_path, result.scores)
    atomic_dataframe(fold_path, pd.DataFrame({
        "row": np.arange(len(result.fold_membership)),
        "fold": result.fold_membership,
    }))
    atomic_npz(
        bundle_path,
        input_digest=np.asarray(input_id),
        source_digest=np.asarray(source_id),
        decision_center=model.feature_map.decision_center,
        decision_scale=model.feature_map.decision_scale,
        influent_center=model.feature_map.influent_center,
        influent_scale=model.feature_map.influent_scale,
        term_center=model.feature_map.term_center,
        term_scale=model.feature_map.term_scale,
        variance_relative_tolerance=np.asarray(
            model.feature_map.variance_relative_tolerance
        ),
        response_center=model.response_center,
        response_scale=model.response_scale,
        coefficients=model.coefficients,
        ridge_penalty=np.asarray(model.ridge_penalty),
        diagnostics_json=np.asarray(json.dumps(asdict(model.diagnostics), sort_keys=True)),
        fold_membership=np.asarray(result.fold_membership, dtype=int),
        out_of_fold_raw=result.out_of_fold_raw,
        elapsed_seconds=np.asarray(result.elapsed_seconds),
    )
    paths = (scores_path, fold_path, bundle_path)
    atomic_json(run / "models" / "ridge_complete.json", {
        "stage": "ridge_cross_validation",
        "source_digest": source_id,
        "input_digest": input_id,
        "selected_penalty": model.ridge_penalty,
        "artifacts": _artifact_hashes(run, paths),
    })


def _load_ridge(
    run: Path,
    *,
    decisions: np.ndarray,
    influents: np.ndarray,
    targets: np.ndarray,
    input_id: str,
    source_id: str,
) -> tuple[QuadraticSurrogate, np.ndarray] | None:
    marker_path = run / "models" / "ridge_complete.json"
    bundle_path = run / "models" / "ridge_surrogate.npz"
    scores_path = run / "metrics" / "ridge_cross_validation.csv"
    fold_path = run / "metrics" / "ridge_fold_membership.csv"
    if not all(path.is_file() for path in (marker_path, bundle_path, scores_path, fold_path)):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            not _checkpoint_source_is_authorized(
                run, stage="ridge", checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("input_digest") != input_id
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return None
        with np.load(bundle_path, allow_pickle=False) as stored:
            if (
                str(stored["input_digest"].item()) != input_id
                or str(stored["source_digest"].item()) != marker_source_id
            ):
                return None
            feature_map = QuadraticFeatureMap(
                decision_center=np.asarray(stored["decision_center"], dtype=float),
                decision_scale=np.asarray(stored["decision_scale"], dtype=float),
                influent_center=np.asarray(stored["influent_center"], dtype=float),
                influent_scale=np.asarray(stored["influent_scale"], dtype=float),
                term_center=np.asarray(stored["term_center"], dtype=float),
                term_scale=np.asarray(stored["term_scale"], dtype=float),
                variance_relative_tolerance=float(stored["variance_relative_tolerance"]),
            )
            diagnostics = LeastSquaresDiagnostics(**json.loads(
                str(stored["diagnostics_json"].item())
            ))
            model = QuadraticSurrogate(
                feature_map=feature_map,
                response_center=np.asarray(stored["response_center"], dtype=float),
                response_scale=np.asarray(stored["response_scale"], dtype=float),
                coefficients=np.asarray(stored["coefficients"], dtype=float),
                diagnostics=diagnostics,
                ridge_penalty=float(stored["ridge_penalty"]),
            )
            oof = np.asarray(stored["out_of_fold_raw"], dtype=float)
            membership = np.asarray(stored["fold_membership"], dtype=int)
        expected_features = QuadraticFeatureMap.expected_feature_count(
            decisions.shape[1], influents.shape[1]
        )
        if (
            feature_map.decision_count != decisions.shape[1]
            or feature_map.influent_count != influents.shape[1]
            or feature_map.feature_count != expected_features
            or model.response_center.shape != (targets.shape[1],)
            or model.response_scale.shape != (targets.shape[1],)
            or model.coefficients.shape != (targets.shape[1], expected_features)
            or oof.shape != targets.shape
            or not np.all(np.isfinite(oof))
            or not all(np.all(np.isfinite(value)) for value in (
                feature_map.decision_center,
                feature_map.decision_scale,
                feature_map.influent_center,
                feature_map.influent_scale,
                feature_map.term_center,
                feature_map.term_scale,
                model.response_center,
                model.response_scale,
                model.coefficients,
            ))
            or not np.all(model.response_scale > 0.0)
            or not np.any(np.isclose(
                model.ridge_penalty, RIDGE_GRID, rtol=1e-12, atol=0.0,
            ))
        ):
            return None
        scores = pd.read_csv(scores_path)
        fold_frame = pd.read_csv(fold_path)
        if not np.array_equal(fold_frame["row"].to_numpy(), np.arange(len(decisions))):
            return None
        if not np.array_equal(fold_frame["fold"].to_numpy(dtype=int), membership):
            return None
        _validate_ridge_scores(scores, membership, len(decisions))
        return model, oof
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def fit_or_resume_ridge(
    run: Path,
    decisions: np.ndarray,
    influents: np.ndarray,
    targets: np.ndarray,
    *,
    source_files: Mapping[str, str],
) -> tuple[QuadraticSurrogate, np.ndarray, str]:
    input_id = _ridge_input_digest(decisions, influents, targets)
    source_id = source_digest(source_files)
    resumed = _load_ridge(
        run, decisions=decisions, influents=influents, targets=targets,
        input_id=input_id, source_id=source_id,
    )
    if resumed is not None:
        return resumed[0], resumed[1], input_id
    result = cross_validate_ridge(decisions, influents, targets)
    _validate_ridge_scores(result.scores, result.fold_membership, len(decisions))
    assert_source_unchanged(source_files)
    save_ridge(run, result, input_id=input_id, source_id=source_id)
    return result.model, result.out_of_fold_raw, input_id


def _validate_log_overflow_scores(
    scores: pd.DataFrame, fold_membership: np.ndarray, row_count: int,
) -> None:
    required = {"fold", "gamma", "log_rmse", "selected"}
    if required - set(scores.columns):
        raise RuntimeError("log-overflow score checkpoint omits required columns")
    if len(scores) != 5 * len(RIDGE_GRID):
        raise RuntimeError("log-overflow checkpoint does not contain the full 5-fold grid")
    if set(np.asarray(scores["fold"], dtype=int)) != {1, 2, 3, 4, 5}:
        raise RuntimeError("log-overflow checkpoint has invalid fold identifiers")
    gamma = np.asarray(scores["gamma"], dtype=float)
    error = np.asarray(scores["log_rmse"], dtype=float)
    if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(error)):
        raise RuntimeError("log-overflow score checkpoint contains non-finite values")
    for candidate in RIDGE_GRID:
        if np.count_nonzero(np.isclose(gamma, candidate, rtol=1e-12, atol=0.0)) != 5:
            raise RuntimeError("log-overflow checkpoint does not cover each penalty five times")
    selected = scores["selected"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if selected.isna().any() or int(selected.sum()) != 5:
        raise RuntimeError("log-overflow checkpoint has an invalid selected penalty")
    chosen = gamma[selected.to_numpy()]
    if not np.allclose(chosen, chosen[0], rtol=0.0, atol=0.0):
        raise RuntimeError("log-overflow checkpoint selects more than one penalty")
    membership = np.asarray(fold_membership, dtype=int)
    if membership.shape != (row_count,) or set(membership) != {1, 2, 3, 4, 5}:
        raise RuntimeError("log-overflow fold membership checkpoint is invalid")
    counts = np.bincount(membership, minlength=6)[1:]
    if int(counts.max() - counts.min()) > 1:
        raise RuntimeError("log-overflow folds do not form the declared balanced partition")


def save_log_overflow_closure(
    run: Path, result: Any, *, input_id: str, source_id: str,
) -> None:
    closure = result.closure
    model = closure.model
    scores_path = run / "metrics" / "log_overflow_closure_cross_validation.csv"
    fold_path = run / "metrics" / "log_overflow_closure_fold_membership.csv"
    bundle_path = run / "models" / "log_overflow_closure.npz"
    atomic_dataframe(scores_path, result.scores)
    atomic_dataframe(fold_path, pd.DataFrame({
        "row": np.arange(len(result.fold_membership)),
        "fold": result.fold_membership,
    }))
    atomic_npz(
        bundle_path,
        input_digest=np.asarray(input_id),
        source_digest=np.asarray(source_id),
        reference_concentration=np.asarray(closure.reference_concentration),
        decision_center=model.feature_map.decision_center,
        decision_scale=model.feature_map.decision_scale,
        influent_center=model.feature_map.influent_center,
        influent_scale=model.feature_map.influent_scale,
        term_center=model.feature_map.term_center,
        term_scale=model.feature_map.term_scale,
        variance_relative_tolerance=np.asarray(
            model.feature_map.variance_relative_tolerance
        ),
        response_center=model.response_center,
        response_scale=model.response_scale,
        coefficients=model.coefficients,
        ridge_penalty=np.asarray(model.ridge_penalty),
        diagnostics_json=np.asarray(json.dumps(asdict(model.diagnostics), sort_keys=True)),
        fold_membership=np.asarray(result.fold_membership, dtype=int),
        out_of_fold_log=result.out_of_fold_log,
        out_of_fold_tss=result.out_of_fold_tss,
        exact_overflow_tss=result.exact_overflow_tss,
        elapsed_seconds=np.asarray(result.elapsed_seconds),
    )
    paths = (scores_path, fold_path, bundle_path)
    atomic_json(run / "models" / "log_overflow_closure_complete.json", {
        "stage": "log_overflow_closure_cross_validation",
        "projection_schema": PROJECTION_SCHEMA,
        "source_digest": source_id,
        "input_digest": input_id,
        "selected_penalty": model.ridge_penalty,
        "reference_concentration_mg_L": closure.reference_concentration,
        "artifacts": _artifact_hashes(run, paths),
    })


def _load_log_overflow_closure(
    run: Path,
    *,
    decisions: np.ndarray,
    influents: np.ndarray,
    targets: np.ndarray,
    input_id: str,
    source_id: str,
    layout: NetworkLayout,
) -> tuple[LogOverflowTSSClosure, np.ndarray] | None:
    marker_path = run / "models" / "log_overflow_closure_complete.json"
    bundle_path = run / "models" / "log_overflow_closure.npz"
    scores_path = run / "metrics" / "log_overflow_closure_cross_validation.csv"
    fold_path = run / "metrics" / "log_overflow_closure_fold_membership.csv"
    if not all(path.is_file() for path in (marker_path, bundle_path, scores_path, fold_path)):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            marker.get("projection_schema") != PROJECTION_SCHEMA
            or not _checkpoint_source_is_authorized(
                run, stage="ridge", checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("input_digest") != input_id
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return None
        with np.load(bundle_path, allow_pickle=False) as stored:
            if (
                str(stored["input_digest"].item()) != input_id
                or str(stored["source_digest"].item()) != marker_source_id
            ):
                return None
            feature_map = QuadraticFeatureMap(
                decision_center=np.asarray(stored["decision_center"], dtype=float),
                decision_scale=np.asarray(stored["decision_scale"], dtype=float),
                influent_center=np.asarray(stored["influent_center"], dtype=float),
                influent_scale=np.asarray(stored["influent_scale"], dtype=float),
                term_center=np.asarray(stored["term_center"], dtype=float),
                term_scale=np.asarray(stored["term_scale"], dtype=float),
                variance_relative_tolerance=float(stored["variance_relative_tolerance"]),
            )
            diagnostics = LeastSquaresDiagnostics(**json.loads(
                str(stored["diagnostics_json"].item())
            ))
            model = QuadraticSurrogate(
                feature_map=feature_map,
                response_center=np.asarray(stored["response_center"], dtype=float),
                response_scale=np.asarray(stored["response_scale"], dtype=float),
                coefficients=np.asarray(stored["coefficients"], dtype=float),
                diagnostics=diagnostics,
                ridge_penalty=float(stored["ridge_penalty"]),
            )
            closure = LogOverflowTSSClosure(
                model=model,
                reference_concentration=float(stored["reference_concentration"]),
            )
            membership = np.asarray(stored["fold_membership"], dtype=int)
            oof_log = np.asarray(stored["out_of_fold_log"], dtype=float)
            oof_tss = np.asarray(stored["out_of_fold_tss"], dtype=float)
            exact = np.asarray(stored["exact_overflow_tss"], dtype=float)
        expected_features = QuadraticFeatureMap.expected_feature_count(
            decisions.shape[1], influents.shape[1]
        )
        expected_exact = overflow_tss_from_response(targets, decisions, layout)
        if (
            feature_map.decision_count != decisions.shape[1]
            or feature_map.influent_count != influents.shape[1]
            or feature_map.feature_count != expected_features
            or model.response_center.shape != (1,)
            or model.response_scale.shape != (1,)
            or model.coefficients.shape != (1, expected_features)
            or oof_log.shape != (len(decisions),)
            or oof_tss.shape != (len(decisions),)
            or exact.shape != (len(decisions),)
            or not np.all(np.isfinite(oof_log))
            or not np.all(np.isfinite(oof_tss))
            or np.any(oof_tss <= 0.0)
            or not np.allclose(exact, expected_exact, rtol=1e-12, atol=1e-12)
            or not np.allclose(
                oof_tss,
                closure.reference_concentration * np.exp(oof_log),
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.any(np.isclose(
                model.ridge_penalty, RIDGE_GRID, rtol=1e-12, atol=0.0,
            ))
        ):
            return None
        scores = pd.read_csv(scores_path)
        fold_frame = pd.read_csv(fold_path)
        if not np.array_equal(fold_frame["row"].to_numpy(), np.arange(len(decisions))):
            return None
        if not np.array_equal(fold_frame["fold"].to_numpy(dtype=int), membership):
            return None
        _validate_log_overflow_scores(scores, membership, len(decisions))
        return closure, oof_tss
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def fit_or_resume_log_overflow_closure(
    run: Path,
    decisions: np.ndarray,
    influents: np.ndarray,
    targets: np.ndarray,
    *,
    layout: NetworkLayout,
    source_files: Mapping[str, str],
) -> tuple[LogOverflowTSSClosure, np.ndarray, str]:
    input_id = array_digest(
        projection_schema=np.frombuffer(PROJECTION_SCHEMA.encode("utf-8"), dtype=np.uint8),
        development_decisions=np.asarray(decisions, dtype="<f8"),
        development_influents=np.asarray(influents, dtype="<f8"),
        development_targets=np.asarray(targets, dtype="<f8"),
    )
    source_id = source_digest(source_files)
    resumed = _load_log_overflow_closure(
        run,
        decisions=decisions,
        influents=influents,
        targets=targets,
        input_id=input_id,
        source_id=source_id,
        layout=layout,
    )
    if resumed is not None:
        return resumed[0], resumed[1], input_id
    result = cross_validate_log_overflow_closure(
        decisions, influents, targets, layout=layout,
    )
    _validate_log_overflow_scores(result.scores, result.fold_membership, len(decisions))
    assert_source_unchanged(source_files)
    save_log_overflow_closure(run, result, input_id=input_id, source_id=source_id)
    return result.closure, result.out_of_fold_tss, input_id


def _validate_assessment(
    assessment: AssessmentResult, *, test_count: int, response_count: int,
) -> None:
    arrays = (assessment.raw, assessment.projected, assessment.projected_targets)
    if any(
        np.asarray(value).shape != (test_count, response_count)
        or not np.all(np.isfinite(value))
        for value in arrays
    ):
        raise RuntimeError("post-selection holdout predictions are incomplete or non-finite")
    if assessment.overflow_tss_closure is not None:
        closure = np.asarray(assessment.overflow_tss_closure, dtype=float)
        if (
            closure.shape != (test_count,)
            or not np.all(np.isfinite(closure))
            or np.any(closure <= 0.0)
        ):
            raise RuntimeError("post-selection overflow-TSS closure predictions are invalid")
    qp = assessment.qp_diagnostics
    if len(qp) != 2 * test_count or set(qp["projection_input"]) != {
        "raw_prediction", "mechanistic_target",
    }:
        raise RuntimeError("projection audit does not cover both inputs for every test row")
    for kind in ("raw_prediction", "mechanistic_target"):
        subset = qp.loc[qp["projection_input"].eq(kind)]
        if len(subset) != test_count or set(np.asarray(subset["row"], dtype=int)) != set(
            range(test_count)
        ):
            raise RuntimeError(f"projection audit coverage is invalid for {kind}")
    violations = assessment.violations
    if len(violations) != 3 * test_count:
        raise RuntimeError("physical audit does not cover raw/projected/mechanistic results")
    for method in ("raw", "projected", "mechanistic"):
        if int(violations["method"].eq(method).sum()) != test_count:
            raise RuntimeError(f"physical audit coverage is invalid for {method}")
    if len(assessment.feasibility) != test_count:
        raise RuntimeError("finite-distance projection bound does not cover every test row")


def assessment_gate_allows_optimization(passed: bool) -> bool:
    """Return whether the recorded gate outcome permits downstream execution."""

    return bool(passed or ASSESSMENT_GATE_EXECUTION_POLICY == "advisory_continue")


def _post_selection_holdout_trust_diagnostics(
    model: QuadraticSurrogate,
    surrogate_assets: Any,
    decisions: np.ndarray,
    influents: np.ndarray,
    raw: np.ndarray,
    projected: np.ndarray,
) -> pd.DataFrame:
    """Evaluate the four frozen trust diagnostics on the holdout once.

    This is a descriptive post-selection calculation.  It reuses the stored
    raw and projected holdout responses and the scales/callbacks frozen from
    development; its values are not inputs to calibration or admission.
    """

    theta = np.asarray(decisions, dtype=float)
    feed = np.asarray(influents, dtype=float)
    raw_values = np.asarray(raw, dtype=float)
    projected_values = np.asarray(projected, dtype=float)
    if (
        theta.ndim != 2
        or feed.ndim != 2
        or len(theta) < 1
        or len(feed) != len(theta)
        or raw_values.ndim != 2
        or raw_values.shape != projected_values.shape
        or len(raw_values) != len(theta)
        or not all(
            np.all(np.isfinite(value))
            for value in (theta, feed, raw_values, projected_values)
        )
    ):
        raise RuntimeError("post-selection holdout trust inputs are invalid")
    response_scale = np.asarray(model.response_scale, dtype=float)
    if (
        response_scale.shape != (raw_values.shape[1],)
        or not np.all(np.isfinite(response_scale))
        or np.any(response_scale <= 0.0)
    ):
        raise RuntimeError("frozen surrogate response scales are invalid")
    features = np.asarray(model.feature_map.transform(theta, feed), dtype=float)
    leverage_precision = np.asarray(
        surrogate_assets.leverage_precision, dtype=float,
    )
    if (
        features.ndim != 2
        or features.shape[0] != len(theta)
        or leverage_precision.shape != (features.shape[1], features.shape[1])
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(leverage_precision))
    ):
        raise RuntimeError("frozen regularized-leverage assets are invalid")
    callbacks = surrogate_assets.trust_callbacks
    split_rows = getattr(callbacks, "split_rows", None)
    reactor_rows = getattr(callbacks, "reactor_rows", None)
    if not callable(split_rows) or not callable(reactor_rows):
        raise RuntimeError("the two frozen residual trust callbacks are unavailable")

    values = np.empty((len(theta), 4), dtype=float)
    values[:, 0] = np.sqrt(np.mean(
        ((projected_values - raw_values) / response_scale) ** 2, axis=1,
    ))
    values[:, 1] = np.einsum(
        "ij,jk,ik->i", features, leverage_precision, features,
    )
    for row in range(len(theta)):
        split = np.asarray(
            split_rows(
                theta[row], raw_values[row], projected_values[row], feed[row],
            ),
            dtype=float,
        ).reshape(-1)
        reactor = np.asarray(
            reactor_rows(
                theta[row], raw_values[row], projected_values[row], feed[row],
            ),
            dtype=float,
        ).reshape(-1)
        if (
            split.size < 1
            or reactor.size < 1
            or not np.all(np.isfinite(split))
            or not np.all(np.isfinite(reactor))
        ):
            raise RuntimeError(
                f"frozen residual trust callbacks failed at holdout row {row}"
            )
        values[row, 2] = float(np.sqrt(np.mean(split**2)))
        values[row, 3] = float(np.sqrt(np.mean(reactor**2)))
    if not np.all(np.isfinite(values)):
        raise RuntimeError("post-selection holdout trust diagnostics are non-finite")
    frame = pd.DataFrame(values, columns=[
        "correction", "regularized_leverage", "particulate_split",
        "reactor_residual",
    ])
    frame.insert(0, "row", np.arange(len(theta)))
    return frame


def evaluate_admission_gate(
    assessment: AssessmentResult,
    *,
    correction_limit: float,
    trust_limits: Mapping[str, float],
    development_oof_projection_accepted: np.ndarray,
    development_oof_complete_nrmse: float,
    development_oof_inventory_nrmse: float,
    development_oof_overflow_metrics_complete: bool = True,
    test_count: int,
) -> dict[str, Any]:
    _validate_assessment(
        assessment, test_count=test_count, response_count=assessment.raw.shape[1],
    )
    complete = assessment.metrics.loc[
        assessment.metrics["method"].eq("raw")
        & assessment.metrics["block"].eq("complete_response")
        & assessment.metrics["coordinate"].eq("ALL")
    ]
    if len(complete) != 1:
        raise RuntimeError("complete-response raw assessment row is missing or duplicated")
    holdout_raw_nrmse = float(complete.iloc[0]["nrmse"])
    if not np.isfinite(holdout_raw_nrmse):
        raise RuntimeError("post-selection holdout complete-response raw nRMSE is non-finite")
    if not (
        np.isfinite(development_oof_complete_nrmse)
        and np.isfinite(development_oof_inventory_nrmse)
    ):
        raise RuntimeError("development OOF response gates must be finite")
    qp_accepted = assessment.qp_diagnostics["accepted"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    feasibility = assessment.feasibility["bound_passed"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if qp_accepted.isna().any() or feasibility.isna().any():
        raise RuntimeError("projection acceptance columns are not Boolean")
    violations = assessment.violations
    physical: dict[str, dict[str, float | bool]] = {}
    for method in ("raw", "projected", "mechanistic"):
        rows = violations.loc[violations["method"].eq(method)]
        mass = float(rows["mass_conservation_violation_max"].max())
        negative = float(rows["nonnegativity_violation_max"].max())
        if not np.isfinite(mass) or not np.isfinite(negative):
            raise RuntimeError(f"{method} physical audit contains non-finite gate values")
        physical[method] = {
            "mass_conservation_violation_max": mass,
            "nonnegativity_violation_max": negative,
            "passed": bool(mass <= 1.0e-8 and negative <= 1.0e-10),
        }
    limits = {name: float(value) for name, value in trust_limits.items()}
    if set(limits) != {
        "correction", "regularized_leverage", "particulate_split",
        "reactor_residual",
    } or not all(np.isfinite(value) and value >= 0.0 for value in limits.values()):
        raise RuntimeError("four finite, nonnegative trust limits were not frozen")
    if not np.isclose(
        limits["correction"], correction_limit, rtol=0.0, atol=0.0,
    ):
        raise RuntimeError("the correction gate and frozen trust limit disagree")
    oof_projection_accepted = np.asarray(development_oof_projection_accepted)
    if (
        oof_projection_accepted.ndim != 1
        or oof_projection_accepted.size < 1
        or oof_projection_accepted.dtype.kind != "b"
    ):
        raise RuntimeError(
            "development OOF projection acceptance must be a nonempty Boolean vector"
        )
    admission_checks = {
        "development_oof_complete_response_nrmse_below_one": (
            development_oof_complete_nrmse < 1.0
        ),
        "development_oof_clarifier_inventory_nrmse_below_one": (
            development_oof_inventory_nrmse < 1.0
        ),
        "all_development_oof_projection_qp_audits_passed": bool(
            oof_projection_accepted.all()
        ),
        "development_oof_log_overflow_metrics_complete": bool(
            development_oof_overflow_metrics_complete
        ),
        "all_four_trust_limits_frozen": True,
        "correction_limit_at_most_0_50": bool(correction_limit <= 0.50),
    }
    holdout_diagnostics = {
        "all_projection_qp_audits_passed": bool(qp_accepted.all()),
        "all_finite_distance_bounds_passed": bool(feasibility.all()),
        "projected_physical_audits_passed": bool(physical["projected"]["passed"]),
        "mechanistic_physical_audits_passed": bool(physical["mechanistic"]["passed"]),
    }
    passed = bool(all(admission_checks.values()))
    optimization_permitted = assessment_gate_allows_optimization(passed)
    return {
        "passed": passed,
        "execution_policy": ASSESSMENT_GATE_EXECUTION_POLICY,
        "optimization_permitted": optimization_permitted,
        "post_selection_holdout_raw_complete_response_nrmse": holdout_raw_nrmse,
        "development_oof_complete_response_nrmse": development_oof_complete_nrmse,
        "development_oof_clarifier_inventory_nrmse": development_oof_inventory_nrmse,
        "post_selection_holdout_is_confirmatory": False,
        "admission_gate_scope": "development_only",
        "post_selection_holdout_checks_are_admission_gates": False,
        **admission_checks,
        **holdout_diagnostics,
        "trust_limits": limits,
        "physical_audit_maxima": physical,
        "failure_action": (
            None if passed else
            "record advisory failure and continue without refitting"
        ),
    }


def _assessment_binding(
    design: Mapping[str, object],
    development_targets: np.ndarray,
    test_targets: np.ndarray,
) -> str:
    shared_count = (N_STAGES + 3) * N_COMPONENTS
    layer_count = int(development_targets.shape[1] - shared_count)
    if layer_count < 1 or test_targets.shape[1] != shared_count + layer_count:
        raise RuntimeError("mechanistic response blocks have inconsistent dimensions")
    development_reduced = reduce_mechanistic_responses(
        development_targets, layer_count,
    )
    test_reduced = reduce_mechanistic_responses(test_targets, layer_count)
    return array_digest(
        development_decisions=np.asarray(design["development_decisions"], dtype="<f8"),
        development_influents=np.asarray(design["development_influents"], dtype="<f8"),
        development_targets=np.asarray(development_targets, dtype="<f8"),
        development_reduced=np.asarray(development_reduced, dtype="<f8"),
        test_decisions=np.asarray(design["test_decisions"], dtype="<f8"),
        test_influents=np.asarray(design["test_influents"], dtype="<f8"),
        test_targets=np.asarray(test_targets, dtype="<f8"),
        test_reduced=np.asarray(test_reduced, dtype="<f8"),
        response_schema=np.frombuffer(RESPONSE_SCHEMA.encode("utf-8"), dtype=np.uint8),
        projection_schema=np.frombuffer(
            PROJECTION_SCHEMA.encode("utf-8"), dtype=np.uint8,
        ),
    )


def _materialize_reduced_response_block(
    run: Path,
    *,
    block: str,
    mechanistic_targets: np.ndarray,
    profile: StudyProfile,
) -> np.ndarray:
    """Derive and immutably checkpoint the statistical response block."""

    full = np.asarray(mechanistic_targets, dtype=np.float64)
    if full.ndim != 2 or full.shape[1] != profile.mechanistic_response_count:
        raise RuntimeError(f"{block} mechanistic targets have the wrong response width")
    layer_volumes = np.full(
        profile.layer_count, 6_000.0 / profile.layer_count, dtype=np.float64,
    )
    reduced = reduce_mechanistic_responses(
        full, profile.layer_count, layer_volumes_m3=layer_volumes,
    )
    if reduced.shape != (len(full), profile.surrogate_response_count):
        raise RuntimeError(f"{block} reduced response transformation has the wrong shape")
    path = run / "datasets" / block / "surrogate_responses_inventory_v1.npz"
    full_digest = array_digest(mechanistic_targets=np.asarray(full, dtype="<f8"))
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            valid = bool(
                set(stored.files) == {
                    "schema", "mechanistic_target_digest", "layer_volumes_m3",
                    "responses",
                }
                and str(stored["schema"].item()) == RESPONSE_SCHEMA
                and str(stored["mechanistic_target_digest"].item()) == full_digest
                and np.array_equal(stored["layer_volumes_m3"], layer_volumes)
                and np.array_equal(stored["responses"], reduced)
            )
        if not valid:
            raise RuntimeError(f"existing {block} reduced-response artifact is inconsistent")
        return reduced
    atomic_npz(
        path,
        schema=np.asarray(RESPONSE_SCHEMA),
        mechanistic_target_digest=np.asarray(full_digest),
        layer_volumes_m3=layer_volumes,
        responses=reduced,
    )
    return reduced


def load_assessment_checkpoint(
    run: Path, *, source_id: str, input_id: str,
) -> dict[str, Any] | None:
    marker_path = run / "metrics" / "assessment_complete.json"
    gate_path = run / "metrics" / "admission_gate.json"
    if not marker_path.is_file() or not gate_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_source_id = str(marker.get("source_digest", ""))
        if (
            not _checkpoint_source_is_authorized(
                run,
                stage="assessment",
                checkpoint=marker_path,
                observed_source_id=marker_source_id,
                current_source_id=source_id,
            )
            or marker.get("input_digest") != input_id
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return None
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            not isinstance(gate.get("passed"), bool)
            or gate.get("execution_policy") != ASSESSMENT_GATE_EXECUTION_POLICY
            or gate.get("optimization_permitted")
            != assessment_gate_allows_optimization(gate["passed"])
        ):
            return None
        return gate
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def run_assessment(
    run: Path,
    design: Mapping[str, object],
    development_targets: np.ndarray,
    test_targets: np.ndarray,
    *,
    profile: StudyProfile,
    source_files: Mapping[str, str],
) -> AnalysisBundle:
    development_decisions = np.asarray(design["development_decisions"])
    development_influents = np.asarray(design["development_influents"])
    development_reduced = _materialize_reduced_response_block(
        run,
        block="development",
        mechanistic_targets=development_targets,
        profile=profile,
    )
    test_reduced = _materialize_reduced_response_block(
        run,
        block="test",
        mechanistic_targets=test_targets,
        profile=profile,
    )
    input_id = _assessment_binding(design, development_targets, test_targets)
    source_id = source_digest(source_files)
    existing_gate = load_assessment_checkpoint(
        run, source_id=source_id, input_id=input_id,
    )
    model, oof_raw, _ = fit_or_resume_ridge(
        run, development_decisions, development_influents, development_reduced,
        source_files=source_files,
    )
    layout = NetworkLayout(layer_count=profile.layer_count)
    overflow_closure, oof_overflow_tss, _ = fit_or_resume_log_overflow_closure(
        run,
        development_decisions,
        development_influents,
        development_reduced,
        layout=layout,
        source_files=source_files,
    )
    direct_assets = fit_direct_assets(
        development_decisions, development_influents, development_targets,
        clarifier=clarifier_for(profile),
    )
    trust = calibrate_trust_diagnostics(
        model, development_decisions, development_influents, development_reduced,
        oof_raw, direct_assets, layout=layout,
        overflow_closure=overflow_closure,
        out_of_fold_overflow_tss=oof_overflow_tss,
    )
    surrogate_assets = build_surrogate_assets(
        model, development_decisions, development_influents, development_reduced,
        layout=layout,
        correction_rms_threshold=trust.correction_limit,
        trust_callbacks=trust.callbacks,
        split_rms_threshold=trust.split_limit,
        reactor_rms_threshold=trust.reactor_limit,
        overflow_closure=overflow_closure,
        development_overflow_tss_closure=oof_overflow_tss,
    )
    features = model.feature_map.transform(development_decisions, development_influents)
    leverage = np.einsum(
        "ij,jk,ik->i", features, surrogate_assets.leverage_precision, features,
    )
    trust_values = np.column_stack((
        trust.development_values[:, 0], leverage, trust.development_values[:, 1:],
    ))
    if trust_values.shape != (profile.development_count, 4) or not np.all(
        np.isfinite(trust_values)
    ):
        raise RuntimeError("four development trust diagnostics were not evaluated")
    limits = {
        "correction": float(trust.correction_limit),
        "regularized_leverage": float(
            surrogate_assets.trust_thresholds.regularized_leverage
        ),
        "particulate_split": float(trust.split_limit),
        "reactor_residual": float(trust.reactor_limit),
    }
    if existing_gate is not None:
        return AnalysisBundle(
            passed=bool(existing_gate["passed"]), model=model,
            direct_assets=direct_assets, surrogate_assets=surrogate_assets,
            assessment=None, gate=existing_gate,
            overflow_closure=overflow_closure,
        )
    trust_frame = pd.DataFrame(trust_values, columns=[
        "correction", "regularized_leverage", "particulate_split",
        "reactor_residual",
    ])
    trust_frame.insert(0, "row", np.arange(profile.development_count))
    trust_frame.insert(
        1, "projection_qp_accepted", trust.out_of_fold_projection_accepted,
    )
    atomic_dataframe(run / "metrics" / "trust_development_oof.csv", trust_frame)
    atomic_json(run / "metrics" / "trust_limits.json", limits)
    atomic_npz(
        run / "models" / "trust_calibration.npz",
        development_values=trust_values,
        out_of_fold_projected=trust.out_of_fold_projected,
        out_of_fold_projection_accepted=trust.out_of_fold_projection_accepted,
        split_scale=trust.split_scale,
    )
    assessment = assess_raw_projected_mechanistic(
        model,
        development_decisions,
        development_influents,
        development_reduced,
        np.asarray(design["test_decisions"]),
        np.asarray(design["test_influents"]),
        test_reduced,
        profile,
        overflow_closure=overflow_closure,
        development_overflow_tss_closure=oof_overflow_tss,
    )
    _validate_assessment(
        assessment, test_count=profile.test_count,
        response_count=profile.surrogate_response_count,
    )
    holdout_trust_path = run / "metrics" / "trust_post_selection_holdout.csv"
    paths = (
        run / "metrics" / "post_selection_prediction_metrics.csv",
        run / "metrics" / "physical_violations_assessment.csv",
        run / "metrics" / "projection_qp_diagnostics.csv",
        run / "metrics" / "projection_feasibility_bound.csv",
        run / "predictions" / "post_selection_holdout.npz",
        run / "metrics" / "trust_development_oof.csv",
        run / "metrics" / "trust_limits.json",
        run / "models" / "trust_calibration.npz",
        run / "metrics" / "admission_gate.json",
        run / "models" / "ridge_surrogate.npz",
        run / "models" / "ridge_complete.json",
        run / "metrics" / "ridge_cross_validation.csv",
        run / "metrics" / "ridge_fold_membership.csv",
        run / "datasets" / "development" / "surrogate_responses_inventory_v1.npz",
        run / "datasets" / "test" / "surrogate_responses_inventory_v1.npz",
        holdout_trust_path,
        run / "models" / "log_overflow_closure.npz",
        run / "models" / "log_overflow_closure_complete.json",
        run / "metrics" / "log_overflow_closure_cross_validation.csv",
        run / "metrics" / "log_overflow_closure_fold_membership.csv",
        run / "metrics" / "log_overflow_closure_development_oof.json",
    )
    atomic_dataframe(paths[0], assessment.metrics)
    atomic_dataframe(paths[1], assessment.violations)
    atomic_dataframe(paths[2], assessment.qp_diagnostics)
    atomic_dataframe(paths[3], assessment.feasibility)
    atomic_npz(
        paths[4], raw=assessment.raw, projected=assessment.projected,
        projected_targets=assessment.projected_targets,
        mechanistic=test_reduced,
        mechanistic_full=test_targets,
        overflow_tss_closure=assessment.overflow_tss_closure,
    )
    oof_scaled = (oof_raw - development_reduced) / model.response_scale
    development_oof_complete_nrmse = float(np.sqrt(np.mean(oof_scaled**2)))
    development_oof_inventory_nrmse = float(
        np.sqrt(np.mean(oof_scaled[:, layout.inventory_index] ** 2))
    )
    exact_development_overflow = overflow_tss_from_response(
        development_reduced, development_decisions, layout,
    )
    oof_closure_error = oof_overflow_tss - exact_development_overflow
    oof_log_error = np.log(oof_overflow_tss) - np.log(exact_development_overflow)
    oof_closure_metrics = {
        "sample_count": int(len(oof_overflow_tss)),
        "rmse_mg_L": float(np.sqrt(np.mean(np.square(oof_closure_error)))),
        "mae_mg_L": float(np.mean(np.abs(oof_closure_error))),
        "bias_mg_L": float(np.mean(oof_closure_error)),
        "log_rmse": float(np.sqrt(np.mean(np.square(oof_log_error)))),
        "log_bias": float(np.mean(oof_log_error)),
        "minimum_prediction_mg_L": float(np.min(oof_overflow_tss)),
        "maximum_prediction_mg_L": float(np.max(oof_overflow_tss)),
        "all_finite_and_positive": bool(
            np.all(np.isfinite(oof_overflow_tss)) and np.all(oof_overflow_tss > 0.0)
        ),
    }
    atomic_json(paths[-1], oof_closure_metrics)
    gate = evaluate_admission_gate(
        assessment, correction_limit=trust.correction_limit,
        trust_limits=limits,
        development_oof_projection_accepted=(
            trust.out_of_fold_projection_accepted
        ),
        development_oof_complete_nrmse=development_oof_complete_nrmse,
        development_oof_inventory_nrmse=development_oof_inventory_nrmse,
        development_oof_overflow_metrics_complete=bool(
            oof_closure_metrics["all_finite_and_positive"]
            and all(
                np.isfinite(float(oof_closure_metrics[name]))
                for name in (
                    "rmse_mg_L", "mae_mg_L", "bias_mg_L", "log_rmse", "log_bias",
                )
            )
        ),
        test_count=profile.test_count,
    )
    holdout_trust = _post_selection_holdout_trust_diagnostics(
        model,
        surrogate_assets,
        np.asarray(design["test_decisions"]),
        np.asarray(design["test_influents"]),
        assessment.raw,
        assessment.projected,
    )
    atomic_dataframe(holdout_trust_path, holdout_trust)
    atomic_json(paths[8], gate)
    assert_source_unchanged(source_files)
    atomic_json(run / "metrics" / "assessment_complete.json", {
        "stage": "post_selection_holdout_assessment",
        "source_digest": source_id,
        "input_digest": input_id,
        "passed": gate["passed"],
        "artifacts": _artifact_hashes(run, paths),
    })
    return AnalysisBundle(
        passed=bool(gate["passed"]), model=model,
        direct_assets=direct_assets, surrogate_assets=surrogate_assets,
        assessment=assessment, gate=gate,
        overflow_closure=overflow_closure,
    )


def _raw_inference_batch(
    model: QuadraticSurrogate,
    decisions: np.ndarray,
    influents: np.ndarray,
    overflow_closure: LogOverflowTSSClosure | None = None,
) -> np.ndarray:
    raw = np.asarray(model.predict(decisions, influents), dtype=float)
    if overflow_closure is not None:
        closure = np.asarray(overflow_closure.predict(decisions, influents), dtype=float)
        if closure.shape != (len(raw),) or not np.all(np.isfinite(closure)):
            raise RuntimeError("timed overflow-TSS closure inference returned invalid output")
    return raw


def _projection_inference_batch(
    cached_raw: np.ndarray,
    decisions: np.ndarray,
    influents: np.ndarray,
    projector: PhysicalProjector,
    layout: NetworkLayout,
    overflow_tss_closure: np.ndarray | None = None,
) -> int:
    """Project one cached-raw batch without evaluating the surrogate."""

    accepted = 0
    if overflow_tss_closure is not None:
        closure = np.asarray(overflow_tss_closure, dtype=float)
        if closure.shape != (len(cached_raw),) or np.any(closure <= 0.0):
            raise RuntimeError("cached overflow-TSS closure predictions are invalid")
    else:
        closure = None
    for row in range(len(cached_raw)):
        theta = decisions[row]
        operators = build_network_operators(
            influents[row],
            internal_recycle=float(theta[4]),
            return_recycle=float(theta[5]),
            waste_fraction=float(theta[6]),
            invariant_operator=INVARIANT_MATRIX,
            tss_weights=TSS_VECTOR,
            layout=layout,
            overflow_tss_closure=(None if closure is None else float(closure[row])),
        )
        result = projector.project(
            cached_raw[row],
            operators.equality_matrix,
            operators.equality_rhs,
            operators.inequality_matrix,
            warm_start=None,
            raise_on_failure=False,
        )
        state = np.asarray(result.state, dtype=float)
        if state.shape != cached_raw[row].shape or not np.all(np.isfinite(state)):
            raise RuntimeError(
                f"timed cached-raw projection returned an invalid state at test row {row}"
            )
        accepted += int(bool(result.accepted))
    return accepted


def _timing_contract_id(
    *,
    source_id: str,
    analysis_id: str,
    decisions: np.ndarray,
    influents: np.ndarray,
    cached_raw: np.ndarray,
    cached_overflow_tss_closure: np.ndarray | None = None,
) -> str:
    digest = sha256()
    digest.update(b"article-v3-inference-timing-v1\0")
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(str(INFERENCE_TIMING_WARMUPS).encode())
    digest.update(str(INFERENCE_TIMING_BATCHES).encode())
    arrays: dict[str, np.ndarray] = {
        "test_decisions": np.asarray(decisions, dtype="<f8"),
        "test_influents": np.asarray(influents, dtype="<f8"),
        "cached_raw": np.asarray(cached_raw, dtype="<f8"),
    }
    if cached_overflow_tss_closure is not None:
        arrays["cached_overflow_tss_closure"] = np.asarray(
            cached_overflow_tss_closure, dtype="<f8",
        )
    digest.update(array_digest(
        **arrays,
    ).encode())
    return digest.hexdigest()


def _load_cached_assessment_raw(
    run: Path,
    analysis: AnalysisBundle,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    path = run / "predictions" / "post_selection_holdout.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            value = np.asarray(stored["raw"], dtype=float)
    elif analysis.assessment is not None:
        value = np.asarray(analysis.assessment.raw, dtype=float)
    else:
        raise RuntimeError("cached post-selection holdout raw predictions are unavailable")
    if value.shape != expected_shape or not np.all(np.isfinite(value)):
        raise RuntimeError("cached post-selection holdout raw predictions are invalid")
    return value


def _run_inference_timing_benchmark(
    run: Path,
    design: Mapping[str, object],
    analysis: AnalysisBundle,
    *,
    source_files: Mapping[str, str],
    analysis_id: str,
) -> pd.DataFrame:
    """Retired: inference timing is not part of the article workflow."""

    raise RuntimeError(
        "untouched-test repeated inference timing is retired; use the completed "
        "robustness-case timing aggregation"
    )

    decisions = np.asarray(design["test_decisions"], dtype=float)
    influents = np.asarray(design["test_influents"], dtype=float)
    if decisions.ndim != 2 or decisions.shape[1] != 7:
        raise RuntimeError("timing test decisions must have seven columns")
    if influents.ndim != 2 or len(influents) != len(decisions):
        raise RuntimeError("timing test influents do not match the decision rows")
    row_count = len(decisions)
    if row_count < 1:
        raise RuntimeError("inference timing requires at least one test row")
    cached_raw = _load_cached_assessment_raw(
        run, analysis, (row_count, analysis.model.response_count),
    )
    cached_overflow_tss_closure = (
        None if analysis.overflow_closure is None else np.asarray(
            analysis.overflow_closure.predict(decisions, influents), dtype=float,
        )
    )
    source_id = source_digest(source_files)
    contract = _timing_contract_id(
        source_id=source_id,
        analysis_id=analysis_id,
        decisions=decisions,
        influents=influents,
        cached_raw=cached_raw,
        cached_overflow_tss_closure=cached_overflow_tss_closure,
    )
    marker_path = run / "metrics" / "inference_timing_complete.json"
    batches_path = run / "metrics" / "inference_timing_batches.csv"
    events_path = run / "metrics" / "timing_events.csv"
    summary_path = run / "metrics" / "inference_timing_summary.json"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("timing_contract") == contract
                and marker.get("source_digest") == source_id
                and marker.get("input_digest") == analysis_id
                and int(marker.get("row_count", -1)) == row_count
                and _artifacts_match(run, marker.get("artifacts", {}))
            ):
                frame = pd.read_csv(batches_path)
                counts = frame.groupby("category").size().to_dict()
                if counts == {
                    "qp_deployment": INFERENCE_TIMING_BATCHES,
                    "raw_inference": INFERENCE_TIMING_BATCHES,
                }:
                    return frame
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    checkpoints = run / "timing" / "inference_batches"
    verification_path = checkpoints / "cached_raw_verification.json"
    verification: dict[str, Any] | None = None
    if verification_path.is_file():
        try:
            candidate = json.loads(verification_path.read_text(encoding="utf-8"))
            if (
                candidate.get("timing_contract") == contract
                and candidate.get("passed") is True
                and float(candidate.get("maximum_scaled_difference", np.inf))
                <= 1.0e-12
            ):
                verification = candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for category in ("raw_inference", "qp_deployment"):
        for batch in range(INFERENCE_TIMING_BATCHES):
            path = checkpoints / f"{category}_{batch:02d}.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("timing_contract") == contract
                    and payload.get("category") == category
                    and int(payload.get("batch", -1)) == batch
                    and int(payload.get("response_count", -1)) == row_count
                    and int(payload.get("warmup_count", -1))
                    == INFERENCE_TIMING_WARMUPS
                    and float(payload.get("elapsed_seconds", -1.0)) >= 0.0
                    and np.isclose(
                        float(payload.get("per_response_latency_seconds", -1.0)),
                        float(payload["elapsed_seconds"]) / row_count,
                        rtol=1.0e-12,
                        atol=0.0,
                    )
                    and (
                        category != "qp_deployment"
                        or (
                            0 <= int(payload.get("projection_accepted_count", -1))
                            <= row_count
                            and np.isclose(
                                float(payload.get(
                                    "projection_accepted_fraction", np.nan,
                                )),
                                int(payload["projection_accepted_count"]) / row_count,
                                rtol=0.0,
                                atol=1.0e-15,
                            )
                        )
                    )
                ):
                    records[(category, batch)] = payload
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass

    projector = PhysicalProjector(
        analysis.model.response_scale,
        analysis.surrogate_assets.row_scales.equality,
        analysis.surrogate_assets.row_scales.inequality,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
        maximum_iterations=100_000,
        polish=True,
    )

    def publish(
        category: str, batch: int, elapsed_ns: int,
        *, projection_accepted_count: int | None = None,
    ) -> None:
        elapsed = float(elapsed_ns) / 1.0e9
        payload = {
            "timing_contract": contract,
            "category": category,
            "batch": batch,
            "elapsed_ns": int(elapsed_ns),
            "elapsed_seconds": elapsed,
            "per_response_latency_seconds": elapsed / row_count,
            "response_count": row_count,
            "warmup_count": INFERENCE_TIMING_WARMUPS,
            "raw_inference_included": category == "raw_inference",
            "fixed_row_order": True,
        }
        if category == "qp_deployment":
            if (
                projection_accepted_count is None
                or not 0 <= projection_accepted_count <= row_count
            ):
                raise RuntimeError("projection timing acceptance count is invalid")
            payload.update({
                "projection_accepted_count": int(projection_accepted_count),
                "projection_accepted_fraction": (
                    float(projection_accepted_count) / row_count
                ),
            })
        atomic_json(
            checkpoints / f"{category}_{batch:02d}.json", payload,
        )
        records[(category, batch)] = payload

    with threadpool_limits(limits=1):
        missing_raw = [
            batch for batch in range(INFERENCE_TIMING_BATCHES)
            if ("raw_inference", batch) not in records
        ]
        verification_was_warmup = False
        if verification is None:
            reproduced = _raw_inference_batch(
                analysis.model, decisions, influents, analysis.overflow_closure,
            )
            if reproduced.shape != cached_raw.shape or not np.all(
                np.isfinite(reproduced)
            ):
                raise RuntimeError("cached-raw verification returned invalid output")
            scaled_difference = float(np.max(
                np.abs(reproduced - cached_raw)
                / analysis.model.response_scale[None, :]
            ))
            if scaled_difference > 1.0e-12:
                raise RuntimeError(
                    "cached assessment raw predictions do not reproduce under "
                    f"the timed model (scaled inf={scaled_difference:.3e})"
                )
            verification = {
                "timing_contract": contract,
                "passed": True,
                "maximum_scaled_difference": scaled_difference,
            }
            atomic_json(verification_path, verification)
            verification_was_warmup = bool(missing_raw)
        if missing_raw:
            remaining_warmups = (
                INFERENCE_TIMING_WARMUPS - 1
                if verification_was_warmup else INFERENCE_TIMING_WARMUPS
            )
            for _ in range(remaining_warmups):
                warm = _raw_inference_batch(
                    analysis.model, decisions, influents, analysis.overflow_closure,
                )
                if warm.shape != cached_raw.shape or not np.all(np.isfinite(warm)):
                    raise RuntimeError("raw inference warmup returned invalid output")
            for batch in missing_raw:
                started = perf_counter_ns()
                raw = _raw_inference_batch(
                    analysis.model, decisions, influents, analysis.overflow_closure,
                )
                elapsed_ns = perf_counter_ns() - started
                if raw.shape != cached_raw.shape or not np.all(np.isfinite(raw)):
                    raise RuntimeError("timed raw inference returned invalid output")
                publish("raw_inference", batch, elapsed_ns)

        missing_projection = [
            batch for batch in range(INFERENCE_TIMING_BATCHES)
            if ("qp_deployment", batch) not in records
        ]
        if missing_projection:
            for _ in range(INFERENCE_TIMING_WARMUPS):
                _projection_inference_batch(
                    cached_raw, decisions, influents, projector,
                    analysis.surrogate_assets.layout,
                    cached_overflow_tss_closure,
                )
            for batch in missing_projection:
                started = perf_counter_ns()
                accepted = _projection_inference_batch(
                    cached_raw, decisions, influents, projector,
                    analysis.surrogate_assets.layout,
                    cached_overflow_tss_closure,
                )
                elapsed_ns = perf_counter_ns() - started
                publish(
                    "qp_deployment", batch, elapsed_ns,
                    projection_accepted_count=accepted,
                )

    ordered = [
        records[(category, batch)]
        for category in ("raw_inference", "qp_deployment")
        for batch in range(INFERENCE_TIMING_BATCHES)
    ]
    frame = pd.DataFrame(ordered)
    atomic_dataframe(batches_path, frame)
    # Reporting uses this generic event stream for one-time and route timings.
    atomic_dataframe(events_path, frame)
    summary: dict[str, Any] = {
        "timing_contract": contract,
        "warmup_count": INFERENCE_TIMING_WARMUPS,
        "timed_batch_count_per_route": INFERENCE_TIMING_BATCHES,
        "response_count_per_batch": row_count,
        "raw_inference_excluded_from_qp_timing": True,
        "cached_raw_reproduction_scaled_inf": verification[
            "maximum_scaled_difference"
        ],
        "categories": {},
    }
    for category, group in frame.groupby("category", sort=True):
        values = np.asarray(group["per_response_latency_seconds"], dtype=float)
        summary["categories"][str(category)] = {
            "count": len(values),
            "total": float(np.sum(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }
        if str(category) == "qp_deployment":
            accepted_counts = np.asarray(
                group["projection_accepted_count"], dtype=int,
            )
            accepted_total = int(np.sum(accepted_counts))
            attempted_total = int(row_count * len(group))
            summary["categories"][str(category)].update({
                "projection_accepted_count": accepted_total,
                "projection_attempt_count": attempted_total,
                "projection_accepted_fraction": (
                    float(accepted_total) / attempted_total
                ),
                "projection_accepted_count_per_batch_minimum": int(
                    np.min(accepted_counts)
                ),
                "projection_accepted_count_per_batch_maximum": int(
                    np.max(accepted_counts)
                ),
            })
    atomic_json(summary_path, summary)
    assert_source_unchanged(source_files)
    batch_checkpoints = tuple(
        checkpoints / f"{category}_{batch:02d}.json"
        for category in ("raw_inference", "qp_deployment")
        for batch in range(INFERENCE_TIMING_BATCHES)
    )
    paths = (
        batches_path, events_path, summary_path, verification_path,
        *batch_checkpoints,
    )
    atomic_json(marker_path, {
        "stage": "inference_timing_benchmark",
        "timing_contract": contract,
        "source_digest": source_id,
        "input_digest": analysis_id,
        "row_count": row_count,
        "warmup_count": INFERENCE_TIMING_WARMUPS,
        "timed_batch_count_per_route": INFERENCE_TIMING_BATCHES,
        "artifacts": _artifact_hashes(run, paths),
    })
    return frame


def _robustness_timing_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {
            "count": 0, "total": None, "mean": None, "median": None,
            "q25": None, "q75": None, "p95_nearest_rank": None,
            "maximum": None,
        }
    ordered = np.sort(finite)
    p95_index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "count": int(len(ordered)),
        "total": float(np.sum(ordered)),
        "mean": float(np.mean(ordered)),
        "median": float(np.median(ordered)),
        "q25": float(np.quantile(ordered, 0.25)),
        "q75": float(np.quantile(ordered, 0.75)),
        "p95_nearest_rank": float(ordered[p95_index]),
        "maximum": float(ordered[-1]),
    }


def _run_robustness_case_timing_aggregation(
    run: Path,
    *,
    source_files: Mapping[str, str],
    analysis_id: str,
) -> pd.DataFrame:
    """Summarize existing optimization timings over the ten robustness cases."""

    source_id = source_digest(source_files)
    cases = tuple(f"robustness_{index:02d}" for index in range(1, 11))
    input_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for case_id in cases:
        case_directory = run / "optimization" / case_id
        comparison_marker = case_directory / "casewise_comparison_complete.json"
        marker = _load_json_object(
            comparison_marker, description=f"{case_id} casewise comparison marker",
        )
        if (
            marker.get("case") != case_id
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            raise RuntimeError(f"completed casewise result changed: {case_id}")
        input_paths.append(comparison_marker)
        for route in ("surrogate", "direct"):
            route_path = case_directory / f"{route}.json"
            reference_path = case_directory / f"{route}_casewise_reference.json"
            route_payload = _load_json_object(
                route_path, description=f"{case_id} {route} route timing",
            )
            reference_payload = _load_json_object(
                reference_path, description=f"{case_id} {route} reference timing",
            )
            input_paths.extend((route_path, reference_path))
            primary_seconds = float(route_payload["elapsed_seconds"])
            certification_seconds = 0.0
            recovery_seconds = 0.0
            if route == "surrogate":
                certification_path = (
                    case_directory / "surrogate_local_convergence.json"
                )
                certification = _load_json_object(
                    certification_path,
                    description=f"{case_id} surrogate certification timing",
                )
                input_paths.append(certification_path)
                certificate = certification.get("certificate")
                if not isinstance(certificate, Mapping):
                    raise RuntimeError(f"{case_id} omits its certification timing")
                certification_seconds = float(certificate["elapsed_seconds"])
            else:
                recovery = reference_payload.get("recovery")
                if isinstance(recovery, Mapping) and recovery.get("attempted") is True:
                    recovery_seconds = float(recovery["elapsed_seconds"])
            complete_seconds = (
                primary_seconds + certification_seconds + recovery_seconds
            )
            reported_complete = reference_payload.get("optimization_elapsed_seconds")
            if reported_complete is not None and not np.isclose(
                complete_seconds, float(reported_complete), rtol=1.0e-12, atol=1.0e-9,
            ):
                raise RuntimeError(f"{case_id} {route} timing components disagree")
            reference_seconds = reference_payload.get("reference_elapsed_seconds")
            rows.append({
                "case": case_id,
                "route": route,
                "route_status": route_payload.get("status"),
                "candidate_available": bool(
                    reference_payload.get("candidate_available")
                ),
                "comparison_valid": bool(reference_payload.get("comparison_valid")),
                "primary_optimization_seconds": primary_seconds,
                "certification_seconds": (
                    certification_seconds if route == "surrogate" else None
                ),
                "recovery_seconds": recovery_seconds if recovery_seconds else None,
                "complete_optimization_seconds": complete_seconds,
                "exact_reference_seconds": (
                    None if reference_seconds is None else float(reference_seconds)
                ),
            })

    input_digests = {
        path.relative_to(run).as_posix(): file_digest(path)
        for path in sorted(set(input_paths))
    }
    contract = _canonical_json_digest({
        "protocol": TIMING_PROTOCOL,
        "source_digest": source_id,
        "analysis_input_digest": analysis_id,
        "cases": cases,
        "inputs": input_digests,
    })
    ledger_path = run / "metrics" / "robustness_case_timing.csv"
    events_path = run / "metrics" / "timing_events.csv"
    summary_path = run / "metrics" / "robustness_case_timing_summary.json"
    marker_path = run / "metrics" / "robustness_case_timing_complete.json"
    if marker_path.is_file():
        marker = _load_json_object(marker_path, description="robustness timing marker")
        if (
            marker.get("timing_contract") == contract
            and _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return pd.read_csv(ledger_path)

    ledger = pd.DataFrame(rows)
    event_rows: list[dict[str, Any]] = []
    for row in rows:
        route = str(row["route"])
        values = {
            f"{route}_primary_optimization": row["primary_optimization_seconds"],
            f"{route}_complete_optimization": row["complete_optimization_seconds"],
            f"{route}_exact_reference": row["exact_reference_seconds"],
        }
        if route == "surrogate":
            values["surrogate_local_certification"] = row["certification_seconds"]
        if row["recovery_seconds"] is not None:
            values["direct_failure_recovery"] = row["recovery_seconds"]
        for category, value in values.items():
            if value is not None and np.isfinite(float(value)):
                event_rows.append({
                    "case": row["case"], "route": route,
                    "category": category, "elapsed_seconds": float(value),
                    "unit": "seconds_per_robustness_case",
                })
    events = pd.DataFrame(event_rows)
    categories = {
        str(category): _robustness_timing_summary(
            pd.to_numeric(group["elapsed_seconds"], errors="coerce").tolist()
        )
        for category, group in events.groupby("category", sort=True)
    }
    atomic_dataframe(ledger_path, ledger)
    atomic_dataframe(events_path, events)
    atomic_json(summary_path, {
        "timing_contract": contract,
        "protocol": TIMING_PROTOCOL,
        "source": "completed robustness/sensitivity cases only",
        "nominal_case_included": False,
        "robustness_case_count": len(cases),
        "warmup_count": 0,
        "repeated_test_batch_count": 0,
        "categories": categories,
    })
    atomic_json(marker_path, {
        "stage": "robustness_case_timing_aggregation",
        "timing_contract": contract,
        "source_digest": source_id,
        "input_digest": analysis_id,
        "case_count": len(cases),
        "artifacts": _artifact_hashes(
            run, (ledger_path, events_path, summary_path),
        ),
    })
    return ledger


def _route_contract_id(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    influent: np.ndarray,
    route: str,
    protocol: str,
    settings: Any,
    starts: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(case_id.encode())
    digest.update(route.encode())
    digest.update(protocol.encode())
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode())
    digest.update(np.ascontiguousarray(influent, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(starts, dtype="<f8").tobytes())
    return digest.hexdigest()


def _start_controls(result: Any, route: str) -> np.ndarray:
    del route
    return np.asarray(result.initial_normalized_controls, dtype=float)


def _validate_route_result_integrity(result: Any, route: str) -> None:
    """Reject non-finite computational results while permitting clean failures."""

    controls = _start_controls(result, route)
    if controls.shape != (7,) or not np.all(np.isfinite(controls)):
        raise RuntimeError(f"{route} attempt returned invalid initial controls")
    if route == "surrogate":
        final = result.final
        if final is None:
            return
        arrays = (
            final.normalized_controls, final.theta, final.raw, final.projected,
            final.displacement, final.objective_components,
            final.engineering_rows, final.engineering_quantities,
            final.trust_rows, final.trust_values,
        )
        scalars = (final.objective,)
    elif route == "direct":
        if not bool(result.feasible):
            return
        arrays = (
            result.normalized_controls, result.theta, result.state,
            result.response, result.engineering, result.objective_components,
        )
        scalars = (result.objective, result.feed_tss)
    else:
        raise ValueError(f"unknown optimization route {route!r}")
    if any(not np.all(np.isfinite(np.asarray(value, dtype=float))) for value in arrays):
        raise RuntimeError(f"{route} attempt returned a non-finite candidate")
    if not np.all(np.isfinite(np.asarray(scalars, dtype=float))):
        raise RuntimeError(f"{route} attempt returned a non-finite objective/state")


def _read_completed_starts(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    starts: np.ndarray,
) -> tuple[dict[int, Any], float]:
    result_type = SurrogateStartResult if route == "surrogate" else DirectStartResult
    completed: dict[int, Any] = {}
    elapsed = 0.0
    for index in range(len(starts)):
        path = case_directory / "checkpoints" / f"{route}_start_{index:02d}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("route_contract") != route_contract:
                continue
            if (
                payload.get("protocol") != route_protocol
                or int(payload.get("start_index", -1)) != index
                or not np.array_equal(
                    np.asarray(payload.get("normalized_start"), dtype=float), starts[index]
                )
            ):
                raise RuntimeError(f"current {route} attempt checkpoint is inconsistent")
            result = result_type.from_dict(payload["result"])
            if result.start_index != index or not np.array_equal(
                _start_controls(result, route), starts[index]
            ):
                raise RuntimeError(f"current {route} attempt result is inconsistent")
            _validate_route_result_integrity(result, route)
            completed[index] = result
            elapsed = max(elapsed, float(payload.get("cumulative_route_wall_seconds", 0.0)))
        except RuntimeError:
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"current {route} attempt checkpoint is corrupt: {path.name}"
            ) from exc
    return completed, elapsed


def _publish_partial_starts(
    case_directory: Path, route: str, completed: Mapping[int, Any],
) -> None:
    atomic_json(
        case_directory / f"{route}_starts.partial.json",
        [completed[index].as_dict() for index in sorted(completed)],
        nonfinite_to_none=True,
    )


def _write_start_checkpoint(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    starts: np.ndarray,
    result: Any,
    cumulative_elapsed: float,
) -> None:
    index = int(result.start_index)
    if not 0 <= index < len(starts) or not np.array_equal(
        _start_controls(result, route), starts[index]
    ):
        raise RuntimeError(f"{route} solver returned a mismatched start index/control")
    _validate_route_result_integrity(result, route)
    atomic_json(
        case_directory / "checkpoints" / f"{route}_start_{index:02d}.json",
        {
            "schema": 1,
            "route": route,
            "route_contract": route_contract,
            "protocol": route_protocol,
            "start_index": index,
            "normalized_start": starts[index],
            "cumulative_route_wall_seconds": cumulative_elapsed,
            "result": result.as_dict(),
        },
        nonfinite_to_none=True,
    )


def _load_complete_route(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    starts: np.ndarray,
) -> Any | None:
    marker_path = case_directory / f"{route}_complete.json"
    payload_path = case_directory / f"{route}.json"
    if not marker_path.is_file() or not payload_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("route_contract") != route_contract:
            return None
        if marker.get("protocol") != route_protocol:
            raise RuntimeError(f"current {route} completion marker has wrong protocol")
        if not _artifacts_match(case_directory, marker.get("artifacts", {})):
            raise RuntimeError(f"current {route} completed artifacts changed")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if (
            payload.get("route_contract") != route_contract
            or payload.get("protocol") != route_protocol
        ):
            raise RuntimeError(f"current {route} result payload is inconsistent")
        result_type = SurrogateStartResult if route == "surrogate" else DirectStartResult
        restored = tuple(result_type.from_dict(item) for item in payload["starts"])
        if (
            len(restored) != len(starts)
            or [item.start_index for item in restored] != list(range(len(starts)))
        ):
            raise RuntimeError(f"current {route} result has wrong attempt count")
        if any(not np.array_equal(_start_controls(item, route), starts[index])
               for index, item in enumerate(restored)):
            raise RuntimeError(f"current {route} result has wrong initial controls")
        for item in restored:
            _validate_route_result_integrity(item, route)
        selected_index = payload.get("selected_start")
        selected = None if selected_index is None else restored[int(selected_index)]
        result_class = SurrogateMultistartResult if route == "surrogate" else DirectMultistartResult
        if route == "surrogate":
            result = result_class(
                restored, selected, str(payload["status"]), protocol=route_protocol,
            )
        else:
            result = result_class(restored, selected, str(payload["status"]))
        return result, payload
    except RuntimeError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"current {route} completion checkpoint is corrupt") from exc


def _load_retained_complete_route(
    case_directory: Path,
    *,
    route: str,
    route_protocol: str,
    starts: np.ndarray,
) -> Any | None:
    """Load a schema-5 route only when the latest migration pins its case."""

    run = case_directory.parent.parent
    contract_path = run / "inputs" / "contract.json"
    if not contract_path.is_file():
        return None
    try:
        contract = _load_json_object(contract_path, description="run contract")
        if int(contract.get("runner_schema", -1)) != RUNNER_SCHEMA:
            return None
        history = contract.get("contract_migrations")
        if not isinstance(history, list) or not history:
            return None
        _validate_migration_history(run, contract)
        stage: Mapping[str, Any] | None = None
        for entry in reversed(history):
            record = _load_json_object(
                run / str(entry.get("record", "")),
                description="optimization-retention migration record",
            )
            retained_reference = record.get("retained_stage_manifest")
            if not isinstance(retained_reference, Mapping):
                continue
            retained_path = run / str(retained_reference.get("path", ""))
            if (
                not retained_path.is_file()
                or file_digest(retained_path) != retained_reference.get("sha256")
            ):
                raise RuntimeError("optimization retained-stage manifest changed")
            retained = _load_json_object(
                retained_path, description="optimization retained-stage manifest",
            )
            candidate_stage = retained.get("stages", {}).get(
                f"optimization/{case_directory.name}"
            )
            if isinstance(candidate_stage, Mapping):
                stage = candidate_stage
                break
        if stage is None:
            return None
        marker_path = run / str(stage.get("checkpoint", ""))
        if (
            not marker_path.is_file()
            or file_digest(marker_path) != stage.get("checkpoint_sha256")
            or not _artifacts_match(run, stage.get("artifacts", {}))
        ):
            raise RuntimeError("retained optimization case changed after migration")
        payload = _load_json_object(
            case_directory / f"{route}.json", description=f"retained {route} route",
        )
        retained_contract = str(payload.get("route_contract", ""))
        if not retained_contract:
            raise RuntimeError("retained route omits its route contract")
        return _load_complete_route(
            case_directory,
            route=route,
            route_contract=retained_contract,
            route_protocol=route_protocol,
            starts=starts,
        )
    except RuntimeError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"retained {route} completion checkpoint is corrupt") from exc


def _publish_complete_route(
    case_directory: Path,
    *,
    route: str,
    route_contract: str,
    route_protocol: str,
    result: Any,
    elapsed_seconds: float,
) -> dict[str, Any]:
    start_count = len(result.starts)
    if start_count != 1:
        raise RuntimeError(f"{route} protocol requires exactly one local attempt")
    for item in result.starts:
        _validate_route_result_integrity(item, route)
    payload = result.as_dict()
    payload.update({
        "route": route,
        "route_contract": route_contract,
        "protocol": route_protocol,
        "optimization_attempt_count": start_count,
        "article_start_count": start_count,
        "maximum_wall_time": None,
        "elapsed_seconds": elapsed_seconds,
    })
    payload_path = case_directory / f"{route}.json"
    atomic_json(payload_path, payload, nonfinite_to_none=True)
    start_paths = tuple(
        case_directory / "checkpoints" / f"{route}_start_{index:02d}.json"
        for index in range(start_count)
    )
    if not all(path.is_file() for path in start_paths):
        raise RuntimeError(f"{route} is missing its atomic attempt checkpoint")
    paths = (payload_path, *start_paths)
    atomic_json(case_directory / f"{route}_complete.json", {
        "stage": f"{route}_single_local_attempt",
        "route_contract": route_contract,
        "protocol": route_protocol,
        "start_count": start_count,
        "optimization_attempt_count": start_count,
        "selected_start": payload["selected_start"],
        "status": payload["status"],
        "artifacts": _artifact_hashes(case_directory, paths),
    })
    return payload


def _run_surrogate_route(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    assets: Any,
    source_id: str,
    analysis_id: str,
    problem: Any | None = None,
) -> tuple[SurrogateMultistartResult, dict[str, Any]]:
    settings = SurrogateSolverSettings(maximum_wall_time=None)
    starts = np.asarray(EXACT_QP_CENTER_START, dtype=float).reshape(1, 7)
    contract = _route_contract_id(
        source_id=source_id, analysis_id=analysis_id, case_id=case_id,
        influent=influent, route="surrogate",
        protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        settings=settings, starts=starts,
    )
    restored = _load_complete_route(
        case_directory, route="surrogate", route_contract=contract,
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL, starts=starts,
    )
    if restored is not None:
        return restored
    restored = _load_retained_complete_route(
        case_directory,
        route="surrogate",
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        starts=starts,
    )
    if restored is not None:
        return restored
    completed, prior_elapsed = _read_completed_starts(
        case_directory, route="surrogate", route_contract=contract,
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL, starts=starts,
    )
    _publish_partial_starts(case_directory, "surrogate", completed)
    started = perf_counter()

    def checkpoint(result: SurrogateStartResult) -> None:
        completed[result.start_index] = result
        _write_start_checkpoint(
            case_directory, route="surrogate", route_contract=contract,
            route_protocol=EXACT_QP_SINGLE_START_PROTOCOL,
            starts=starts, result=result,
            cumulative_elapsed=prior_elapsed + perf_counter() - started,
        )
        _publish_partial_starts(case_directory, "surrogate", completed)

    result = solve_surrogate_exact_qp_local(
        assets,
        SurrogateCase(influent=influent, case_id=case_id),
        settings=settings,
        problem=problem,
        name="article_surrogate_exact_qp",
        progress_callback=checkpoint,
        completed_result=completed.get(0),
    )
    if (
        len(result.starts) != 1
        or result.protocol != EXACT_QP_SINGLE_START_PROTOCOL
        or result.starts[0].stages
    ):
        raise RuntimeError("surrogate route violated the single exact-QP attempt contract")
    if 0 not in completed:
        # A valid fresh solve always invokes the callback exactly once. This is
        # also a guard against a solver result that was never atomically saved.
        raise RuntimeError("surrogate exact-QP attempt was not checkpointed")
    elapsed = prior_elapsed + perf_counter() - started
    payload = _publish_complete_route(
        case_directory, route="surrogate", route_contract=contract,
        route_protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        result=result, elapsed_seconds=elapsed,
    )
    return result, payload


def _run_direct_route(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    assets: Any,
    development_decisions: np.ndarray,
    development_influents: np.ndarray,
    development_targets: np.ndarray,
    source_id: str,
    analysis_id: str,
) -> tuple[DirectMultistartResult, dict[str, Any]]:
    settings = SolverSettings(maximum_wall_time=None)
    starts = np.asarray(direct_normalized_starts()[0], dtype=float).reshape(1, 7)
    if not np.array_equal(starts[0], np.full(7, 0.5)):
        raise RuntimeError("direct route center start is not deterministic")
    contract = _route_contract_id(
        source_id=source_id, analysis_id=analysis_id, case_id=case_id,
        influent=influent, route="direct", protocol=DIRECT_SINGLE_CENTER_PROTOCOL,
        settings=settings, starts=starts,
    )
    restored = _load_complete_route(
        case_directory, route="direct", route_contract=contract,
        route_protocol=DIRECT_SINGLE_CENTER_PROTOCOL, starts=starts,
    )
    if restored is not None:
        return restored
    restored = _load_retained_complete_route(
        case_directory,
        route="direct",
        route_protocol=DIRECT_SINGLE_CENTER_PROTOCOL,
        starts=starts,
    )
    if restored is not None:
        return restored
    completed, prior_elapsed = _read_completed_starts(
        case_directory, route="direct", route_contract=contract,
        route_protocol=DIRECT_SINGLE_CENTER_PROTOCOL, starts=starts,
    )
    _publish_partial_starts(case_directory, "direct", completed)
    started = perf_counter()

    def checkpoint(result: DirectStartResult) -> None:
        completed[result.start_index] = result
        _write_start_checkpoint(
            case_directory, route="direct", route_contract=contract,
            route_protocol=DIRECT_SINGLE_CENTER_PROTOCOL,
            starts=starts, result=result,
            cumulative_elapsed=prior_elapsed + perf_counter() - started,
        )
        _publish_partial_starts(case_directory, "direct", completed)

    result = solve_direct_multistart(
        assets,
        DirectCase(influent=influent, case_id=case_id),
        development_decisions,
        development_influents,
        development_targets,
        settings=settings,
        starts=starts,
        allow_reduced_starts=True,
        completed_starts=completed,
        progress_callback=checkpoint,
    )
    if len(result.starts) != 1:
        raise RuntimeError("direct route violated the single-attempt contract")
    elapsed = prior_elapsed + perf_counter() - started
    payload = _publish_complete_route(
        case_directory, route="direct", route_contract=contract,
        route_protocol=DIRECT_SINGLE_CENTER_PROTOCOL,
        result=result, elapsed_seconds=elapsed,
    )
    return result, payload


def _unavailable_violation(method: str, case: str, reason: str) -> dict[str, Any]:
    return {
        "case": case,
        "method": method,
        "audit_available": False,
        "audit_unavailable_reason": reason,
        "mass_conservation_violation_max": np.nan,
        "mass_conservation_violation_count": 0,
        "nonnegativity_violation_max": np.nan,
        "nonnegativity_violation_count": 0,
    }


def _physical_record(
    method: str,
    case: str,
    response: np.ndarray,
    theta: np.ndarray,
    influent: np.ndarray,
    analysis: AnalysisBundle,
) -> dict[str, Any]:
    if response.shape != (analysis.surrogate_assets.layout.state_size,) or not np.all(
        np.isfinite(response)
    ):
        return _unavailable_violation(method, case, "response unavailable or non-finite")
    overflow_tss_closure = None
    if analysis.overflow_closure is not None:
        overflow_tss_closure = float(
            analysis.overflow_closure.predict(theta, influent)
        )
    record = violation_record(
        method,
        case,
        response,
        theta,
        influent,
        analysis.surrogate_assets.layout,
        analysis.surrogate_assets.row_scales.equality,
        analysis.surrogate_assets.row_scales.inequality,
        analysis.model.response_scale,
        overflow_tss_closure=overflow_tss_closure,
    )
    record["audit_available"] = True
    record["audit_unavailable_reason"] = None
    return record


def _equivalence_error_payload(exc: Exception, elapsed: float) -> dict[str, Any]:
    return {
        "smooth_accepted": False,
        "reference_accepted": False,
        "accepted": False,
        "state_rms": None,
        "state_inf": None,
        "own_smooth_residual": None,
        "own_reference_residual": None,
        "cross_residual": None,
        "relative_objective_difference": None,
        "engineering_difference": None,
        "reference_root_difference_generation": None,
        "reference_root_difference_state_scale": None,
        "branch_agreement": False,
        "feasibility_agreement": False,
        "elapsed_seconds": elapsed,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _strict_equivalence(
    theta: np.ndarray,
    influent: np.ndarray,
    analysis: AnalysisBundle,
) -> tuple[Any | None, dict[str, Any]]:
    return _strict_equivalence_assets(
        theta, influent, analysis.direct_assets,
    )


def _strict_equivalence_assets(
    theta: np.ndarray,
    influent: np.ndarray,
    direct_assets: Any,
) -> tuple[Any | None, dict[str, Any]]:
    settings = SolverSettings(maximum_wall_time=None)
    started = perf_counter()
    try:
        smooth = solve_fixed_input_two_start(
            theta, influent, direct_assets, settings=settings,
        )
        diagnostics = compare_smooth_reference(
            theta, influent, direct_assets,
            smooth=smooth, settings=settings,
        )
        payload = asdict(diagnostics)
        payload["elapsed_seconds"] = perf_counter() - started
        payload["maximum_wall_time"] = None
        return smooth, payload
    except Exception as exc:
        return None, _equivalence_error_payload(exc, perf_counter() - started)


_EQUIVALENCE_WORKER_ASSETS: Any | None = None


def _initialize_equivalence_worker(direct_assets: Any) -> None:
    global _EQUIVALENCE_WORKER_ASSETS
    _EQUIVALENCE_WORKER_ASSETS = direct_assets


def _test_equivalence_worker(
    payload: tuple[int, np.ndarray, np.ndarray],
) -> tuple[int, np.ndarray, dict[str, Any]]:
    row, theta, influent = payload
    if _EQUIVALENCE_WORKER_ASSETS is None:
        raise RuntimeError("fixed-input equivalence worker was not initialized")
    with threadpool_limits(limits=1):
        smooth, equivalence = _strict_equivalence_assets(
            theta, influent, _EQUIVALENCE_WORKER_ASSETS,
        )
    response_count = int(_EQUIVALENCE_WORKER_ASSETS.response_count)
    smooth_response = (
        np.asarray(smooth.routes[0].response, dtype=float)
        if smooth is not None and smooth.accepted
        else np.full(response_count, np.nan)
    )
    return row, smooth_response, equivalence


def _reference_two_start(
    theta: np.ndarray,
    influent: np.ndarray,
    analysis: AnalysisBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Legacy unconditional two-start replay retained for old artifact readers."""

    clarifier = analysis.direct_assets.clarifier
    state_size = analysis.direct_assets.state_count
    response_size = analysis.direct_assets.response_count
    started = perf_counter()
    try:
        operating = ArticleOperatingPoint(*map(float, theta))
        first = solve_steady_state(
            operating, influent, starts=(1,), clarifier=clarifier,
            logarithmic_only=True, strict_v3=True,
        )
        second = solve_steady_state(
            operating, influent, starts=(2,), clarifier=clarifier,
            logarithmic_only=True, strict_v3=True,
        )
        reactors, _ = unpack_state(first.state, clarifier)
        scale = generation_scale(influent, reactors[-1], clarifier)
        difference = float(np.max(np.abs(first.state - second.state) / scale))
        branches_match = branch_classification(
            first.state, clarifier,
        ) == branch_classification(second.state, clarifier)
        accepted = bool(
            first.accepted and second.accepted
            and difference <= 1.0e-6 and branches_match
        )
        response = assemble_target(first.state, operating, influent, clarifier)
        return (
            np.asarray(response, dtype=float),
            np.asarray(first.state, dtype=float),
            np.asarray(second.state, dtype=float),
            {
                "accepted": accepted,
                "scaled_root_difference": difference,
                "branch_agreement": branches_match,
                "elapsed_seconds": perf_counter() - started,
                "start_1_accepted": bool(first.accepted),
                "start_2_accepted": bool(second.accepted),
            },
        )
    except Exception as exc:
        return (
            np.full(response_size, np.nan),
            np.full(state_size, np.nan),
            np.full(state_size, np.nan),
            {
                "accepted": False,
                "scaled_root_difference": None,
                "branch_agreement": False,
                "elapsed_seconds": perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


def _casewise_exact_reference(
    theta: np.ndarray,
    influent: np.ndarray,
    analysis: AnalysisBundle,
    *,
    retained_reference_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate a decision with the exact nonsmooth reference model."""

    controls = np.asarray(theta, dtype=np.float64)
    feed = np.asarray(influent, dtype=np.float64)
    clarifier = analysis.direct_assets.clarifier
    state_size = analysis.direct_assets.state_count
    response_size = analysis.direct_assets.response_count
    started = perf_counter()
    first_state: np.ndarray | None = None
    second_state: np.ndarray | None = None
    first_accepted = False
    second_accepted: bool | None = None
    source = "adaptive_exact_solve"
    errors: list[str] = []
    retained_original_elapsed_seconds: float | None = None

    if retained_reference_path is not None and retained_reference_path.is_file():
        try:
            with np.load(retained_reference_path, allow_pickle=False) as stored:
                retained_theta = np.asarray(stored["theta"], dtype=np.float64)
                if np.array_equal(retained_theta, controls):
                    first_state = np.asarray(stored["state"], dtype=np.float64)
                    second_state = np.asarray(stored["state_start_2"], dtype=np.float64)
                    if (
                        first_state.shape == (state_size,)
                        and second_state.shape == (state_size,)
                        and np.all(np.isfinite(first_state))
                        and np.all(np.isfinite(second_state))
                    ):
                        source = "retained_two_start_exact_replay"
                        equivalence_path = retained_reference_path.with_name(
                            retained_reference_path.name.replace(
                                "_reference.npz", "_equivalence.json",
                            )
                        )
                        if equivalence_path.is_file():
                            equivalence = _load_json_object(
                                equivalence_path,
                                description="retained exact-reference timing",
                            )
                            replay = equivalence.get("reference_replay")
                            if isinstance(replay, Mapping):
                                retained_elapsed = replay.get("elapsed_seconds")
                                if retained_elapsed is not None:
                                    retained_original_elapsed_seconds = float(
                                        retained_elapsed
                                    )
                    else:
                        first_state = second_state = None
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"retained replay: {type(exc).__name__}: {exc}")
            first_state = second_state = None

    operating = ArticleOperatingPoint(*map(float, controls))
    if first_state is None:
        try:
            first = solve_steady_state(
                operating, feed, starts=(1,), clarifier=clarifier,
                logarithmic_only=True, strict_v3=True,
            )
            first_accepted = bool(first.accepted)
            if first.accepted:
                first_state = np.asarray(first.state, dtype=np.float64)
            else:
                errors.append(f"start 1 rejected: {first.message}")
        except Exception as exc:
            errors.append(f"start 1: {type(exc).__name__}: {exc}")
    else:
        first_accepted = True

    first_branch = (
        classify_branches(first_state, analysis.direct_assets)
        if first_state is not None else None
    )
    # Both deterministic exact starts are required for every selected decision.
    # Branch ambiguity is a reported qualifier, but reproducibility across the
    # two nonsmooth solves remains part of the common-reference validity check.
    second_required = True
    if second_state is not None:
        second_accepted = True
    elif second_required:
        try:
            second = solve_steady_state(
                operating, feed, starts=(2,), clarifier=clarifier,
                logarithmic_only=True, strict_v3=True,
            )
            second_accepted = bool(second.accepted)
            if second.accepted:
                second_state = np.asarray(second.state, dtype=np.float64)
            else:
                errors.append(f"start 2 rejected: {second.message}")
        except Exception as exc:
            second_accepted = False
            errors.append(f"start 2: {type(exc).__name__}: {exc}")

    selected_state = first_state if first_state is not None else second_state
    if selected_state is None:
        return (
            np.full(response_size, np.nan), np.full(state_size, np.nan),
            np.full(state_size, np.nan), {
                "accepted": False, "status": "solver_failed", "source": source,
                "start_1_accepted": first_accepted,
                "start_2_required": second_required,
                "start_2_accepted": second_accepted,
                "elapsed_seconds": perf_counter() - started, "errors": errors,
            },
        )

    selected_diagnostics = mechanistic_diagnostics(
        selected_state, operating, feed, clarifier=clarifier, strict_v3=True,
    )
    physical_passed = bool(selected_diagnostics.get("passed", False))
    selected_branch = classify_branches(selected_state, analysis.direct_assets)
    branch_agreement: bool | None = None
    generation_difference: float | None = None
    state_scale_difference: float | None = None
    first_diagnostics: dict[str, Any] | None = None
    second_diagnostics: dict[str, Any] | None = None
    second_branch = None
    if first_state is not None:
        first_diagnostics = mechanistic_diagnostics(
            first_state, operating, feed, clarifier=clarifier, strict_v3=True,
        )
    if second_state is not None:
        second_diagnostics = mechanistic_diagnostics(
            second_state, operating, feed, clarifier=clarifier, strict_v3=True,
        )
        comparison_state = first_state if first_state is not None else selected_state
        reactors, _ = unpack_state(comparison_state, clarifier)
        scale = generation_scale(feed, reactors[-1], clarifier)
        generation_difference = float(np.max(np.abs(comparison_state - second_state) / scale))
        state_scale_difference = float(np.max(
            np.abs(comparison_state - second_state) / analysis.direct_assets.state_scale
        ))
        first_comparison_branch = classify_branches(
            comparison_state, analysis.direct_assets,
        )
        second_branch = classify_branches(second_state, analysis.direct_assets)
        branch_agreement = smooth_branches_match(first_comparison_branch, second_branch)
        physical_passed = bool(physical_passed and second_diagnostics.get("passed", False))

    replay_agreement = bool(
        (
            first_accepted
            and second_accepted is True
            and second_state is not None
            and generation_difference is not None and generation_difference <= 1.0e-6
            and state_scale_difference is not None and state_scale_difference <= 1.0e-6
            and (
                branch_agreement is True
                or selected_branch.ambiguous
                or (second_branch is not None and second_branch.ambiguous)
            )
        )
    )
    accepted = bool(first_accepted and physical_passed and replay_agreement)
    response = assemble_target(selected_state, operating, feed, clarifier)
    reference_components = objective_components(controls, response, analysis.direct_assets)
    reference_objective = float(DEFAULT_OBJECTIVE_WEIGHTS @ reference_components)
    quantities = engineering_quantities(controls, response, analysis.direct_assets)
    reference_engineering_feasible = engineering_feasible(
        controls, response, analysis.direct_assets,
    )
    boundary_ambiguous = bool(
        selected_branch.ambiguous
        or (second_branch is not None and second_branch.ambiguous)
    )
    status = (
        "valid_branch_boundary" if accepted and boundary_ambiguous
        else "valid_interior" if accepted
        else "start_1_failed" if not first_accepted
        else "start_2_failed" if second_accepted is not True
        else "root_disagreement" if second_state is not None and not replay_agreement
        else "physical_audit_failed"
    )
    current_execution_elapsed_seconds = perf_counter() - started
    payload = {
        "accepted": accepted, "status": status, "source": source,
        "start_1_accepted": first_accepted, "start_2_required": second_required,
        "start_2_accepted": second_accepted,
        "two_start_agreement_checked": second_state is not None,
        "scaled_root_difference_generation": generation_difference,
        "scaled_root_difference_state": state_scale_difference,
        "branch_agreement": branch_agreement,
        "branch_disagreement_excused_by_boundary_ambiguity": bool(
            branch_agreement is False
            and (
                selected_branch.ambiguous
                or (second_branch is not None and second_branch.ambiguous)
            )
        ),
        "branch_ambiguous": boundary_ambiguous,
        "minimum_normalized_branch_margin": float(selected_branch.minimum_normalized_margin),
        "branch_start_1": None if first_branch is None else asdict(first_branch),
        "branch_start_2": None if second_branch is None else asdict(second_branch),
        "physical_stability_accepted": physical_passed,
        "diagnostics_start_1": first_diagnostics,
        "diagnostics_start_2": second_diagnostics,
        "engineering_feasible": bool(reference_engineering_feasible),
        "engineering_quantities": quantities,
        "objective": reference_objective,
        "objective_components": reference_components.tolist(),
        "elapsed_seconds": (
            retained_original_elapsed_seconds
            if retained_original_elapsed_seconds is not None
            else current_execution_elapsed_seconds
        ),
        "current_run_reuse_overhead_seconds": current_execution_elapsed_seconds,
        "retained_original_solve_elapsed_seconds": retained_original_elapsed_seconds,
        "errors": errors,
    }
    return (
        np.asarray(response, dtype=np.float64),
        (
            np.full(state_size, np.nan)
            if first_state is None else np.asarray(first_state, dtype=np.float64)
        ),
        np.full(state_size, np.nan) if second_state is None else np.asarray(second_state, dtype=np.float64),
        payload,
    )


def _test_equivalence_contract(
    source_id: str,
    analysis_id: str,
    row: int,
    theta: np.ndarray,
    influent: np.ndarray,
    reference: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(str(row).encode())
    for value in (theta, influent, reference):
        digest.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    return digest.hexdigest()


def _run_untouched_test_equivalence(
    run: Path,
    design: Mapping[str, object],
    test_targets: np.ndarray,
    analysis: AnalysisBundle,
    *,
    source_files: Mapping[str, str],
    analysis_id: str,
    parallel_workers: int,
) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    source_id = source_digest(source_files)
    marker_path = run / "metrics" / "smooth_reference_test_complete.json"
    diagnostics_path = run / "metrics" / "smooth_reference_equivalence_test.csv"
    violations_path = run / "metrics" / "physical_violations_equivalence_test.csv"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("source_digest") == source_id
                and marker.get("input_digest") == analysis_id
                and int(marker.get("row_count", -1)) == len(test_targets)
                and _artifacts_match(run, marker.get("artifacts", {}))
            ):
                diagnostics = pd.read_csv(diagnostics_path)
                violations = pd.read_csv(violations_path)
                if len(diagnostics) == len(test_targets):
                    return bool(marker["all_accepted"]), diagnostics, violations
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    decisions = np.asarray(design["test_decisions"], dtype=float)
    influents = np.asarray(design["test_influents"], dtype=float)
    rows_directory = run / "validation" / "untouched_test_equivalence" / "rows"
    payloads: dict[int, dict[str, Any]] = {}
    contracts: dict[int, str] = {}
    missing: list[int] = []
    for row in range(len(test_targets)):
        contract = _test_equivalence_contract(
            source_id, analysis_id, row, decisions[row], influents[row], test_targets[row],
        )
        contracts[row] = contract
        path = rows_directory / f"row_{row:06d}.json"
        if path.is_file():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if (
                    candidate.get("row_contract") == contract
                    and int(candidate["row"]) == row
                    and isinstance(candidate.get("equivalence"), Mapping)
                    and len(candidate.get("violations", [])) == 2
                ):
                    payloads[row] = candidate
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
        if row not in payloads:
            missing.append(row)

    def publish(row: int, smooth_response: np.ndarray, equivalence: Mapping[str, Any]) -> None:
        violations = [
            _physical_record(
                "smooth", f"test_{row:04d}:fixed_input", smooth_response,
                decisions[row], influents[row], analysis,
            ),
            _physical_record(
                "reference", f"test_{row:04d}:fixed_input", test_targets[row],
                decisions[row], influents[row], analysis,
            ),
        ]
        payload = {
                "row": row,
                "row_contract": contracts[row],
                "equivalence": dict(equivalence),
                "smooth_response": smooth_response,
                "reference_response": test_targets[row],
                "violations": violations,
        }
        atomic_json(
            rows_directory / f"row_{row:06d}.json",
            payload,
            nonfinite_to_none=True,
        )
        payloads[row] = _json_ready(payload, nonfinite_to_none=True)

    if missing:
        worker_count = max(1, min(int(parallel_workers), len(missing)))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_equivalence_worker,
            initargs=(analysis.direct_assets,),
        ) as pool:
            futures = {
                pool.submit(
                    _test_equivalence_worker,
                    (row, decisions[row], influents[row]),
                ): row
                for row in missing
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                row, smooth_response, equivalence = future.result()
                publish(row, smooth_response, equivalence)
                if completed_count % max(1, len(missing) // 100) == 0:
                    _write_state(
                        run, "untouched_test_equivalence", "running",
                        completed_rows=len(payloads), total_rows=len(test_targets),
                        parallel_workers=worker_count,
                    )
    diagnostic_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    for row in range(len(test_targets)):
        payload = payloads[row]
        equivalence_row = dict(payload["equivalence"])
        equivalence_row["row"] = row
        diagnostic_rows.append(equivalence_row)
        physical_rows.extend(payload["violations"])
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values("row").reset_index(drop=True)
    violations = pd.DataFrame(physical_rows)
    atomic_dataframe(diagnostics_path, diagnostics)
    atomic_dataframe(violations_path, violations)
    accepted = diagnostics["accepted"].astype(str).str.lower().eq("true")
    all_accepted = bool(len(diagnostics) == len(test_targets) and accepted.all())
    summary_path = run / "metrics" / "smooth_reference_equivalence_test_summary.json"
    atomic_json(summary_path, {
        "row_count": len(diagnostics),
        "accepted_count": int(accepted.sum()),
        "all_accepted": all_accepted,
        "required_row_count": len(test_targets),
        "actual_selected_decision_denominator_reported_separately": True,
        "parallel_workers": max(1, min(int(parallel_workers), len(test_targets))),
    })
    assert_source_unchanged(source_files)
    paths = (diagnostics_path, violations_path, summary_path)
    atomic_json(marker_path, {
        "stage": "untouched_test_smooth_reference_equivalence",
        "source_digest": source_id,
        "input_digest": analysis_id,
        "row_count": len(test_targets),
        "all_accepted": all_accepted,
        "artifacts": _artifact_hashes(run, paths),
    })
    return all_accepted, diagnostics, violations


def _selection_contract_id(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    route: str,
    route_payload: Mapping[str, Any],
    influent: np.ndarray,
    route_artifact_digest: str,
) -> str:
    digest = sha256()
    for value in (
        source_id, analysis_id, case_id, route,
        str(route_payload.get("route_contract", "")),
        str(route_payload.get("selected_start", "none")),
        route_artifact_digest,
    ):
        digest.update(value.encode())
    digest.update(np.ascontiguousarray(influent, dtype="<f8").tobytes())
    return digest.hexdigest()


def _load_selection_checkpoint(
    case_directory: Path,
    *,
    route: str,
    selection_contract: str,
) -> tuple[bool, bool, pd.DataFrame] | None:
    marker_path = case_directory / f"{route}_selection_complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("selection_contract") != selection_contract
            or not _artifacts_match(case_directory, marker.get("artifacts", {}))
        ):
            return None
        selected = bool(marker["selected"])
        accepted = bool(marker["accepted"])
        if not selected:
            return False, accepted, pd.DataFrame()
        violations = pd.read_csv(case_directory / f"{route}_physical_violations.csv")
        if set(violations["method"]) != {"raw", "projected", "smooth", "reference"}:
            return None
        return True, accepted, violations
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _combined_constraint_function(problem: Any, *, name: str) -> ca.Function:
    """Return equality and inequality rows in IPOPT's original order."""

    variable_count = int(problem.objective_function.numel_in(0))
    parameter_count = int(problem.objective_function.numel_in(1))
    variable = ca.MX.sym(f"{name}_x", variable_count)
    parameter = ca.MX.sym(f"{name}_p", parameter_count)
    return ca.Function(
        f"{name}_constraints",
        [variable, parameter],
        [ca.vertcat(
            problem.equality_function(variable, parameter),
            problem.inequality_function(variable, parameter),
        )],
    )


def _derivative_audit_contract(
    selection_contract: str,
    point: np.ndarray,
    multipliers: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(selection_contract.encode())
    digest.update(np.ascontiguousarray(point, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(multipliers, dtype="<f8").tobytes())
    return digest.hexdigest()


def _run_selected_derivative_audit(
    case_directory: Path,
    *,
    route: str,
    case_id: str,
    influent: np.ndarray,
    selected: Any,
    analysis: AnalysisBundle,
    selection_contract: str,
) -> dict[str, Any]:
    """Persist the route-appropriate first-order derivative audit."""

    path = case_directory / f"{route}_derivative_audit.json"
    marker_path = case_directory / f"{route}_derivative_audit_complete.json"
    exact_qp_route = bool(
        route == "surrogate"
        and getattr(selected, "protocol", None) == EXACT_QP_SINGLE_START_PROTOCOL
    )
    if exact_qp_route:
        final = selected.final
        if final is None:
            raise RuntimeError("selected exact-QP attempt has no final candidate")
        evidence = {
            "stationarity": final.stationarity.as_dict(),
            "lower_active_set": final.lower_active_set,
            "upper_kkt": final.upper_kkt,
            "projection_reproduction_passed": (
                final.feasibility.projection_reproduction_passed
            ),
        }
        evidence = _json_ready(evidence, nonfinite_to_none=True)
        audit_contract = sha256(
            selection_contract.encode()
            + _canonical_json_digest(evidence).encode()
        ).hexdigest()
        if marker_path.is_file() and path.is_file():
            marker = _load_json_object(
                marker_path, description="exact-QP derivative audit marker",
            )
            payload = _load_json_object(path, description="exact-QP derivative audit")
            if marker.get("audit_contract") != audit_contract:
                # A different selection is stale, not corrupt; overwrite it.
                pass
            elif (
                payload.get("audit_contract") != audit_contract
                or not isinstance(payload.get("passed"), bool)
                or not _artifacts_match(case_directory, marker.get("artifacts", {}))
            ):
                raise RuntimeError("current exact-QP derivative audit is corrupt")
            else:
                return payload
        lower = final.lower_active_set
        upper = final.upper_kkt
        passed = bool(
            final.stationarity.resolved
            and final.stationarity.stationary
            and final.stationarity.lower_qp_kkt_passed
            and isinstance(lower, Mapping)
            and lower.get("stable") is True
            and isinstance(upper, Mapping)
            and upper.get("feasible") is True
            and upper.get("stationary") is True
            and final.feasibility.projection_reproduction_passed is True
        )
        payload = {
            "route": route,
            "case": case_id,
            "selected_start": int(selected.start_index),
            "protocol": EXACT_QP_SINGLE_START_PROTOCOL,
            "audit_contract": audit_contract,
            "required": True,
            "audited_point": "exact_qp_active_set_endpoint",
            "passed": passed,
            "status": "passed" if passed else "stationarity_unresolved",
            "result": evidence,
        }
        atomic_json(path, payload, nonfinite_to_none=True)
        atomic_json(marker_path, {
            "stage": "selected_exact_qp_active_set_audit",
            "audit_contract": audit_contract,
            "route": route,
            "case": case_id,
            "passed": passed,
            "artifacts": _artifact_hashes(case_directory, (path,)),
        })
        return payload

    stages = tuple(selected.stages)
    point = (
        np.asarray(stages[-1].primal, dtype=float)
        if stages else np.empty(0, dtype=float)
    )
    multipliers_value = (
        None if not stages else stages[-1].constraint_multipliers
    )
    multipliers = (
        np.asarray(multipliers_value, dtype=float)
        if multipliers_value is not None else np.empty(0, dtype=float)
    )
    audit_contract = _derivative_audit_contract(
        selection_contract, point, multipliers,
    )
    if marker_path.is_file() and path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                marker.get("audit_contract") == audit_contract
                and payload.get("audit_contract") == audit_contract
                and isinstance(payload.get("passed"), bool)
                and _artifacts_match(case_directory, marker.get("artifacts", {}))
            ):
                return payload
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    payload: dict[str, Any] = {
        "route": route,
        "case": case_id,
        "selected_start": int(selected.start_index),
        "audit_contract": audit_contract,
        "required": True,
        "audited_point": "final_continuation_stage_primal",
        "passed": False,
        "status": "audit_exception",
    }
    try:
        if not stages:
            raise RuntimeError("selected start has no continuation stages")
        if multipliers_value is None:
            raise RuntimeError(
                "final continuation stage does not retain IPOPT constraint multipliers"
            )
        if route == "surrogate":
            expected_stage = float(GAP_CONTINUATION[-1])
            if not np.isclose(stages[-1].tau, expected_stage, rtol=0.0, atol=0.0):
                raise RuntimeError("selected surrogate start lacks the final gap stage")
            settings = SurrogateSolverSettings(maximum_wall_time=None)
            case = SurrogateCase(influent=influent, case_id=case_id)
            problem = build_surrogate_nlp(
                analysis.surrogate_assets,
                expected_stage,
                settings=settings,
                name=f"audit_surrogate_{case_id}",
                compile_solver=False,
            )
            parameters = case.parameter_vector(analysis.surrogate_assets)
        elif route == "direct":
            epsilon, receiver_half_width = CONTINUATION_SCHEDULE[-1]
            if (
                not np.isclose(stages[-1].epsilon, epsilon, rtol=0.0, atol=0.0)
                or not np.isclose(
                    stages[-1].receiver_half_width,
                    receiver_half_width,
                    rtol=0.0,
                    atol=0.0,
                )
            ):
                raise RuntimeError("selected direct start lacks the final smoothing stage")
            settings = SolverSettings(maximum_wall_time=None)
            case = DirectCase(influent=influent, case_id=case_id)
            problem = build_direct_nlp(
                analysis.direct_assets,
                epsilon=epsilon,
                receiver_half_width=receiver_half_width,
                settings=settings,
                name=f"audit_direct_{case_id}",
                compile_solver=False,
            )
            parameters = case.parameter_vector()
        else:
            raise ValueError(f"unknown optimization route {route!r}")
        constraints = _combined_constraint_function(
            problem, name=f"audit_{route}_{case_id}",
        )
        result = audit_casadi_nlp_derivatives(
            problem.objective_function,
            constraints,
            point,
            parameters,
            problem.lower_bounds,
            problem.upper_bounds,
            multipliers,
            name=f"article_{route}_{case_id}",
            raise_on_failure=False,
        )
        payload.update({
            "passed": bool(result.passed),
            "status": "passed" if result.passed else "failed_tolerance",
            "result": result.as_dict(),
        })
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(path, payload, nonfinite_to_none=True)
    atomic_json(marker_path, {
        "stage": "selected_continuation_derivative_audit",
        "audit_contract": audit_contract,
        "route": route,
        "case": case_id,
        "passed": bool(payload["passed"]),
        "artifacts": _artifact_hashes(case_directory, (path,)),
    })
    return payload


def _surrogate_certification_contract(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    influent: np.ndarray,
    route_artifact_digest: str,
    settings: SurrogateCertificationSettings,
) -> str:
    digest = sha256()
    digest.update(COMPARISON_PROTOCOL.encode())
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(case_id.encode())
    digest.update(route_artifact_digest.encode())
    digest.update(np.ascontiguousarray(influent, dtype="<f8").tobytes())
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode())
    return digest.hexdigest()


def _run_surrogate_certification(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    result: SurrogateMultistartResult,
    analysis: AnalysisBundle,
    source_id: str,
    analysis_id: str,
    problem: Any,
) -> tuple[FinalCandidateRecord | None, dict[str, Any]]:
    """Certify/polish one retained surrogate endpoint without rerunning search."""

    settings = SurrogateCertificationSettings()
    route_path = case_directory / "surrogate.json"
    contract = _surrogate_certification_contract(
        source_id=source_id,
        analysis_id=analysis_id,
        case_id=case_id,
        influent=influent,
        route_artifact_digest=file_digest(route_path),
        settings=settings,
    )
    payload_path = case_directory / "surrogate_local_convergence.json"
    candidate_path = case_directory / "surrogate_certified.npz"
    marker_path = case_directory / "surrogate_local_convergence_complete.json"
    if marker_path.is_file() and payload_path.is_file():
        marker = _load_json_object(marker_path, description="surrogate certification marker")
        payload = _load_json_object(payload_path, description="surrogate certification")
        if (
            marker.get("certification_contract") == contract
            and payload.get("certification_contract") == contract
            and _artifacts_match(case_directory, marker.get("artifacts", {}))
        ):
            candidate_value = payload.get("candidate")
            candidate = (
                None
                if candidate_value is None
                else FinalCandidateRecord.from_dict(candidate_value)
            )
            return candidate, payload

    if result.selected is None or result.selected.final is None:
        candidate_path.unlink(missing_ok=True)
        payload = {
            "stage": "surrogate_local_convergence",
            "case": case_id,
            "certification_contract": contract,
            "selected": False,
            "locally_converged": False,
            "first_order_certified": False,
            "status": "no_surrogate_candidate",
            "settings": asdict(settings),
            "candidate": None,
            "certificate": None,
        }
        atomic_json(payload_path, payload)
        atomic_json(marker_path, {
            "stage": "surrogate_local_convergence",
            "certification_contract": contract,
            "selected": False,
            "artifacts": _artifact_hashes(case_directory, (payload_path, route_path)),
        })
        return None, payload

    certification = certify_surrogate_local_convergence(
        analysis.surrogate_assets,
        SurrogateCase(influent=influent, case_id=case_id),
        result.selected.final,
        settings=settings,
        problem=problem,
        name=f"article_surrogate_certificate_{case_id}",
    )
    candidate = certification.candidate
    certificate = certification.certificate
    payload = {
        "stage": "surrogate_local_convergence",
        "case": case_id,
        "certification_contract": contract,
        "selected": candidate is not None,
        "locally_converged": bool(certificate.locally_converged),
        "first_order_certified": bool(certificate.first_order_certified),
        "status": certificate.classification,
        "settings": asdict(settings),
        "candidate": None if candidate is None else candidate.as_dict(),
        "certificate": certificate.as_dict(),
    }
    atomic_json(payload_path, payload, nonfinite_to_none=True)
    artifacts: list[Path] = [payload_path, route_path]
    if candidate is not None:
        atomic_npz(
            candidate_path,
            normalized_controls=candidate.normalized_controls,
            theta=candidate.theta,
            raw=candidate.raw,
            projected=candidate.projected,
            objective=np.asarray(candidate.objective),
            objective_components=candidate.objective_components,
        )
        artifacts.append(candidate_path)
    atomic_json(marker_path, {
        "stage": "surrogate_local_convergence",
        "certification_contract": contract,
        "selected": candidate is not None,
        "locally_converged": bool(certificate.locally_converged),
        "first_order_certified": bool(certificate.first_order_certified),
        "artifacts": _artifact_hashes(case_directory, tuple(artifacts)),
    })
    return candidate, payload


def _direct_recovery_contract(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    influent: np.ndarray,
    recovery_start: np.ndarray,
    route_artifact_digest: str,
) -> str:
    digest = sha256()
    digest.update(b"smooth-direct-single-failure-recovery-v1\0")
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(case_id.encode())
    digest.update(route_artifact_digest.encode())
    digest.update(np.ascontiguousarray(influent, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(recovery_start, dtype="<f8").tobytes())
    return digest.hexdigest()


def _run_direct_failure_recovery(
    case_directory: Path,
    *,
    case_id: str,
    influent: np.ndarray,
    result: DirectMultistartResult,
    surrogate_candidate: FinalCandidateRecord | None,
    assets: Any,
    development_decisions: np.ndarray,
    development_influents: np.ndarray,
    development_targets: np.ndarray,
    source_id: str,
    analysis_id: str,
) -> tuple[DirectMultistartResult, dict[str, Any]]:
    """Permit one declared recovery start only after a primary direct failure."""

    if result.selected is not None:
        return result, {
            "attempted": False,
            "selected_from": "primary_center_start",
            "status": "not_required",
        }
    if surrogate_candidate is None:
        return result, {
            "attempted": False,
            "selected_from": None,
            "status": "unavailable_without_convergence_certified_surrogate_candidate",
        }
    start = np.asarray(surrogate_candidate.normalized_controls, dtype=np.float64)
    route_path = case_directory / "direct.json"
    contract = _direct_recovery_contract(
        source_id=source_id,
        analysis_id=analysis_id,
        case_id=case_id,
        influent=influent,
        recovery_start=start,
        route_artifact_digest=file_digest(route_path),
    )
    payload_path = case_directory / "direct_recovery.json"
    marker_path = case_directory / "direct_recovery_complete.json"
    if marker_path.is_file() and payload_path.is_file():
        marker = _load_json_object(marker_path, description="direct recovery marker")
        payload = _load_json_object(payload_path, description="direct recovery")
        if (
            marker.get("recovery_contract") == contract
            and payload.get("recovery_contract") == contract
            and _artifacts_match(case_directory, marker.get("artifacts", {}))
        ):
            stored_result = payload.get("result")
            if stored_result is None:
                return result, payload
            if not isinstance(stored_result, Mapping):
                raise RuntimeError("cached direct recovery result is malformed")
            restored = tuple(
                DirectStartResult.from_dict(item) for item in stored_result["starts"]
            )
            if (
                len(restored) != 1
                or restored[0].start_index != 0
                or not np.array_equal(restored[0].initial_normalized_controls, start)
            ):
                raise RuntimeError("cached direct recovery violates its one-start contract")
            for item in restored:
                _validate_route_result_integrity(item, "direct")
            selected_index = stored_result.get("selected_start")
            if selected_index not in (None, 0):
                raise RuntimeError("cached direct recovery has an invalid selection")
            selected = None if selected_index is None else restored[int(selected_index)]
            return (
                DirectMultistartResult(restored, selected, str(stored_result["status"])),
                payload,
            )
    started = perf_counter()
    try:
        recovered = solve_direct_multistart(
            assets,
            DirectCase(influent=influent, case_id=case_id),
            development_decisions,
            development_influents,
            development_targets,
            settings=SolverSettings(maximum_wall_time=None),
            starts=start.reshape(1, 7),
            allow_reduced_starts=True,
        )
        if (
            len(recovered.starts) != 1
            or recovered.starts[0].start_index != 0
            or not np.array_equal(
                recovered.starts[0].initial_normalized_controls, start,
            )
        ):
            raise RuntimeError("fresh direct recovery violated its one-start contract")
        for item in recovered.starts:
            _validate_route_result_integrity(item, "direct")
        error = None
    except Exception as exc:
        recovered = result
        error = f"{type(exc).__name__}: {exc}"
    payload = {
        "stage": "direct_failure_recovery",
        "protocol": "smooth_direct_single_failure_recovery_v1",
        "recovery_contract": contract,
        "attempted": True,
        "recovery_start": start.tolist(),
        "selected_from": (
            "single_surrogate_endpoint_recovery" if recovered.selected is not None else None
        ),
        "status": recovered.status if error is None else "recovery_execution_failed",
        "elapsed_seconds": perf_counter() - started,
        "error": error,
        "result": None if error is not None else recovered.as_dict(),
    }
    atomic_json(payload_path, payload, nonfinite_to_none=True)
    atomic_json(marker_path, {
        "stage": "direct_failure_recovery",
        "recovery_contract": contract,
        "attempted": True,
        "selected": recovered.selected is not None,
        "artifacts": _artifact_hashes(case_directory, (payload_path, route_path)),
    })
    return recovered, payload


def _casewise_reference_contract(
    *,
    source_id: str,
    analysis_id: str,
    case_id: str,
    route: str,
    theta: np.ndarray | None,
    candidate_source_digest: str,
) -> str:
    digest = sha256()
    digest.update(COMPARISON_PROTOCOL.encode())
    digest.update(source_id.encode())
    digest.update(analysis_id.encode())
    digest.update(case_id.encode())
    digest.update(route.encode())
    digest.update(candidate_source_digest.encode())
    if theta is not None:
        digest.update(np.ascontiguousarray(theta, dtype="<f8").tobytes())
    return digest.hexdigest()


def _scaled_response_errors(
    response: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float | None]:
    if (
        response.shape != reference.shape
        or response.shape != scale.shape
        or not np.all(np.isfinite(response))
        or not np.all(np.isfinite(reference))
    ):
        return {"nrmse": None, "nmae": None, "scaled_inf": None}
    scaled = (response - reference) / scale
    return {
        "nrmse": float(np.sqrt(np.mean(scaled**2))),
        "nmae": float(np.mean(np.abs(scaled))),
        "scaled_inf": float(np.max(np.abs(scaled))),
    }


def _run_casewise_route_reference_evaluation(
    case_directory: Path,
    *,
    case_id: str,
    route: str,
    influent: np.ndarray,
    selected: Any | None,
    surrogate_candidate: FinalCandidateRecord | None,
    route_payload: Mapping[str, Any],
    certification_payload: Mapping[str, Any] | None,
    recovery_payload: Mapping[str, Any] | None,
    analysis: AnalysisBundle,
    source_id: str,
    analysis_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one returned decision with the common exact reference model."""

    if route == "surrogate":
        final = surrogate_candidate
    else:
        final = selected
    theta = None if final is None else np.asarray(final.theta, dtype=np.float64)
    candidate_source = case_directory / (
        "surrogate_local_convergence.json" if route == "surrogate"
        else "direct_recovery.json"
        if recovery_payload is not None and recovery_payload.get("attempted")
        else "direct.json"
    )
    if not candidate_source.is_file():
        candidate_source = case_directory / f"{route}.json"
    contract = _casewise_reference_contract(
        source_id=source_id,
        analysis_id=analysis_id,
        case_id=case_id,
        route=route,
        theta=theta,
        candidate_source_digest=file_digest(candidate_source),
    )
    payload_path = case_directory / f"{route}_casewise_reference.json"
    arrays_path = case_directory / f"{route}_casewise_reference.npz"
    physical_path = case_directory / f"{route}_casewise_physical_violations.csv"
    marker_path = case_directory / f"{route}_casewise_reference_complete.json"
    if marker_path.is_file() and payload_path.is_file():
        marker = _load_json_object(marker_path, description="casewise reference marker")
        payload = _load_json_object(payload_path, description="casewise reference result")
        if (
            marker.get("reference_contract") == contract
            and payload.get("reference_contract") == contract
            and _artifacts_match(case_directory, marker.get("artifacts", {}))
        ):
            physical = (
                pd.read_csv(physical_path)
                if physical_path.is_file() else pd.DataFrame()
            )
            return payload, physical

    if final is None:
        arrays_path.unlink(missing_ok=True)
        physical_path.unlink(missing_ok=True)
        payload = {
            "stage": "casewise_exact_common_reference",
            "reference_contract": contract,
            "case": case_id,
            "route": route,
            "candidate_available": False,
            "native_feasible": False,
            "exact_replay_valid": False,
            "comparison_valid": False,
            "status": f"unpaired_{route}_no_candidate",
            "native_status": route_payload.get("status"),
            "recovery": recovery_payload,
        }
        atomic_json(payload_path, payload, nonfinite_to_none=True)
        atomic_json(marker_path, {
            "stage": "casewise_exact_common_reference",
            "reference_contract": contract,
            "candidate_available": False,
            "artifacts": _artifact_hashes(case_directory, (payload_path, candidate_source)),
        })
        return payload, pd.DataFrame()

    if theta is None or theta.shape != (7,) or not np.all(np.isfinite(theta)):
        raise RuntimeError(f"{route} selected candidate has invalid controls")
    normalized = (theta - DECISION_LOWER) / (DECISION_UPPER - DECISION_LOWER)
    surrogate_case = SurrogateCase(influent=influent, case_id=case_id)
    raw = np.asarray(analysis.model.predict(theta, influent), dtype=np.float64)
    projection = cold_reproject(
        analysis.surrogate_assets,
        surrogate_case,
        normalized,
        raise_on_failure=False,
    )
    projected = np.asarray(projection.state, dtype=np.float64)
    native_full_response = (
        np.asarray(final.projected, dtype=np.float64)
        if route == "surrogate"
        else np.asarray(final.response, dtype=np.float64)
    )
    native_response = (
        native_full_response
        if route == "surrogate"
        else reduce_mechanistic_responses(
            native_full_response, analysis.surrogate_assets.layout.layer_count,
        )
    )
    native_objective = float(final.objective)
    expected_response_shape = (analysis.surrogate_assets.layout.state_size,)
    for label, values in (
        ("raw", raw), ("projected", projected),
        ("optimizer-native", native_response),
    ):
        if values.shape != expected_response_shape or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"{route} {label} response is non-finite or has the wrong shape"
            )
    if not np.isfinite(native_objective):
        raise RuntimeError(f"{route} selected candidate has a non-finite objective")
    native_feasible = bool(
        final.feasibility.feasible
        if route == "surrogate" else final.feasible
    )
    if not native_feasible:
        raise RuntimeError(f"{route} selected candidate failed its native feasibility audit")
    retained_reference = case_directory / f"{route}_reference.npz"
    reference_full, state_1, state_2, reference_payload = _casewise_exact_reference(
        theta,
        influent,
        analysis,
        retained_reference_path=retained_reference,
    )
    reference = (
        reduce_mechanistic_responses(
            reference_full, analysis.surrogate_assets.layout.layer_count,
        )
        if np.all(np.isfinite(reference_full))
        else np.full(expected_response_shape, np.nan)
    )
    exact_replay_valid = bool(
        reference_payload.get("accepted") is True
        and np.all(np.isfinite(reference))
    )
    reference_valid = bool(
        exact_replay_valid
        and reference_payload.get("engineering_feasible") is True
    )
    reference_objective = (
        float(reference_payload["objective"]) if exact_replay_valid else None
    )
    reference_components = (
        list(reference_payload["objective_components"])
        if exact_replay_valid else None
    )
    response_scale = np.asarray(analysis.model.response_scale, dtype=np.float64)
    local_converged = (
        bool(certification_payload.get("locally_converged"))
        if route == "surrogate" and certification_payload is not None
        else bool(getattr(final, "stationary", False))
    )
    first_order_certified = (
        bool(certification_payload.get("first_order_certified"))
        if route == "surrogate" and certification_payload is not None
        else bool(getattr(final, "stationary", False))
    )
    primary_optimization_seconds = float(
        route_payload.get("elapsed_seconds", np.nan)
    )
    recovery_seconds = (
        float(recovery_payload.get("elapsed_seconds", 0.0))
        if recovery_payload is not None and recovery_payload.get("attempted")
        else 0.0
    )
    certification_seconds = (
        float(certification_payload["certificate"].get("elapsed_seconds", 0.0))
        if certification_payload is not None
        and certification_payload.get("certificate") is not None
        else 0.0
    )
    total_optimization_seconds = (
        primary_optimization_seconds + recovery_seconds + certification_seconds
        if np.isfinite(primary_optimization_seconds) else None
    )
    payload = {
        "stage": "casewise_exact_common_reference",
        "reference_contract": contract,
        "case": case_id,
        "route": route,
        "candidate_available": True,
        "native_feasible": native_feasible,
        "exact_replay_valid": exact_replay_valid,
        "comparison_valid": reference_valid,
        "status": (
            str(reference_payload.get("status")) if reference_valid
            else "exact_valid_engineering_infeasible" if exact_replay_valid
            else f"reference_{reference_payload.get('status', 'failed')}"
        ),
        "selected_start": int(selected.start_index) if selected is not None else 0,
        "normalized_controls": normalized.tolist(),
        "theta": theta.tolist(),
        "native_status": getattr(final, "status", route_payload.get("status")),
        "native_objective": native_objective,
        "exact_reference_objective": reference_objective,
        "exact_reference_objective_components": reference_components,
        "native_minus_reference_objective": (
            None if reference_objective is None else native_objective - reference_objective
        ),
        "reference": reference_payload,
        "local_convergence_certified": local_converged,
        "first_order_stationarity_certified": first_order_certified,
        "local_convergence_classification": (
            certification_payload.get("status")
            if route == "surrogate" and certification_payload is not None
            else getattr(final, "status", None)
        ),
        "branch_ambiguity_is_qualifier_not_rejection": True,
        "minimum_srt_is_descriptive_not_eligibility_gate": True,
        "projection_accepted": bool(projection.accepted),
        "prediction_error_raw": _scaled_response_errors(raw, reference, response_scale),
        "prediction_error_projected": _scaled_response_errors(
            projected, reference, response_scale,
        ),
        "native_model_error": _scaled_response_errors(
            native_response, reference, response_scale,
        ),
        "primary_optimization_elapsed_seconds": (
            primary_optimization_seconds
            if np.isfinite(primary_optimization_seconds) else None
        ),
        "recovery_elapsed_seconds": recovery_seconds if recovery_seconds else None,
        "optimization_elapsed_seconds": total_optimization_seconds,
        "certification_elapsed_seconds": (
            certification_seconds if certification_seconds else None
        ),
        "reference_elapsed_seconds": reference_payload.get("elapsed_seconds"),
        "recovery": recovery_payload,
    }
    atomic_json(payload_path, payload, nonfinite_to_none=True)
    atomic_npz(
        arrays_path,
        theta=theta,
        normalized_controls=normalized,
        raw=raw,
        projected=projected,
        optimizer_native=native_response,
        exact_reference=reference,
        optimizer_native_full=(
            native_full_response if route == "direct" else np.empty(0, dtype=np.float64)
        ),
        exact_reference_full=reference_full,
        exact_state_start_1=state_1,
        exact_state_start_2=state_2,
    )
    operating = ArticleOperatingPoint(*map(float, theta))
    unavailable_response = np.full(analysis.direct_assets.response_count, np.nan)
    response_1 = (
        assemble_target(
            state_1, operating, influent, analysis.direct_assets.clarifier,
        )
        if np.all(np.isfinite(state_1)) else unavailable_response.copy()
    )
    response_2 = (
        assemble_target(
            state_2, operating, influent, analysis.direct_assets.clarifier,
        )
        if np.all(np.isfinite(state_2)) else unavailable_response.copy()
    )
    response_rows: list[tuple[str, np.ndarray]] = [
        ("raw", raw),
        ("projected", projected),
        ("optimizer_native", native_response),
        (
            "exact_mechanistic_start_1",
            reduce_mechanistic_responses(
                response_1, analysis.surrogate_assets.layout.layer_count,
            ) if np.all(np.isfinite(response_1)) else np.full(expected_response_shape, np.nan),
        ),
        (
            "exact_mechanistic_start_2",
            reduce_mechanistic_responses(
                response_2, analysis.surrogate_assets.layout.layer_count,
            ) if np.all(np.isfinite(response_2)) else np.full(expected_response_shape, np.nan),
        ),
    ]
    physical = pd.DataFrame([
        {
            **_physical_record(method, case_id, response, theta, influent, analysis),
            "decision_route": route,
            "response_source": method,
        }
        for method, response in response_rows
    ])
    atomic_dataframe(physical_path, physical)
    atomic_json(marker_path, {
        "stage": "casewise_exact_common_reference",
        "reference_contract": contract,
        "candidate_available": True,
        "comparison_valid": reference_valid,
        "artifacts": _artifact_hashes(
            case_directory,
            (payload_path, arrays_path, physical_path, candidate_source),
        ),
    })
    return payload, physical


def _casewise_comparison_row(
    case_id: str,
    surrogate: Mapping[str, Any],
    direct: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not surrogate.get("candidate_available"):
        reasons.append("surrogate_no_candidate")
    if not direct.get("candidate_available"):
        reasons.append("direct_no_candidate")
    if surrogate.get("candidate_available"):
        if not surrogate.get("exact_replay_valid"):
            reasons.append("surrogate_reference_invalid")
        elif not surrogate.get("comparison_valid"):
            reasons.append("surrogate_exact_engineering_infeasible")
    if direct.get("candidate_available"):
        if not direct.get("exact_replay_valid"):
            reasons.append("direct_reference_invalid")
        elif not direct.get("comparison_valid"):
            reasons.append("direct_exact_engineering_infeasible")
    if surrogate.get("candidate_available") and not surrogate.get("native_feasible"):
        reasons.append("surrogate_native_infeasible")
    if direct.get("candidate_available") and not direct.get("native_feasible"):
        reasons.append("direct_native_infeasible")
    eligible = not reasons
    s_objective = surrogate.get("exact_reference_objective")
    d_objective = direct.get("exact_reference_objective")
    delta = (
        float(s_objective) - float(d_objective)
        if eligible and s_objective is not None and d_objective is not None else None
    )
    symmetric = (
        delta / max(1.0, abs(float(s_objective)), abs(float(d_objective)))
        if delta is not None else None
    )
    direct_relative = (
        100.0 * delta / max(abs(float(d_objective)), 1.0e-12)
        if delta is not None else None
    )
    s_controls = surrogate.get("normalized_controls")
    d_controls = direct.get("normalized_controls")
    control_difference = None
    if eligible and isinstance(s_controls, list) and isinstance(d_controls, list):
        difference = np.asarray(s_controls, dtype=float) - np.asarray(d_controls, dtype=float)
        control_difference = {
            "rms": float(np.sqrt(np.mean(difference**2))),
            "maximum": float(np.max(np.abs(difference))),
        }
    component_differences = None
    s_components = surrogate.get("exact_reference_objective_components")
    d_components = direct.get("exact_reference_objective_components")
    if eligible and isinstance(s_components, list) and isinstance(d_components, list):
        component_differences = (
            np.asarray(s_components, dtype=float) - np.asarray(d_components, dtype=float)
        ).tolist()
    surrogate_seconds = surrogate.get("optimization_elapsed_seconds")
    direct_seconds = direct.get("optimization_elapsed_seconds")
    timing_ratio = (
        float(direct_seconds) / float(surrogate_seconds)
        if surrogate_seconds is not None
        and direct_seconds is not None
        and np.isfinite(float(surrogate_seconds))
        and np.isfinite(float(direct_seconds))
        and float(surrogate_seconds) > 0.0
        else None
    )
    row = {
        "case": case_id,
        "comparison_eligible": eligible,
        "minimum_srt_is_descriptive_not_eligibility_gate": True,
        "ineligibility_reasons": ";".join(reasons) if reasons else None,
        "surrogate_candidate_available": bool(surrogate.get("candidate_available")),
        "direct_candidate_available": bool(direct.get("candidate_available")),
        "surrogate_local_convergence_certified": bool(
            surrogate.get("local_convergence_certified")
        ),
        "direct_first_order_stationarity_certified": bool(
            direct.get("first_order_stationarity_certified")
        ),
        "both_local_convergence_qualified": bool(
            eligible
            and surrogate.get("local_convergence_certified")
            and direct.get("first_order_stationarity_certified")
        ),
        "surrogate_reference_status": surrogate.get("status"),
        "direct_reference_status": direct.get("status"),
        "surrogate_branch_ambiguous": (
            surrogate.get("reference", {}).get("branch_ambiguous")
        ),
        "direct_branch_ambiguous": direct.get("reference", {}).get("branch_ambiguous"),
        "J_S_reference": s_objective,
        "J_M_reference": d_objective,
        "delta_J_S_minus_M": delta,
        "symmetric_relative_difference": symmetric,
        "surrogate_penalty_percent_relative_to_direct": direct_relative,
        "control_rms_difference": (
            None if control_difference is None else control_difference["rms"]
        ),
        "control_maximum_difference": (
            None if control_difference is None else control_difference["maximum"]
        ),
        "objective_component_differences": component_differences,
        "surrogate_optimization_seconds": surrogate_seconds,
        "direct_optimization_seconds": direct_seconds,
        "mechanistic_surrogate_time_ratio": timing_ratio,
        "surrogate_certification_seconds": surrogate.get("certification_elapsed_seconds"),
        "surrogate_reference_seconds": surrogate.get("reference_elapsed_seconds"),
        "direct_reference_seconds": direct.get("reference_elapsed_seconds"),
    }
    if component_differences is not None:
        row.update({
            f"delta_component_{name}": value
            for name, value in zip(
                OBJECTIVE_COMPONENT_NAMES, component_differences, strict=True,
            )
        })
    return row


def _evaluate_selected_route(
    case_directory: Path,
    *,
    case_id: str,
    route: str,
    influent: np.ndarray,
    result: Any,
    route_payload: Mapping[str, Any],
    analysis: AnalysisBundle,
    source_id: str,
    analysis_id: str,
) -> tuple[bool, bool, pd.DataFrame]:
    selection_contract = _selection_contract_id(
        source_id=source_id, analysis_id=analysis_id, case_id=case_id,
        route=route, route_payload=route_payload, influent=influent,
        route_artifact_digest=file_digest(case_directory / f"{route}.json"),
    )
    restored = _load_selection_checkpoint(
        case_directory, route=route, selection_contract=selection_contract,
    )
    if restored is not None:
        return restored
    selected = result.selected
    if selected is None:
        route_path = case_directory / f"{route}.json"
        atomic_json(case_directory / f"{route}_selection_complete.json", {
            "stage": "selected_decision_cross_evaluation",
            "selection_contract": selection_contract,
            "selected": False,
            "accepted": False,
            "artifacts": _artifact_hashes(case_directory, (route_path,)),
        })
        return False, False, pd.DataFrame()
    final = selected.final if route == "surrogate" else selected
    if final is None:
        raise RuntimeError(f"{route} selected start has no final candidate")
    _validate_route_result_integrity(selected, route)
    derivative_audit = _run_selected_derivative_audit(
        case_directory,
        route=route,
        case_id=case_id,
        influent=influent,
        selected=selected,
        analysis=analysis,
        selection_contract=selection_contract,
    )
    theta = np.asarray(final.theta, dtype=float)
    if theta.shape != (7,) or not np.all(np.isfinite(theta)):
        raise RuntimeError(f"{route} selected candidate has invalid controls")
    normalized = (theta - DECISION_LOWER) / (DECISION_UPPER - DECISION_LOWER)
    case = SurrogateCase(influent=influent, case_id=case_id)
    raw = np.asarray(analysis.model.predict(theta, influent), dtype=float)
    projection = cold_reproject(
        analysis.surrogate_assets, case, normalized, raise_on_failure=False,
    )
    projected = np.asarray(projection.state, dtype=float)
    smooth, equivalence = _strict_equivalence(theta, influent, analysis)
    smooth_response = (
        np.asarray(smooth.routes[0].response, dtype=float)
        if smooth is not None and smooth.accepted
        else np.full(analysis.direct_assets.response_count, np.nan)
    )
    reference, reference_state_1, reference_state_2, replay = _reference_two_start(
        theta, influent, analysis,
    )
    optimizer_root = {
        "applicable": route == "direct",
        "state_scaled_inf": None,
        "feed_tss_scaled_absolute": None,
        "maximum_scaled_difference": None,
        "branch_agreement": None,
        "accepted": True,
    }
    if route == "direct":
        if smooth is not None and smooth.accepted and smooth.routes[0].branch is not None:
            state_difference = float(np.max(np.abs(
                (np.asarray(smooth.routes[0].state) - np.asarray(final.state))
                / analysis.direct_assets.state_scale
            )))
            feed_difference = float(abs(
                float(smooth.routes[0].feed_tss) - float(final.feed_tss)
            ) / analysis.direct_assets.feed_scale)
            branch_agreement = bool(
                final.branch is not None
                and smooth_branches_match(smooth.routes[0].branch, final.branch)
            )
            optimizer_root = {
                "applicable": True,
                "state_scaled_inf": state_difference,
                "feed_tss_scaled_absolute": feed_difference,
                "maximum_scaled_difference": max(state_difference, feed_difference),
                "branch_agreement": branch_agreement,
                "accepted": bool(
                    max(state_difference, feed_difference) <= 1.0e-6
                    and branch_agreement
                ),
            }
        else:
            optimizer_root["accepted"] = False
    accepted = bool(
        projection.accepted
        and equivalence.get("accepted") is True
        and replay.get("accepted") is True
        and optimizer_root["accepted"] is True
        and derivative_audit.get("passed") is True
        and np.all(np.isfinite(raw))
        and np.all(np.isfinite(projected))
        and np.all(np.isfinite(smooth_response))
        and np.all(np.isfinite(reference))
    )
    equivalence = dict(equivalence)
    equivalence.update({
        "accepted": accepted,
        "smooth_reference_comparison_accepted": bool(
            equivalence.get("accepted") is True
        ),
        "independent_reference_replay_accepted": bool(replay.get("accepted") is True),
        "selected_route": route,
        "selected_start": int(selected.start_index),
        "cold_projection_accepted": bool(projection.accepted),
        "reference_replay": replay,
        "optimizer_root_reproduction": optimizer_root,
        "derivative_audit": derivative_audit,
    })
    selected_path = case_directory / f"{route}_selected.npz"
    selected_arrays: dict[str, np.ndarray] = {
        "start_index": np.asarray(selected.start_index),
        "theta": theta,
        "raw": raw,
        "projected": projected,
        "smooth": smooth_response,
        "reference": reference,
    }
    if route == "direct":
        selected_arrays.update({
            "response": np.asarray(final.response, dtype=float),
            "state": np.asarray(final.state, dtype=float),
            "optimized_response": np.asarray(final.response, dtype=float),
        })
    else:
        selected_arrays["continuation_projected"] = np.asarray(final.projected, dtype=float)
    atomic_npz(selected_path, **selected_arrays)
    reference_path = case_directory / f"{route}_reference.npz"
    atomic_npz(
        reference_path, theta=theta, response=reference,
        state=reference_state_1, state_start_2=reference_state_2,
    )
    equivalence_path = case_directory / f"{route}_equivalence.json"
    atomic_json(equivalence_path, equivalence, nonfinite_to_none=True)
    case_label = f"{case_id}:{route}"
    violations = pd.DataFrame([
        _physical_record(method, case_label, response, theta, influent, analysis)
        for method, response in (
            ("raw", raw), ("projected", projected),
            ("smooth", smooth_response), ("reference", reference),
        )
    ])
    violation_path = case_directory / f"{route}_physical_violations.csv"
    atomic_dataframe(violation_path, violations)
    audit_path = case_directory / f"{route}_cross_evaluation.json"
    atomic_json(audit_path, {
        "selection_contract": selection_contract,
        "accepted": accepted,
        "cold_projection_accepted": bool(projection.accepted),
        "cold_projection_diagnostics": projection.diagnostics.as_dict(),
        "smooth_fixed_input_accepted": bool(smooth is not None and smooth.accepted),
        "strict_equivalence_accepted": bool(
            equivalence["smooth_reference_comparison_accepted"]
        ),
        "reference_replay": replay,
        "optimizer_root_reproduction": optimizer_root,
        "derivative_audit_passed": bool(derivative_audit.get("passed") is True),
        "derivative_audit_status": derivative_audit.get("status"),
    }, nonfinite_to_none=True)
    paths = (
        case_directory / f"{route}.json", selected_path, reference_path,
        equivalence_path, violation_path, audit_path,
        case_directory / f"{route}_derivative_audit.json",
        case_directory / f"{route}_derivative_audit_complete.json",
    )
    atomic_json(case_directory / f"{route}_selection_complete.json", {
        "stage": "selected_decision_cross_evaluation",
        "selection_contract": selection_contract,
        "selected": True,
        "accepted": accepted,
        "artifacts": _artifact_hashes(case_directory, paths),
    })
    return True, accepted, violations


def _case_contract_id(
    source_id: str, analysis_id: str, case_id: str, influent: np.ndarray,
) -> str:
    return sha256(
        source_id.encode() + analysis_id.encode() + OPTIMIZATION_PROTOCOL.encode()
        + case_id.encode()
        + np.ascontiguousarray(influent, dtype="<f8").tobytes()
    ).hexdigest()


def _load_case_checkpoint(
    run: Path,
    case_directory: Path,
    *,
    case_contract: str,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    marker_path = case_directory / "case_complete.json"
    violations_path = case_directory / "physical_violations.csv"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("case_contract") != case_contract
            or not _artifacts_match(run, marker.get("artifacts", {}))
        ):
            return None
        violations = (
            pd.read_csv(violations_path)
            if bool(marker.get("selected_decision_count"))
            else pd.DataFrame()
        )
        return marker, violations
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def run_optimization_stage(
    *,
    run: Path,
    profile: StudyProfile,
    design: Mapping[str, object],
    development_targets: np.ndarray,
    test_targets: np.ndarray,
    analysis: AnalysisBundle,
    source_files: Mapping[str, str],
) -> bool:
    """Run searches and compare selected decisions on one exact reference.

    The frozen post-selection holdout remains reserved for descriptive
    surrogate assessment. Smooth/reference equivalence is not rerun over the
    rows. Instead, every available nominal/robustness decision is evaluated
    by the same exact nonsmooth mechanistic equations.  Search failures and
    unresolved convergence certificates are recorded casewise and never stop
    the remaining scientific comparisons.
    """

    validate_authorized_profile(profile)
    if profile.robustness_count != 10 or profile.layer_count != 10:
        raise RuntimeError(
            "optimization requires ten robustness cases and ten clarifier layers"
        )
    if not assessment_gate_allows_optimization(analysis.passed):
        raise RuntimeError("optimization cannot bypass the enforced admission gate")
    source_id = source_digest(source_files)
    casewise_source_id = _casewise_artifact_source_id(run, source_id)
    analysis_id = _assessment_binding(design, development_targets, test_targets)
    case_inputs = [
        ("nominal", np.asarray(NOMINAL_INFLUENT, dtype=float)),
        *(
            (f"robustness_{index + 1:02d}", np.asarray(row, dtype=float))
            for index, row in enumerate(np.asarray(design["robustness_influents"]))
        ),
    ]
    if len(case_inputs) != 11:
        raise RuntimeError("the full article requires nominal plus ten robustness cases")
    development_decisions = np.asarray(design["development_decisions"], dtype=float)
    development_influents = np.asarray(design["development_influents"], dtype=float)
    selected_physical_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    route_statuses: list[dict[str, Any]] = []
    shared_surrogate_problem: Any | None = None
    for case_number, (case_id, influent) in enumerate(case_inputs, start=1):
        _write_state(
            run, "casewise_common_reference", "running", case=case_id,
            completed_cases=case_number - 1, total_cases=len(case_inputs),
        )
        case_directory = run / "optimization" / case_id
        case_directory.mkdir(parents=True, exist_ok=True)
        if shared_surrogate_problem is None:
            shared_surrogate_problem = build_surrogate_nlp(
                analysis.surrogate_assets,
                GAP_CONTINUATION[-1],
                settings=SurrogateSolverSettings(maximum_wall_time=None),
                name="article_surrogate_exact_qp_expressions",
                compile_solver=False,
            )
        surrogate, surrogate_payload = _run_surrogate_route(
            case_directory, case_id=case_id, influent=influent,
            assets=analysis.surrogate_assets, source_id=casewise_source_id,
            analysis_id=analysis_id, problem=shared_surrogate_problem,
        )
        direct, direct_payload = _run_direct_route(
            case_directory, case_id=case_id, influent=influent,
            assets=analysis.direct_assets,
            development_decisions=development_decisions,
            development_influents=development_influents,
            development_targets=development_targets,
            source_id=casewise_source_id, analysis_id=analysis_id,
        )
        surrogate_candidate, certification = _run_surrogate_certification(
            case_directory,
            case_id=case_id,
            influent=influent,
            result=surrogate,
            analysis=analysis,
            source_id=casewise_source_id,
            analysis_id=analysis_id,
            problem=shared_surrogate_problem,
        )
        direct_for_comparison, recovery = _run_direct_failure_recovery(
            case_directory,
            case_id=case_id,
            influent=influent,
            result=direct,
            surrogate_candidate=(
                surrogate_candidate
                if certification.get("locally_converged") is True else None
            ),
            assets=analysis.direct_assets,
            development_decisions=development_decisions,
            development_influents=development_influents,
            development_targets=development_targets,
            source_id=casewise_source_id,
            analysis_id=analysis_id,
        )
        surrogate_evaluation, surrogate_physical = (
            _run_casewise_route_reference_evaluation(
                case_directory,
                case_id=case_id,
                route="surrogate",
                influent=influent,
                selected=surrogate.selected,
                surrogate_candidate=surrogate_candidate,
                route_payload=surrogate_payload,
                certification_payload=certification,
                recovery_payload=None,
                analysis=analysis,
                source_id=casewise_source_id,
                analysis_id=analysis_id,
            )
        )
        direct_evaluation, direct_physical = _run_casewise_route_reference_evaluation(
            case_directory,
            case_id=case_id,
            route="direct",
            influent=influent,
            selected=direct_for_comparison.selected,
            surrogate_candidate=None,
            route_payload=direct_payload,
            certification_payload=None,
            recovery_payload=recovery,
            analysis=analysis,
            source_id=casewise_source_id,
            analysis_id=analysis_id,
        )
        for evaluation in (surrogate_evaluation, direct_evaluation):
            reference = evaluation.get("reference", {})
            raw_error = evaluation.get("prediction_error_raw", {})
            projected_error = evaluation.get("prediction_error_projected", {})
            native_error = evaluation.get("native_model_error", {})
            normalized_controls = evaluation.get("normalized_controls")
            theta = evaluation.get("theta")
            row = {
                "case": case_id,
                "route": evaluation.get("route"),
                "candidate_available": evaluation.get("candidate_available"),
                "native_feasible": evaluation.get("native_feasible"),
                "exact_replay_valid": evaluation.get("exact_replay_valid"),
                "comparison_valid": evaluation.get("comparison_valid"),
                "status": evaluation.get("status"),
                "native_status": evaluation.get("native_status"),
                "native_objective": evaluation.get("native_objective"),
                "exact_reference_objective": evaluation.get(
                    "exact_reference_objective"
                ),
                "native_minus_reference_objective": evaluation.get(
                    "native_minus_reference_objective"
                ),
                "local_convergence_certified": evaluation.get(
                    "local_convergence_certified"
                ),
                "first_order_stationarity_certified": evaluation.get(
                    "first_order_stationarity_certified"
                ),
                "local_convergence_classification": evaluation.get(
                    "local_convergence_classification"
                ),
                "reference_status": reference.get("status"),
                "reference_source": reference.get("source"),
                "reference_start_1_accepted": reference.get("start_1_accepted"),
                "reference_start_2_accepted": reference.get("start_2_accepted"),
                "reference_scaled_root_difference_generation": reference.get(
                    "scaled_root_difference_generation"
                ),
                "reference_scaled_root_difference_state": reference.get(
                    "scaled_root_difference_state"
                ),
                "reference_branch_agreement": reference.get("branch_agreement"),
                "reference_branch_ambiguous": reference.get("branch_ambiguous"),
                "reference_minimum_normalized_branch_margin": reference.get(
                    "minimum_normalized_branch_margin"
                ),
                "reference_physical_stability_accepted": reference.get(
                    "physical_stability_accepted"
                ),
                "reference_engineering_feasible": reference.get(
                    "engineering_feasible"
                ),
                "raw_reference_nrmse": raw_error.get("nrmse"),
                "raw_reference_nmae": raw_error.get("nmae"),
                "raw_reference_scaled_inf": raw_error.get("scaled_inf"),
                "projected_reference_nrmse": projected_error.get("nrmse"),
                "projected_reference_nmae": projected_error.get("nmae"),
                "projected_reference_scaled_inf": projected_error.get("scaled_inf"),
                "optimizer_native_reference_nrmse": native_error.get("nrmse"),
                "optimizer_native_reference_nmae": native_error.get("nmae"),
                "optimizer_native_reference_scaled_inf": native_error.get("scaled_inf"),
                "optimization_elapsed_seconds": evaluation.get(
                    "optimization_elapsed_seconds"
                ),
                "primary_optimization_elapsed_seconds": evaluation.get(
                    "primary_optimization_elapsed_seconds"
                ),
                "recovery_elapsed_seconds": evaluation.get("recovery_elapsed_seconds"),
                "certification_elapsed_seconds": evaluation.get(
                    "certification_elapsed_seconds"
                ),
                "reference_elapsed_seconds": evaluation.get(
                    "reference_elapsed_seconds"
                ),
                "reference_current_run_reuse_overhead_seconds": reference.get(
                    "current_run_reuse_overhead_seconds"
                ),
                "reference_retained_original_solve_elapsed_seconds": reference.get(
                    "retained_original_solve_elapsed_seconds"
                ),
            }
            if isinstance(normalized_controls, list) and len(normalized_controls) == 7:
                row.update({
                    f"normalized_{name}": value
                    for name, value in zip(DECISION_NAMES, normalized_controls, strict=True)
                })
            if isinstance(theta, list) and len(theta) == 7:
                row.update({
                    name: value
                    for name, value in zip(DECISION_NAMES, theta, strict=True)
                })
            reference_rows.append(row)
        comparison = _casewise_comparison_row(
            case_id, surrogate_evaluation, direct_evaluation,
        )
        comparison_rows.append(comparison)
        case_comparison_path = case_directory / "common_reference_comparison.json"
        atomic_json(case_comparison_path, comparison, nonfinite_to_none=True)
        case_frames = [
            frame for frame in (surrogate_physical, direct_physical) if not frame.empty
        ]
        case_violations = (
            pd.concat(case_frames, ignore_index=True, sort=False)
            if case_frames else pd.DataFrame()
        )
        if not case_violations.empty:
            selected_physical_frames.append(case_violations)
        new_artifacts = (
            case_directory / "surrogate_local_convergence_complete.json",
            case_directory / "surrogate_casewise_reference_complete.json",
            case_directory / "direct_casewise_reference_complete.json",
            case_comparison_path,
        )
        case_contract = _case_contract_id(source_id, analysis_id, case_id, influent)
        route_rows = [
            {
                "case": case_id,
                "route": "surrogate",
                "primary_status": surrogate.status,
                "selected": surrogate_evaluation.get("candidate_available"),
                "comparison_valid": surrogate_evaluation.get("comparison_valid"),
                "locally_converged": surrogate_evaluation.get(
                    "local_convergence_certified"
                ),
                "first_order_stationary": surrogate_evaluation.get(
                    "first_order_stationarity_certified"
                ),
                "primary_attempts": 1,
                "recovery_attempts": 0,
            },
            {
                "case": case_id,
                "route": "direct",
                "primary_status": direct.status,
                "selected": direct_evaluation.get("candidate_available"),
                "comparison_valid": direct_evaluation.get("comparison_valid"),
                "locally_converged": direct_evaluation.get(
                    "local_convergence_certified"
                ),
                "first_order_stationary": direct_evaluation.get(
                    "first_order_stationarity_certified"
                ),
                "primary_attempts": 1,
                "recovery_attempts": int(bool(recovery.get("attempted"))),
            },
        ]
        route_statuses.extend(route_rows)
        assert_source_unchanged(source_files)
        atomic_json(case_directory / "casewise_comparison_complete.json", {
            "stage": "casewise_exact_common_reference",
            "case": case_id,
            "case_contract": case_contract,
            "routes": route_rows,
            "comparison_eligible": comparison["comparison_eligible"],
            "artifacts": _artifact_hashes(run, new_artifacts),
        })

    comparison_frame = pd.DataFrame(comparison_rows)
    reference_frame = pd.DataFrame(reference_rows)
    atomic_dataframe(
        run / "metrics" / "case_common_reference_comparison.csv", comparison_frame,
    )
    atomic_dataframe(
        run / "metrics" / "selected_candidate_reference_evaluation.csv",
        reference_frame,
    )
    legacy_rows = run / "validation" / "untouched_test_equivalence" / "rows"
    retired_count = len(tuple(legacy_rows.glob("row_*.json"))) if legacy_rows.is_dir() else 0
    retired_path = run / "metrics" / "untouched_test_equivalence_retired.json"
    atomic_json(retired_path, {
        "stage": "untouched_test_smooth_reference_equivalence",
        "status": "retired_incomplete_excluded_from_analysis",
        "validation_protocol": COMPARISON_PROTOCOL,
        "partial_legacy_rows_retained_but_unused": retired_count,
        "replacement_scope": "nominal_plus_ten_robustness_cases",
        "selected_candidate_count": int(reference_frame["candidate_available"].fillna(False).sum()),
        "paired_case_count": int(comparison_frame["comparison_eligible"].sum()),
    })
    preliminary_v1_archive = (
        run / str(CONVERGENCE_POLL_REFINEMENT_MIGRATION.retired_casewise_snapshot)
    )
    preliminary_v2_archive = (
        run / str(POLL_LINESEARCH_FORK_MIGRATION.retired_casewise_snapshot)
    )

    def archived_poll_statuses(archive: Path, protocol: str) -> list[str]:
        statuses: list[str] = []
        if not archive.is_dir():
            return statuses
        for case_id, _ in case_inputs:
            archived = archive / "optimization" / case_id / (
                "surrogate_local_convergence.json"
            )
            if archived.is_file():
                payload = _load_json_object(
                    archived, description="archived preliminary poll result",
                )
                if payload.get("certificate", {}).get("protocol") == protocol:
                    statuses.append(str(payload.get("status", "unknown")))
        return statuses

    preliminary_v1_statuses = archived_poll_statuses(
        preliminary_v1_archive, "exact_qp_two_scale_feasible_poll_v1",
    )
    preliminary_v2_statuses = archived_poll_statuses(
        preliminary_v2_archive, "exact_qp_two_scale_feasible_poll_v2",
    )
    refinement_record_path = run / "metrics" / "convergence_poll_refinement.json"
    atomic_json(refinement_record_path, {
        "stage": "surrogate_convergence_poll_refinement",
        "predecessor_protocol": "exact_qp_two_scale_feasible_poll_v1",
        "intermediate_protocol": "exact_qp_two_scale_feasible_poll_v2",
        "successor_protocol": "exact_qp_two_scale_accelerated_feasible_poll_v3",
        "predecessor_case_count": len(preliminary_v1_statuses),
        "predecessor_status_counts": {
            status: preliminary_v1_statuses.count(status)
            for status in sorted(set(preliminary_v1_statuses))
        },
        "intermediate_case_count": len(preliminary_v2_statuses),
        "intermediate_status_counts": {
            status: preliminary_v2_statuses.count(status)
            for status in sorted(set(preliminary_v2_statuses))
        },
        "reason": (
            "The full-rank feasible-direction rule is invalid at an active "
            "constrained endpoint, and the 120-evaluation budget interrupted "
            "otherwise-progressing v1 polls. The corrected v2 poll then exposed "
            "a fixed-step fine-radius crawl in robustness case 05. Protocol v3 "
            "adds exact-QP ray acceleration while preserving fresh final polls. "
            "Both predecessor result sets are archived and excluded."
        ),
        "retired_v1_archive": (
            None
            if not preliminary_v1_archive.is_dir()
            else preliminary_v1_archive.relative_to(run).as_posix()
        ),
        "retired_v2_archive": (
            None
            if not preliminary_v2_archive.is_dir()
            else preliminary_v2_archive.relative_to(run).as_posix()
        ),
    })
    _write_state(
        run, "robustness_timing_aggregation", "running",
        case_count=profile.robustness_count,
        nominal_case_included=False,
    )
    _run_robustness_case_timing_aggregation(
        run,
        source_files=source_files,
        analysis_id=analysis_id,
    )
    selected_physical = (
        pd.concat(selected_physical_frames, ignore_index=True, sort=False)
        if selected_physical_frames else pd.DataFrame(columns=("case", "method"))
    )
    atomic_dataframe(
        run / "metrics" / "selected_response_physical_audit.csv",
        selected_physical,
    )
    # A combined ledger gives the manuscript one explicit location containing
    # post-selection holdout prediction audits and casewise response audits.
    assessment_physical = pd.read_csv(
        run / "metrics" / "physical_violations_assessment.csv"
    )
    combined_frames = []
    for scope, frame in (
        ("post_selection_holdout", assessment_physical),
        ("selected_decision_common_reference", selected_physical),
    ):
        item = frame.copy()
        item.insert(0, "analysis_scope", scope)
        combined_frames.append(item)
    atomic_dataframe(
        run / "metrics" / "physical_violations_all_analysis.csv",
        pd.concat(combined_frames, ignore_index=True, sort=False),
    )
    expected_cases = tuple(case_id for case_id, _ in case_inputs)
    report = write_reporting_tables(
        run,
        output_directory=run / "report" / "tables",
        expected_cases=expected_cases,
    )
    report_manifest = run / "report" / "tables" / "report_manifest.json"
    available_reference = reference_frame["candidate_available"].fillna(False)
    exact_replay_valid = reference_frame.loc[
        available_reference, "exact_replay_valid"
    ].fillna(False)
    reference_valid = reference_frame.loc[
        available_reference, "comparison_valid"
    ].fillna(False)
    surrogate_certified = reference_frame.loc[
        reference_frame["route"].eq("surrogate") & available_reference,
        "local_convergence_certified",
    ].fillna(False)
    scientific_passed = bool(
        len(reference_frame) == 2 * len(case_inputs)
        and int(available_reference.sum()) == 2 * len(case_inputs)
        and bool(reference_valid.all())
        and len(surrogate_certified) == len(case_inputs)
        and bool(surrogate_certified.all())
        and int(comparison_frame["comparison_eligible"].sum()) == len(case_inputs)
    )
    selected_count = int(available_reference.sum())
    paired_count = int(comparison_frame["comparison_eligible"].sum())
    final_status_path = run / "optimization" / "final_status.json"
    atomic_json(final_status_path, {
        "case_count": len(case_inputs),
        "route_count": len(route_statuses),
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "validation_protocol": COMPARISON_PROTOCOL,
        "required_attempts_per_route": 1,
        "required_starts_per_route": 1,
        "surrogate_ipopt_continuation_stage_count": 0,
        "direct_smoothing_continuation_stage_count": len(CONTINUATION_SCHEDULE),
        "wall_time_ceiling": None,
        "untouched_test_equivalence_executed": False,
        "retired_partial_untouched_test_equivalence_rows": retired_count,
        "selected_decision_count": selected_count,
        "exact_reference_valid_selected_decision_count": int(exact_replay_valid.sum()),
        "exact_reference_engineering_feasible_selected_decision_count": int(
            reference_valid.sum()
        ),
        "paired_common_reference_case_count": paired_count,
        "surrogate_locally_converged_count": int(surrogate_certified.sum()),
        "casewise_reference_validation_passed": bool(
            selected_count == 2 * len(case_inputs)
            and len(exact_replay_valid) == 2 * len(case_inputs)
            and exact_replay_valid.all()
        ),
        "all_pairs_comparison_eligible": paired_count == len(case_inputs),
        "scientific_validation_passed": scientific_passed,
        "report_warning_count": len(report.warnings),
        "routes": route_statuses,
    }, nonfinite_to_none=True)
    assert_source_unchanged(source_files)
    final_paths = (
        run / "metrics" / "selected_response_physical_audit.csv",
        run / "metrics" / "physical_violations_all_analysis.csv",
        run / "metrics" / "case_common_reference_comparison.csv",
        run / "metrics" / "selected_candidate_reference_evaluation.csv",
        retired_path,
        refinement_record_path,
        final_status_path,
        report_manifest,
        run / "metrics" / "robustness_case_timing_complete.json",
        run / "metrics" / "robustness_case_timing.csv",
        run / "metrics" / "timing_events.csv",
        run / "metrics" / "robustness_case_timing_summary.json",
        *(run / "optimization" / case_id / "casewise_comparison_complete.json"
          for case_id, _ in case_inputs),
        *(path for path in sorted((run / "report" / "tables").glob("*.csv"))),
    )
    atomic_json(run / "optimization" / "optimization_complete.json", {
        "stage": "article_optimization_replay_reporting",
        "source_digest": source_id,
        "input_digest": analysis_id,
        "case_count": 11,
        "route_count": 22,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "required_attempts_per_route": 1,
        "selected_decision_count": selected_count,
        "paired_common_reference_case_count": paired_count,
        "untouched_test_equivalence_executed": False,
        "scientific_validation_passed": scientific_passed,
        "artifacts": _artifact_hashes(run, final_paths),
    })
    return scientific_passed


def _prepare_run_directories(run: Path) -> None:
    for relative in (
        "inputs", "datasets", "models", "predictions", "metrics",
        "optimization", "report/tables", "report/figures",
    ):
        (run / relative).mkdir(parents=True, exist_ok=True)


def _write_state(run: Path, stage: str, status: str, **details: Any) -> None:
    atomic_json(run / "run_state.json", {
        "stage": stage, "status": status, "pid": os.getpid(), **details,
    })


def main(
    run_id: str,
    through: str,
    *,
    profile: StudyProfile = ARTICLE_FULL,
    use_frozen_accepted_checkpoints: bool = False,
    use_random_sampled_accepted_checkpoints: bool = False,
    reuse_from_run_id: str | None = None,
    authorize_generation_replacement_migration: bool = False,
    authorize_assessment_recovery_migration: bool = False,
    authorize_single_start_exact_qp_migration: bool = False,
    authorize_casewise_common_reference_migration: bool = False,
    authorize_convergence_poll_refinement_migration: bool = False,
    authorize_casewise_timing_migration: bool = False,
    authorize_fresh_route_loader_fix_migration: bool = False,
) -> None:
    if through not in {"generation", "assessment", "complete"}:
        raise ValueError("through must be generation, assessment, or complete")
    validate_authorized_profile(profile)
    if (
        use_frozen_accepted_checkpoints
        != (profile.name == FROZEN_PROFILE_NAME)
        or use_random_sampled_accepted_checkpoints
        != (profile.name == SAMPLED_PROFILE_NAME)
    ):
        raise ValueError("accepted-checkpoint mode and profile must be selected together")
    run = resolve_run_directory(run_id)
    dataset_total = profile.development_count + profile.test_count
    run_id_total = 50_000 if use_frozen_accepted_checkpoints else dataset_total
    if not run_id.startswith(f"article_full_{run_id_total}_"):
        raise ValueError(
            f"run id {run_id!r} does not match the {run_id_total}-row run lineage"
        )
    source_files = source_file_digests()
    contract = _build_contract(run_id, profile, source_files)
    if reuse_from_run_id is not None:
        initialize_reused_run(
            run,
            source_run_id=reuse_from_run_id,
            successor_contract=contract,
        )
    _prepare_run_directories(run)
    establish_contract(
        run,
        contract,
        authorize_generation_replacement_migration=(
            authorize_generation_replacement_migration
        ),
        authorize_assessment_recovery_migration=(
            authorize_assessment_recovery_migration
        ),
        authorize_single_start_exact_qp_migration=(
            authorize_single_start_exact_qp_migration
        ),
        authorize_casewise_common_reference_migration=(
            authorize_casewise_common_reference_migration
        ),
        authorize_convergence_poll_refinement_migration=(
            authorize_convergence_poll_refinement_migration
        ),
        authorize_casewise_timing_migration=authorize_casewise_timing_migration,
        authorize_fresh_route_loader_fix_migration=(
            authorize_fresh_route_loader_fix_migration
        ),
    )
    try:
        if use_frozen_accepted_checkpoints:
            _write_state(run, "dataset_freeze", "running")
            generation = freeze_accepted_generation(
                run, profile=profile, source_files=source_files,
            )
        elif use_random_sampled_accepted_checkpoints:
            _write_state(run, "dataset_sampling", "running")
            generation = sample_accepted_generation(
                run, profile=profile, source_files=source_files,
            )
        else:
            design = load_or_create_design(run, profile)
            assert_source_unchanged(source_files)
            _write_state(run, "generation", "running")
            generation = run_generation(
                run, design, profile=profile, source_files=source_files,
            )
        design = generation.design
        development_targets = generation.development_targets
        test_targets = generation.test_targets
        if through == "generation":
            _write_state(run, "generation", "complete")
            return
        assessment_id = _assessment_binding(design, development_targets, test_targets)
        existing_gate = load_assessment_checkpoint(
            run, source_id=source_digest(source_files), input_id=assessment_id,
        )
        if (
            existing_gate is not None
            and not assessment_gate_allows_optimization(existing_gate["passed"])
        ):
            _write_state(run, "assessment", "admission_gate_failed")
            raise RuntimeError(
                f"full article admission gate failed; see "
                f"{run / 'metrics/admission_gate.json'}"
            )
        if through == "assessment" and existing_gate is not None:
            gate_passed = bool(existing_gate["passed"])
            _write_state(
                run, "assessment",
                "complete" if gate_passed else "complete_with_advisory_failures",
                admission_gate_passed=gate_passed,
                assessment_gate_execution_policy=ASSESSMENT_GATE_EXECUTION_POLICY,
            )
            return
        _write_state(run, "assessment", "running")
        analysis = run_assessment(
            run, design, development_targets, test_targets,
            profile=profile, source_files=source_files,
        )
        if not assessment_gate_allows_optimization(analysis.passed):
            _write_state(run, "assessment", "admission_gate_failed")
            raise RuntimeError(
                f"full article admission gate failed; see "
                f"{run / 'metrics/admission_gate.json'}"
            )
        if through == "assessment":
            _write_state(
                run, "assessment",
                "complete" if analysis.passed else "complete_with_advisory_failures",
                admission_gate_passed=bool(analysis.passed),
                assessment_gate_execution_policy=ASSESSMENT_GATE_EXECUTION_POLICY,
            )
            return
        _write_state(run, "optimization", "running")
        scientific_passed = run_optimization_stage(
            run=run, profile=profile, design=design,
            development_targets=development_targets, test_targets=test_targets,
            analysis=analysis, source_files=source_files,
        )
        assert_source_unchanged(source_files)
        _write_state(
            run, "complete",
            (
                "complete" if scientific_passed and analysis.passed
                else "complete_with_validation_failures"
            ),
            admission_gate_passed=bool(analysis.passed),
            assessment_gate_execution_policy=ASSESSMENT_GATE_EXECUTION_POLICY,
            optimization_validation_passed=scientific_passed,
            scientific_validation_passed=bool(
                analysis.passed and scientific_passed
            ),
        )
    except Exception as exc:
        state_path = run / "run_state.json"
        current = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file() else {}
        )
        if current.get("status") != "admission_gate_failed":
            _write_state(
                run, str(current.get("stage", "unknown")), "failed",
                error_type=type(exc).__name__, message=str(exc),
            )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", default=os.environ.get("ARTICLE_V3_RUN_ID", DEFAULT_RUN_ID),
    )
    parser.add_argument(
        "--dataset-count",
        type=int,
        choices=AUTHORIZED_DATASET_TOTALS,
        default=int(os.environ.get("ARTICLE_V3_DATASET_COUNT", "5000")),
        help="accepted development-plus-test rows; the split remains 80/20",
    )
    parser.add_argument(
        "--use-frozen-accepted-checkpoints",
        action="store_true",
        help=(
            "freeze the 16,714 accepted checkpoints in the interrupted 50k "
            "run into a 13,371/3,343 development/holdout dataset"
        ),
    )
    parser.add_argument(
        "--use-random-sampled-accepted-checkpoints",
        action="store_true",
        help=(
            "randomly sample 10,000 rows without replacement from the 16,714 "
            "accepted frozen checkpoints, then use the fixed 8,000/2,000 split"
        ),
    )
    parser.add_argument(
        "--reuse-from-run-id",
        default=os.environ.get("ARTICLE_V3_REUSE_FROM_RUN_ID"),
        help=(
            "create a new self-contained rerun directory by byte-copying the "
            "validated upstream and primary-search artifacts from this run id"
        ),
    )
    parser.add_argument(
        "--through", choices=("generation", "assessment", "complete"),
        default="complete",
    )
    parser.add_argument(
        "--authorize-generation-replacement-migration",
        action="store_true",
        help=(
            "apply the one-time, pinned article-v3 generation-replacement source "
            "contract migration to the existing default run"
        ),
    )
    parser.add_argument(
        "--authorize-assessment-recovery-migration",
        action="store_true",
        help=(
            "apply the pinned article-v3 projection/audit recovery migration "
            "to the existing default run"
        ),
    )
    parser.add_argument(
        "--authorize-single-start-exact-qp-migration",
        action="store_true",
        help=(
            "apply the pinned article-v3 single-center exact-QP optimization "
            "migration to the existing default run"
        ),
    )
    parser.add_argument(
        "--authorize-casewise-common-reference-migration",
        action="store_true",
        help=(
            "apply the pinned casewise exact common-reference migration while "
            "retaining completed generation, fit, assessment, and optimization"
        ),
    )
    parser.add_argument(
        "--authorize-convergence-poll-refinement-migration",
        action="store_true",
        help=(
            "apply the pinned casewise convergence-poll refinement while "
            "retaining all upstream artifacts and completed primary searches"
        ),
    )
    parser.add_argument(
        "--authorize-casewise-timing-migration",
        action="store_true",
        help=(
            "retire the unfinished repeated untouched-test timing benchmark and "
            "resume from the completed robustness-case timing records"
        ),
    )
    parser.add_argument(
        "--authorize-fresh-route-loader-fix-migration",
        action="store_true",
        help=(
            "apply the pinned fresh-route loader correction while retaining "
            "the completed sampled-data assessment checkpoints"
        ),
    )
    arguments = parser.parse_args()
    if (
        arguments.use_frozen_accepted_checkpoints
        and arguments.use_random_sampled_accepted_checkpoints
    ):
        parser.error("only one accepted-checkpoint mode may be selected")
    selected_profile = (
        frozen_accepted_profile()
        if arguments.use_frozen_accepted_checkpoints
        else sampled_accepted_profile()
        if arguments.use_random_sampled_accepted_checkpoints
        else profile_for_dataset_total(arguments.dataset_count)
    )
    reuse_from_run_id = arguments.reuse_from_run_id
    migration_requested = any((
        arguments.authorize_generation_replacement_migration,
        arguments.authorize_assessment_recovery_migration,
        arguments.authorize_single_start_exact_qp_migration,
        arguments.authorize_casewise_common_reference_migration,
        arguments.authorize_convergence_poll_refinement_migration,
        arguments.authorize_casewise_timing_migration,
        arguments.authorize_fresh_route_loader_fix_migration,
    ))
    if (
        reuse_from_run_id is None
        and arguments.run_id == DEFAULT_RUN_ID
        and not migration_requested
    ):
        reuse_from_run_id = LEGACY_RUN_ID
    main(
        arguments.run_id,
        arguments.through,
        profile=selected_profile,
        use_frozen_accepted_checkpoints=arguments.use_frozen_accepted_checkpoints,
        use_random_sampled_accepted_checkpoints=(
            arguments.use_random_sampled_accepted_checkpoints
        ),
        reuse_from_run_id=reuse_from_run_id,
        authorize_generation_replacement_migration=(
            arguments.authorize_generation_replacement_migration
        ),
        authorize_assessment_recovery_migration=(
            arguments.authorize_assessment_recovery_migration
        ),
        authorize_single_start_exact_qp_migration=(
            arguments.authorize_single_start_exact_qp_migration
        ),
        authorize_casewise_common_reference_migration=(
            arguments.authorize_casewise_common_reference_migration
        ),
        authorize_convergence_poll_refinement_migration=(
            arguments.authorize_convergence_poll_refinement_migration
        ),
        authorize_casewise_timing_migration=(
            arguments.authorize_casewise_timing_migration
        ),
        authorize_fresh_route_loader_fix_migration=(
            arguments.authorize_fresh_route_loader_fix_migration
        ),
    )
