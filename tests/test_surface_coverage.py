"""Module-to-test coverage guard.

Requirement (openspec/specs/verification/surface-coverage): every module under
``src/`` other than ``__init__.py`` is mapped to a test file that imports it and
exercises at least one public behaviour. A module that is not mapped fails this
test with the module path and the expected test filename, so a new untested
module becomes a red test rather than a review miss.

The mapping is explicit, not a filename heuristic: ``tests/test_droid_engine.py``
covers ``src/model/droid.py``, and renaming files to satisfy a pattern would add
churn without adding coverage.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TESTS_ROOT = REPO_ROOT / "tests"

# module path (relative to the repo root) -> test file that covers it
MODULE_TESTS = {
    "src/model/parallel_scan.py": "tests/test_parallel_scan.py",
    "src/cli.py": "tests/test_cli.py",
    "src/data.py": "tests/test_data.py",
    "src/mcp_server.py": "tests/test_mcp_server.py",
    "src/model/byte_rtu_backend.py": "tests/test_droid_rtu_fusion.py",
    "src/model/confidence.py": "tests/test_confidence.py",
    "src/model/decision_head.py": "tests/test_decision_head.py",
    "src/model/droid.py": "tests/test_droid_engine.py",
    "src/model/droid_manager.py": "tests/test_droid_profiles.py",
    "src/model/droid_package.py": "tests/test_droid_package.py",
    "src/model/delta_memory.py": "tests/test_delta_memory.py",
    "src/model/sleep_consolidation.py": "tests/test_sleep_consolidation.py",
    "src/model/hybrid_retrieval.py": "tests/test_hybrid_retrieval.py",
    "src/model/edge_slm.py": "tests/test_edge_slm.py",
    "src/model/test_time_compute.py": "tests/test_test_time_compute.py",
    "src/model/continual_bench.py": "tests/test_continual_bench.py",
    "src/model/backend_contracts.py": "tests/test_backend_contracts.py",
    "src/model/torch_decoder_backend.py": "tests/test_torch_decoder_backend.py",
    "src/model/gguf_backend.py": "tests/test_gguf_backend.py",
    "src/model/knowledge_store.py": "tests/test_knowledge_store.py",
    "src/model/memory_injection.py": "tests/test_memory_injection.py",
    "src/model/memory_lifecycle.py": "tests/test_memory_lifecycle.py",
    "src/model/onnx_anchor.py": "tests/test_onnx_anchor.py",
    "src/model/plastic_adapter.py": "tests/test_plastic_adapter.py",
    "src/model/propositions.py": "tests/test_propositions.py",
    "src/model/benchmarks.py": "tests/test_benchmarks.py",
    "src/model/probes.py": "tests/test_probes.py",
    "src/model/web_teacher.py": "tests/test_web_teacher.py",
    "src/model/search_learn.py": "tests/test_search_learn.py",
    "src/model/edge_pretraining.py": "tests/test_edge_pretraining.py",
    "src/model/rtu.py": "tests/test_rtu.py",
    "src/model/teacher.py": "tests/test_teacher.py",
    "src/model/cores/contracts.py": "tests/test_core_candidates.py",
    "src/model/cores/param_matcher.py": "tests/test_core_candidates.py",
    "src/model/cores/diagonal_cores.py": "tests/test_core_candidates.py",
    "src/model/cores/ssm_core.py": "tests/test_core_candidates.py",
    "src/model/cores/matrix_cores.py": "tests/test_core_candidates.py",
    "src/model/cores/attention_cores.py": "tests/test_core_candidates.py",
    "src/model/cores/registry.py": "tests/test_core_candidates.py",
    "src/model/cores/tokenizer.py": "tests/test_candidate_objectives.py",
    "src/model/cores/objectives.py": "tests/test_candidate_objectives.py",
    "src/model/tournament/trainer.py": "tests/test_candidate_objectives.py",
    "src/model/cores/decision_readouts.py": "tests/test_action_router.py",
    "src/model/cores/calibration.py": "tests/test_action_router.py",
    "src/model/cores/routers.py": "tests/test_action_router.py",
    "src/model/cores/memory_arms.py": "tests/test_candidate_stores.py",
    "src/model/tournament/workloads.py": "tests/test_architecture_tournament.py",
    "src/model/tournament/large_workloads.py": "tests/test_architecture_tournament.py",
    "src/model/tournament/stats.py": "tests/test_architecture_tournament.py",
    "src/model/tournament/funnel.py": "tests/test_architecture_tournament.py",
    "src/model/recamem_core.py": "tests/test_recamem_core.py",
}

# Modules covered by pre-existing test files: the steady state must keep these.
PRE_EXISTING_MODULES = {
    "src/model/droid.py",
    "src/model/rtu.py",
    "src/model/droid_manager.py",
    "src/model/knowledge_store.py",
    "src/model/onnx_anchor.py",
    "src/model/teacher.py",
    "src/data.py",
    "src/model/byte_rtu_backend.py",
}


def _shipped_modules():
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in SRC_ROOT.rglob("*.py")
        if path.name != "__init__.py"
    }


def _expected_test_file(module_path):
    stem = Path(module_path).stem
    return f"tests/test_{stem}.py"


def test_every_shipped_module_has_an_explicit_test_mapping():
    untested = sorted(_shipped_modules() - set(MODULE_TESTS))
    assert not untested, (
        "Untested modules (add a test file or an entry to MODULE_TESTS):\n"
        + "\n".join(f"  {module} -> expected {_expected_test_file(module)}" for module in untested)
    )


def test_mapping_has_no_stale_entries():
    shipped = _shipped_modules()
    stale = sorted(set(MODULE_TESTS) - shipped)
    assert not stale, f"MODULE_TESTS names modules that no longer exist: {stale}"


def test_mapping_covers_the_pre_existing_modules():
    missing = sorted(PRE_EXISTING_MODULES - set(MODULE_TESTS))
    assert not missing, f"pre-existing coverage lost from MODULE_TESTS: {missing}"


def test_mapped_test_files_exist_and_contain_a_test():
    problems = []
    for module, test_file in sorted(MODULE_TESTS.items()):
        path = REPO_ROOT / test_file
        if not path.exists():
            problems.append(f"{module}: mapped test file {test_file} does not exist")
            continue
        if not re.search(r"^def test_|^\s+def test_", path.read_text(encoding="utf-8"), re.M):
            problems.append(f"{module}: {test_file} contains no test function")
    assert not problems, "Coverage mapping problems:\n" + "\n".join(f"  {p}" for p in problems)


def test_mapped_test_files_import_the_module_they_cover():
    problems = []
    for module, test_file in sorted(MODULE_TESTS.items()):
        path = REPO_ROOT / test_file
        if not path.exists():
            problems.append(f"{module}: mapped test file {test_file} does not exist")
            continue
        source = path.read_text(encoding="utf-8")
        dotted = module.removesuffix(".py").replace("/", ".")
        if dotted not in source and module not in source:
            problems.append(f"{module}: {test_file} never references {dotted}")
    assert not problems, "Unreferenced modules:\n" + "\n".join(f"  {p}" for p in problems)


def test_mapping_targets_live_under_the_tests_directory():
    stray = sorted(f for f in MODULE_TESTS.values() if not f.startswith("tests/test_"))
    assert not stray, f"test mapping must point at tests/test_*.py: {stray}"


def test_expected_filename_derivation_matches_per_module_convention():
    # Documented naming convention used in the failure message above.
    assert _expected_test_file("src/cli.py") == "tests/test_cli.py"
    assert _expected_test_file("src/model/memory_lifecycle.py") == "tests/test_memory_lifecycle.py"
