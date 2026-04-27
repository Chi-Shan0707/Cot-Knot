from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knot_glm_common import NAD_ROOT, REPO_ROOT


MATH_SYMPTOMS = (
    "case_split_instability",
    "assumption_drift",
    "subgoal_frame_loss",
    "variable_binding_drift",
    "invariant_drift",
    "repair_without_recovery",
)

SCIENCE_SYMPTOMS = (
    "entity_binding_drift",
    "mechanism_frame_swap",
    "constraint_mismatch",
    "evidence_misread",
    "unit_scale_slip",
    "repair_without_recovery",
)

CODING_SYMPTOMS = (
    "spec_state_drift",
    "variable_role_swap",
    "branch_entanglement",
    "loop_state_forgetting",
    "patch_backtracking",
    "invariant_drop",
)


@dataclass(frozen=True)
class DomainConfig:
    domain: str
    report_paths: dict[str, Path]
    symptoms: tuple[str, ...]
    default_n_problems: int
    default_runs_per_problem: int
    default_requests_per_second: float
    balance_groups: bool
    default_out_dir: Path
    calibration_out_dir: Path


DOMAIN_CONFIGS = {
    "math": DomainConfig(
        domain="math",
        report_paths={
            "aime24": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610/evaluation_report.json",
            "aime25": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548/evaluation_report.json",
            "brumo25": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142/evaluation_report.json",
            "hmmt25": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151/evaluation_report.json",
        },
        symptoms=MATH_SYMPTOMS,
        default_n_problems=64,
        default_runs_per_problem=4,
        default_requests_per_second=2.0,
        balance_groups=True,
        default_out_dir=REPO_ROOT / "results" / "glm_math_knot_raw_v4",
        calibration_out_dir=REPO_ROOT / "results" / "glm_math_knot_raw_v4_calibration",
    ),
    "science": DomainConfig(
        domain="science",
        report_paths={
            "gpqa": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853/evaluation_report.json",
        },
        symptoms=SCIENCE_SYMPTOMS,
        default_n_problems=64,
        default_runs_per_problem=4,
        default_requests_per_second=2.5,
        balance_groups=False,
        default_out_dir=REPO_ROOT / "results" / "glm_science_knot_raw_v4",
        calibration_out_dir=REPO_ROOT / "results" / "glm_science_knot_raw_v4_calibration",
    ),
    "coding": DomainConfig(
        domain="coding",
        report_paths={
            "livecodebench_v5": NAD_ROOT / "MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808/evaluation_report.json",
        },
        symptoms=CODING_SYMPTOMS,
        default_n_problems=64,
        default_runs_per_problem=4,
        default_requests_per_second=2.0,
        balance_groups=False,
        default_out_dir=REPO_ROOT / "results" / "glm_coding_knot_raw_v4",
        calibration_out_dir=REPO_ROOT / "results" / "glm_coding_knot_raw_v4_calibration",
    ),
}
