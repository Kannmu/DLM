import math

FOCUS_CENTER = (0.0, 0.0, 0.1)
MODULATION_FREQUENCY_HZ = 200.0
TRIAL_STIMULUS_DURATION_S = 1.5
INTERVAL_GAP_DURATION_S = 1.0
STAIRCASE_REVERSALS_TARGET = 8
STAIRCASE_REVERSAL_MEAN_COUNT = 4
STAIRCASE_CONSECUTIVE_CORRECT_FOR_DOWN = 3
STAIRCASE_MIN_TRIALS = 12
STAIRCASE_MAX_TRIALS = 80
INITIAL_STRENGTH = 55.0
MIN_STRENGTH = 5.0
MAX_STRENGTH = 100.0
STRENGTH_STEP_INTENSITY = 0.08
DEFAULT_STEP_SIZE = 0.08
DEFAULT_PORT_SCAN_TEXT = "Click 'Scan Ports' to find the UMH device."
DEFAULT_DEMO_SCAN_TEXT = "Click 'Scan Demos' to map the on-device demo names."


def intensity_to_strength(intensity_level: float) -> float:
    intensity = max(0.0, min(1.0, float(intensity_level)))
    strength = (200.0 / math.pi) * math.asin(math.sqrt(intensity))
    return max(MIN_STRENGTH, min(MAX_STRENGTH, strength))


def strength_to_intensity(strength: float) -> float:
    normalized = max(0.0, min(100.0, float(strength))) / 100.0
    return math.sin((math.pi / 2.0) * normalized) ** 2


def build_experiment2_strength_grid(step_intensity: float = STRENGTH_STEP_INTENSITY):
    step = max(0.001, float(step_intensity))
    levels = []
    current = 0.0
    while current < 1.0 + 1e-9:
        levels.append(round(min(current, 1.0), 6))
        current += step
    if levels[-1] < 1.0:
        levels.append(1.0)
    strengths = [round(intensity_to_strength(level), 2) for level in levels]
    unique_strengths = []
    for value in strengths:
        if not unique_strengths or abs(unique_strengths[-1] - value) > 1e-6:
            unique_strengths.append(value)
    return unique_strengths


EXPERIMENT2_STRENGTH_GRID = build_experiment2_strength_grid()

SUPP1_MACH_LEVELS = [0.2, 0.5, 1.0, 1.2, 2.0]
SHEAR_WAVE_SPEED_M_PER_S = 5.0
ULM_BASELINE_STRENGTH = 80.0

MODULATION_DEFINITIONS = {
    "DLM_2": {
        "label": "DLM_2",
        "type": "demo",
        "description": "Discrete 2-point circular switching",
    },
    "DLM_3": {
        "label": "DLM_3",
        "type": "demo",
        "description": "Discrete 3-point circular switching",
    },
    "ULM_L": {
        "label": "ULM_L",
        "type": "linear",
        "description": "Unidirectional linear scan, Mach 1.2",
        "start_point": (-0.015, 0.0, 0.1),
        "end_point": (0.015, 0.0, 0.1),
        "strength": ULM_BASELINE_STRENGTH,
        "frequency": MODULATION_FREQUENCY_HZ,
        "bidirectional": False,
    },
    "LM_L": {
        "label": "LM_L",
        "type": "demo",
        "description": "Bidirectional linear scan, Mach 1.2",
    },
    "LM_C": {
        "label": "LM_C",
        "type": "demo",
        "description": "Continuous circular scan, Mach 1.2",
    },
}


def build_supplementary_ulm_conditions():
    conditions = {}
    for mach in SUPP1_MACH_LEVELS:
        half_length = 0.5 * (mach * SHEAR_WAVE_SPEED_M_PER_S / MODULATION_FREQUENCY_HZ)
        start_point = (-half_length, 0.0, FOCUS_CENTER[2])
        end_point = (half_length, 0.0, FOCUS_CENTER[2])
        key = f"ULM_M{mach:.1f}".replace(".", "p")
        conditions[key] = {
            "label": f"ULM_L (M={mach:.1f})",
            "type": "linear",
            "description": f"ULM linear scan with Mach {mach:.1f}",
            "start_point": start_point,
            "end_point": end_point,
            "strength": ULM_BASELINE_STRENGTH,
            "frequency": MODULATION_FREQUENCY_HZ,
            "bidirectional": False,
            "mach": mach,
            "scan_length_m": mach * SHEAR_WAVE_SPEED_M_PER_S / MODULATION_FREQUENCY_HZ,
        }
    return conditions


SUPP1_CONDITIONS = build_supplementary_ulm_conditions()

EXPERIMENT_MODE_DEFINITIONS = {
    "Experiment 2": {
        "task_type": "threshold",
        "label": "Experiment 2: Absolute Detection Threshold",
        "conditions": ["DLM_2", "DLM_3", "ULM_L", "LM_L", "LM_C"],
    },
    "Supplementary Experiment 1": {
        "task_type": "pairwise",
        "label": "Supplementary Experiment 1: ULM Mach Comparison",
        "conditions": list(SUPP1_CONDITIONS.keys()),
        "blocks": ["Intensity", "Spatial"],
    },
}
