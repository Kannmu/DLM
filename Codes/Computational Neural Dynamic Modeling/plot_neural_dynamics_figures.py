from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, get_window, sosfilt, find_peaks


# =============================================================================
# Configuration
# =============================================================================

METHOD_ORDER = ['ULM_L', 'DLM_2', 'DLM_3', 'LM_C', 'LM_L']
PRIMARY_COMPARE = ['ULM_L', 'LM_L']
RASTER_METHODS = ['ULM_L', 'DLM_2', 'LM_L']
ORTHO_COMPONENTS = ['xy', 'xz', 'yz']
COMPONENT_LABELS = {'xy': 'XY', 'xz': 'XZ', 'yz': 'YZ'}
COMPONENT_COLORS = {'xy': '#d55e00', 'xz': '#009e73', 'yz': '#0072b2'}

MAT_FILE_PATH = Path(r'k:\Work\SWIM\Sim\Outputs_Experiment1\experiment1_data.mat')
MODEL_DIR = Path(r'd:/Data/OneDrive/Papers/SWIM/Reference/Neural Dynamics Model V2')
MODEL_RESULTS_DIR = MODEL_DIR / 'data' / 'results'
EXPERIMENT_ANALYSIS_DIR = Path(r'd:/Data/OneDrive/Papers/SWIM/Codes/Experiment 1/Analysis')
OUTPUT_DIR = Path(r'd:/Data/OneDrive/Papers/SWIM/Codes/Computational Neural Dynamic Modeling')

FONT_NAME = 'Arial'
SNS_CONTEXT = 'talk'
SNS_STYLE = 'white'
TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 16
LINE_WIDTH = 3.2
DPI = 300

FIGSIZE_SINGLE = (8, 6)
FIGSIZE_Figure_4 = (15, 6.4)
FIGSIZE_TALL = (8.5, 9.5)
FIGSIZE_MATRIX = (18, 4.7)
FIGSIZE_REGRESSION = (8.2, 6.4)

BANDPASS_LOW = 150.0
BANDPASS_HIGH = 300.0
BANDPASS_ORDER = 4
CARRIER_FREQ = 200.0
SHEAR_SPEED = 5.0
LAMBDA_SPACE = 0.004
RECEPTOR_SPACING = 0.002
N_RASTER_NEURONS = 50
RASTER_DURATION_MS = 15.0
WATERFALL_RECEPTORS = 11
FREQ_MAX = 800.0

METHOD_COLORS = {
    'ULM_L': '#440154',
    'DLM_2': '#3b528b',
    'DLM_3': '#21918c',
    'LM_C': '#5ec962',
    'LM_L': '#fde725',
}
SPECTRUM_COLORS = {'ULM_L': '#440154', 'LM_L': '#fde725'}
POPULATION_CMAP = 'magma'
WAVE_CMAP = LinearSegmentedColormap.from_list('shear_div', ['#3b4cc0', '#f7f7f7', '#b40426'])


# =============================================================================
# Styling
# =============================================================================


def setup_style() -> None:
    sns.set_theme(context=SNS_CONTEXT, style=SNS_STYLE)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [FONT_NAME, 'DejaVu Sans']
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE
    plt.rcParams['legend.fontsize'] = LEGEND_SIZE
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# Data loading
# =============================================================================


def ensure_time_last(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        return arr
    time_axis = int(np.argmax(arr.shape))
    if time_axis != 2:
        arr = np.moveaxis(arr, time_axis, 2)
    return arr


class KWaveMatLoader:
    def __init__(self, path: Path):
        self.path = Path(path)

    @staticmethod
    def _decode_utf16(dataset) -> str:
        arr = np.asarray(dataset, dtype=np.uint16).ravel()
        return ''.join(chr(v) for v in arr if v != 0)

    @staticmethod
    def _read_array(dataset):
        arr = np.asarray(dataset)
        return np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))

    def load(self) -> Dict:
        methods: Dict[str, Dict] = {}
        with h5py.File(self.path, 'r') as f:
            results = f['results']
            dt = float(np.asarray(f['dt']).squeeze())
            for idx in range(results.shape[0]):
                group = f[results[idx, 0]]
                name = self._decode_utf16(group['name'])
                methods[name] = {
                    'tau_xy': ensure_time_last(self._read_array(group['tau_roi_steady_xy'])),
                    'tau_xz': ensure_time_last(self._read_array(group['tau_roi_steady_xz'])),
                    'tau_yz': ensure_time_last(self._read_array(group['tau_roi_steady_yz'])),
                    'tau_eq': ensure_time_last(self._read_array(group['tau_roi_steady'])),
                    'roi_x': np.asarray(group['roi_x_vec']).reshape(-1),
                    'roi_y': np.asarray(group['roi_y_vec']).reshape(-1),
                    't': np.asarray(group['t_vec_steady']).reshape(-1),
                }
        return {'dt': dt, 'methods': methods}


# =============================================================================
# Mechanics / neural helpers
# =============================================================================


def compute_dynamic_components(method_data: Dict) -> Dict[str, np.ndarray]:
    out = {}
    for key in ORTHO_COMPONENTS:
        tau = np.asarray(method_data[f'tau_{key}'], dtype=np.float64)
        out[key] = tau - tau.mean(axis=2, keepdims=True)
    return out


def build_receptor_lattice(roi_x: np.ndarray, roi_y: np.ndarray, spacing_m: float = RECEPTOR_SPACING) -> Dict:
    xs = np.arange(float(np.min(roi_x)), float(np.max(roi_x)) + spacing_m * 0.5, spacing_m)
    ys = np.arange(float(np.min(roi_y)), float(np.max(roi_y)) + spacing_m * 0.5, spacing_m)
    gx, gy = np.meshgrid(xs, ys, indexing='xy')
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    return {'coords_m': coords, 'x_m': xs, 'y_m': ys, 'shape': gx.shape}


class CoherentIntegrator:
    def __init__(
        self,
        roi_x: np.ndarray,
        roi_y: np.ndarray,
        receptor_coords: np.ndarray,
        conduction_velocity_m_s: float,
        spatial_decay_lambda_m: float,
        dt: float,
    ):
        self.roi_x = np.asarray(roi_x, dtype=np.float64)
        self.roi_y = np.asarray(roi_y, dtype=np.float64)
        self.receptor_coords = np.asarray(receptor_coords, dtype=np.float64)
        gx, gy = np.meshgrid(self.roi_x, self.roi_y, indexing='xy')
        self.source_coords = np.column_stack([gx.ravel(), gy.ravel()])
        self.distance_m = np.linalg.norm(self.receptor_coords[:, None, :] - self.source_coords[None, :, :], axis=-1)
        self.weight = np.exp(-self.distance_m / max(float(spatial_decay_lambda_m), 1e-12))
        self.delay_steps = np.rint(self.distance_m / max(float(conduction_velocity_m_s) * float(dt), 1e-12)).astype(np.int32)

    def integrate(self, tau_dyn: np.ndarray) -> np.ndarray:
        source_signal = np.reshape(np.asarray(tau_dyn, dtype=np.float64), (-1, tau_dyn.shape[-1]))
        n_receptors = self.receptor_coords.shape[0]
        n_time = source_signal.shape[1]
        integrated = np.zeros((n_receptors, n_time), dtype=np.float64)
        time_index = np.arange(n_time, dtype=np.int32)
        for ridx in range(n_receptors):
            acc = np.zeros(n_time, dtype=np.float64)
            for sidx in range(source_signal.shape[0]):
                shifted_idx = time_index - self.delay_steps[ridx, sidx]
                valid = shifted_idx >= 0
                clipped = np.clip(shifted_idx, 0, n_time - 1)
                acc += source_signal[sidx, clipped] * valid * self.weight[ridx, sidx]
            integrated[ridx] = acc
        return integrated


def apply_pacinian_filter(signal: np.ndarray, dt: float) -> np.ndarray:
    fs = 1.0 / dt
    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH], btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, signal, axis=-1)


def compute_single_sided_spectrum(signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    n = len(signal)
    win = get_window('hann', n)
    y = np.fft.rfft(signal * win, n=2 ** int(np.ceil(np.log2(n * 8))))
    f = np.fft.rfftfreq((len(y) - 1) * 2, d=dt)
    amp = np.abs(y) / n
    if amp.size > 2:
        amp[1:-1] *= 2
    return f, amp


def compute_vector_strength_from_spike_times(spike_times_s: np.ndarray, f0: float = CARRIER_FREQ) -> float:
    if spike_times_s.size == 0:
        return 0.0
    phases = np.exp(1j * 2.0 * np.pi * f0 * spike_times_s)
    return float(np.abs(np.mean(phases)))


<<<<<<< HEAD
def compute_vector_strength_from_spike_train(spike_train: np.ndarray, dt: float, f0: float = CARRIER_FREQ) -> float:
    spike_train = np.asarray(spike_train)
    if spike_train.size == 0:
        return 0.0
    spike_idx = np.flatnonzero(spike_train > 0)
    if spike_idx.size == 0:
        return 0.0
    spike_times_s = spike_idx.astype(np.float64) * float(dt)
    return compute_vector_strength_from_spike_times(spike_times_s, f0)


=======
>>>>>>> 1aab52592594dc4dde9dbed65fe7e1e4792c83ce
# =============================================================================
# Analysis data assembly
# =============================================================================


def load_model_summary() -> Dict:
    with (MODEL_RESULTS_DIR / 'summary.json').open('r', encoding='utf-8') as f:
        return json.load(f)


def load_experiment_scores() -> Dict:
    with (EXPERIMENT_ANALYSIS_DIR / 'Intensity_detailed_results.json').open('r', encoding='utf-8') as f:
        payload = json.load(f)
    return {
        'scores': payload['score_by_method'],
        'se': payload['se_by_method'],
        'pairwise_wald': payload['pairwise_wald'],
        'win_matrix_csv': payload['win_matrix_csv'],
    }


def load_population_outputs() -> Dict[str, Dict]:
    outputs = {}
    for method in METHOD_ORDER:
        data = np.load(MODEL_RESULTS_DIR / f'{method}_population_outputs.npz', allow_pickle=True)
        spikes = np.load(MODEL_RESULTS_DIR / f'{method}_spikes.npy', allow_pickle=True)
        outputs[method] = {
            'weights': data['weights'],
            'rates': data['rates'],
            'vector_strength': data['vector_strength'],
            'population_map': data['population_map'],
            'receptor_coords_m': data['receptor_coords_m'],
            'spikes': spikes,
        }
    return outputs


def load_all_data() -> Dict:
    loader = KWaveMatLoader(MAT_FILE_PATH)
    kwave = loader.load()
    summary = load_model_summary()
    experiment = load_experiment_scores()
    population = load_population_outputs()
    return {'kwave': kwave, 'summary': summary, 'experiment': experiment, 'population': population}


def choose_centerline_receptors(coords_m: np.ndarray, n_select: int = WATERFALL_RECEPTORS) -> np.ndarray:
    coords = np.asarray(coords_m, dtype=np.float64)
    y_abs = np.abs(coords[:, 1])
    center_y = np.min(y_abs)
    candidates = np.where(np.isclose(y_abs, center_y))[0]
    xs = coords[candidates, 0]
    order = np.argsort(xs)
    candidates = candidates[order]
    if len(candidates) <= n_select:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, n_select).round().astype(int)
    return candidates[positions]


def choose_central_neurons(coords_m: np.ndarray, n_select: int = N_RASTER_NEURONS) -> np.ndarray:
    coords = np.asarray(coords_m, dtype=np.float64)
    radius = np.linalg.norm(coords, axis=1)
    order = np.argsort(radius)
    return order[:n_select]


def build_pairwise_matrix(methods: List[str], pairwise_items: List[Dict]) -> np.ndarray:
    idx = {m: i for i, m in enumerate(methods)}
    mat = np.full((len(methods), len(methods)), np.nan, dtype=np.float64)
    np.fill_diagonal(mat, 0.5)
    for item in pairwise_items:
        a = item['A']
        b = item['B']
        p = float(item['preference_index'])
        i, j = idx[a], idx[b]
        mat[i, j] = p
        mat[j, i] = 1.0 - p
    return mat


def build_experiment_win_fraction_matrix(methods: List[str], csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, index_col=0)
    df = df.reindex(index=methods, columns=methods)
    wins = df.to_numpy(dtype=np.float64)
    total = wins + wins.T
    out = np.full_like(wins, np.nan, dtype=np.float64)
    for i in range(len(methods)):
        out[i, i] = 0.5
        for j in range(len(methods)):
            if i == j:
                continue
            if total[i, j] > 0:
                out[i, j] = wins[i, j] / total[i, j]
    return out


# =============================================================================
# Figure 1
# =============================================================================
def plot_figure1(data: Dict) -> None:
    kwave = data['kwave']['methods']
    fig = plt.figure(figsize=(15, 8))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.18)

    global_max_tau = 0.0
    global_max_u = 0.0  # 改为追踪滤波后的 u(t)
    plot_data_cache = {}

    for method in PRIMARY_COMPARE:
        method_data = kwave[method]
        dyn = compute_dynamic_components(method_data)['xy'] 
        lattice = build_receptor_lattice(method_data['roi_x'], method_data['roi_y'])
        receptor_idx = choose_centerline_receptors(lattice['coords_m'], WATERFALL_RECEPTORS)
        
        integrator = CoherentIntegrator(method_data['roi_x'], method_data['roi_y'], 
                                        lattice['coords_m'][receptor_idx], 
                                        SHEAR_SPEED, LAMBDA_SPACE, data['kwave']['dt'])
        m_drive = integrator.integrate(dyn)
        
        # 【核心修改】：在这里直接应用 Pacinian 滤波器
        u_drive = apply_pacinian_filter(m_drive, data['kwave']['dt'])

        y_idx = int(np.argmin(np.abs(method_data['roi_y'])))
        x_indices = [int(np.argmin(np.abs(method_data['roi_x'] - x0))) for x0 in lattice['coords_m'][receptor_idx, 0]]
        tau_traces = np.stack([dyn[y_idx, xi, :] for xi in x_indices], axis=0)

        t_ms = method_data['t'] * 1000.0
        window_mask = t_ms >= (t_ms.max() - RASTER_DURATION_MS)
        
        tau_win = tau_traces[:, window_mask]
        u_win = u_drive[:, window_mask]  # 提取窗口内的 u(t)
        
        global_max_tau = max(global_max_tau, np.max(np.abs(tau_win)))
        global_max_u = max(global_max_u, np.max(np.abs(u_win)))
        
        plot_data_cache[method] = (t_ms[window_mask], tau_win, u_win)

    offset_tau = global_max_tau * 1.5 + 1e-6
    offset_u = global_max_u * 1.2 + 1e-6

    for col, method in enumerate(PRIMARY_COMPARE):
        t_win, tau_win, u_win = plot_data_cache[method]
        
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[col], height_ratios=[2.0, 1.0], hspace=0.15)
        ax_top = fig.add_subplot(inner[0])
        ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)

        for ridx in range(tau_win.shape[0]):
            base = ridx * offset_tau
            smooth = gaussian_filter1d(tau_win[ridx], sigma=1.0)
            ax_top.fill_between(t_win, base, base + np.clip(smooth, 0, None), color='#b40426', alpha=0.45, linewidth=0)
            ax_top.fill_between(t_win, base, base + np.clip(smooth, None, 0), color='#3b4cc0', alpha=0.45, linewidth=0)
            ax_top.plot(t_win, base + smooth, color='0.35', lw=0.9, alpha=0.9)

        for ridx in range(u_win.shape[0]):
            ax_bottom.plot(t_win, u_win[ridx] + ridx * offset_u, color='black', lw=1.6, alpha=0.9)

        # 计算对受体的“有效驱动增益”
        method_max_tau = np.max(np.abs(tau_win))
        method_max_u = np.max(np.abs(u_win))
        gain = method_max_u / max(method_max_tau, 1e-12)

        ax_top.set_title(f'{method} | Raw shear wavefronts', fontweight='bold')
        ax_bottom.set_title(f'{method} | Effective neural drive u(t)', fontweight='bold', pad=6) # 标题修改
        ax_top.text(0.02, 0.95, f'Effective Gain ≈ {gain:.1f}×', transform=ax_top.transAxes, 
                    ha='left', va='top', fontsize=LEGEND_SIZE, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
        
        ax_top.set_ylabel('Virtual receptors', fontweight='bold')
        ax_bottom.set_ylabel('Filtered traces', fontweight='bold') # Y轴标签修改
        ax_bottom.set_xlabel('Time [ms]', fontweight='bold')
        
        ax_top.set_ylim(-offset_tau, tau_win.shape[0] * offset_tau)
        ax_bottom.set_ylim(-offset_u, u_win.shape[0] * offset_u)
        
        ax_top.set_yticks(np.arange(tau_win.shape[0]) * offset_tau)
        ax_top.set_yticklabels([str(i + 1) for i in range(tau_win.shape[0])])
        ax_bottom.set_yticks([])
        ax_top.grid(False)
        ax_bottom.grid(True, linestyle='--', alpha=0.35)
        ax_top.tick_params(labelbottom=False)

    # fig.suptitle('Figure 1 | Spatiotemporal Coherent Integration Dynamics', fontweight='bold', y=0.995)
    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_1_Coherent_Integration_Dynamics')


# =============================================================================
# Figure 2
# =============================================================================

def plot_figure2(data: Dict) -> None:
    kwave = data['kwave']['methods']
    fig, ax = plt.subplots(figsize=(12,8))
    ax.axvspan(BANDPASS_LOW, BANDPASS_HIGH, color='0.85', alpha=0.7, zorder=0)
    ax.text((BANDPASS_LOW + BANDPASS_HIGH) / 2.0, -180, 'Pacinian band-pass', 
            ha='center', va='bottom', fontsize=LEGEND_SIZE, fontweight='bold')

    for method in PRIMARY_COMPARE:
        method_data = kwave[method]
        lattice = build_receptor_lattice(method_data['roi_x'], method_data['roi_y'])
        
        # 修复 1: 精准找到几何中心点 (x=0, y=0)
        distances = np.linalg.norm(lattice['coords_m'], axis=1)
        center_idx = [np.argmin(distances)]
        
        dyn_all = compute_dynamic_components(method_data)
        integrator = CoherentIntegrator(method_data['roi_x'], method_data['roi_y'], 
                                        lattice['coords_m'][center_idx], 
                                        SHEAR_SPEED, LAMBDA_SPACE, data['kwave']['dt'])
        
        # 修复 3: 提取中心点三个分量中响应最强的一个
        best_m_drive = None
        max_energy = -1
        for comp in ORTHO_COMPONENTS:
            m_comp = integrator.integrate(dyn_all[comp])[0]
            if np.max(np.abs(m_comp)) > max_energy:
                max_energy = np.max(np.abs(m_comp))
                best_m_drive = m_comp

        u_drive = apply_pacinian_filter(best_m_drive[None, :], data['kwave']['dt'])[0]
        freqs_m, spec_m = compute_single_sided_spectrum(best_m_drive, data['kwave']['dt'])
        freqs_u, spec_u = compute_single_sided_spectrum(u_drive, data['kwave']['dt'])
        
        spec_m_db = 20.0 * np.log10(spec_m + 1e-12)
        spec_u_db = 20.0 * np.log10(spec_u + 1e-12)
        color = SPECTRUM_COLORS[method]
        
        ax.plot(freqs_m, spec_m_db, color=color, lw=LINE_WIDTH, label=f'{method} | m(t)', zorder=100)
        ax.plot(freqs_u, spec_u_db, color=color, lw=2.2, ls='--', alpha=0.8, label=f'{method} | u(t)', zorder=50)

    ax.set_xlim(0, FREQ_MAX)
    ax.set_ylim(-200, 80) # 建议固定Y轴范围，避免数据抖动导致留白过大
    ax.set_xlabel('Frequency [Hz]', fontweight='bold')
    ax.set_ylabel('PSD [dB]', fontweight='bold')
    # ax.set_title('Figure 2 | Frequency Fidelity & Receptor Tuning', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(frameon=True, ncol=1, loc='lower right')
    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_2_Frequency_Fidelity')

<<<<<<< HEAD
def plot_figure3(data: Dict) -> None:
    """
    Figure 3 | Phase-folded centerline effective neural-drive magnitude.

    Final repaired version.

    What is plotted:
        E(x,t) = sqrt(mean_c( u_c(x,t)^2 ))
    where u_c is the signed Pacinian-bandpassed coherent drive for each shear component.

    Why this fixes the previous figure:
    - removes the artificial left-right bias introduced by half-wave rectification
    - removes unstable cross-component max-pooling
    - respects rotational methods (DLM_3, LM_C)
    - phase-folds the final 15 ms into one 5 ms modulation cycle, improving robustness
    - directly visualizes within-cycle spatiotemporal recruitment structure
    """

    print("\n" + "=" * 110)
    print("[DEBUG][FIG3-FINAL] START | Figure 3 final repaired version")
    print("=" * 110)

    kwave = data['kwave']['methods']
    population = data['population']
    dt = float(data['kwave']['dt'])
    dt_ms = dt * 1000.0

    cycle_ms = 1000.0 / CARRIER_FREQ   # 200 Hz -> 5 ms
    n_phase_bins = 240                 # robust and smooth enough
    print(f"[DEBUG][FIG3-FINAL] dt = {dt:.9f} s ({dt_ms:.6f} ms)")
    print(f"[DEBUG][FIG3-FINAL] cycle_ms = {cycle_ms:.6f} ms")
    print(f"[DEBUG][FIG3-FINAL] phase bins = {n_phase_bins}")
    print(f"[DEBUG][FIG3-FINAL] window = last {RASTER_DURATION_MS:.3f} ms")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _safe_stats(arr: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return {
                'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan,
                'p01': np.nan, 'p50': np.nan, 'p99': np.nan
            }
        return {
            'min': float(np.nanmin(arr)),
            'max': float(np.nanmax(arr)),
            'mean': float(np.nanmean(arr)),
            'std': float(np.nanstd(arr)),
            'p01': float(np.nanpercentile(arr, 1)),
            'p50': float(np.nanpercentile(arr, 50)),
            'p99': float(np.nanpercentile(arr, 99)),
        }

    def _print_stats(tag: str, arr: np.ndarray) -> None:
        s = _safe_stats(arr)
        print(
            f"{tag} | "
            f"min={s['min']:.6g}, max={s['max']:.6g}, mean={s['mean']:.6g}, std={s['std']:.6g}, "
            f"p01={s['p01']:.6g}, p50={s['p50']:.6g}, p99={s['p99']:.6g}"
        )

    def _centers_to_edges(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.size == 1:
            dv = 1.0
            return np.array([v[0] - 0.5 * dv, v[0] + 0.5 * dv], dtype=np.float64)
        mids = 0.5 * (v[:-1] + v[1:])
        first = v[0] - 0.5 * (v[1] - v[0])
        last = v[-1] + 0.5 * (v[-1] - v[-2])
        return np.concatenate([[first], mids, [last]])

    def _weighted_centroid_mm(x_mm: np.ndarray, w: np.ndarray) -> float:
        x_mm = np.asarray(x_mm, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        denom = float(np.nansum(w))
        if denom <= 0:
            return np.nan
        return float(np.nansum(x_mm * w) / denom)

    def _left_right_asymmetry(x_mm: np.ndarray, w: np.ndarray) -> Tuple[float, float, float]:
        x_mm = np.asarray(x_mm, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        left = float(np.nansum(w[x_mm < 0]))
        right = float(np.nansum(w[x_mm > 0]))
        denom = left + right
        asym = (right - left) / denom if denom > 0 else np.nan
        return left, right, float(asym)

    def _choose_symmetric_centerline_indices(
        coords_m: np.ndarray,
        y_round_decimals: int = 6,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Select receptors from the row nearest y=0, then enforce strict left-right
        symmetry around x=0 by pairing receptors by radius.
        """
        coords = np.asarray(coords_m, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"receptor_coords_m must have shape (N, 2), got {coords.shape}")

        y_round = np.round(coords[:, 1], y_round_decimals)
        unique_y = np.unique(y_round)
        center_y = unique_y[np.argmin(np.abs(unique_y))]

        row_idx = np.where(np.isclose(y_round, center_y, atol=10 ** (-y_round_decimals)))[0]
        row_idx = row_idx[np.argsort(coords[row_idx, 0])]
        x_row_mm = coords[row_idx, 0] * 1000.0

        if row_idx.size < 3:
            raise RuntimeError(f"Not enough receptors on centerline row: {row_idx.size}")

        dx_mm = np.diff(x_row_mm)
        median_dx_mm = float(np.median(np.abs(dx_mm))) if dx_mm.size > 0 else 2.0
        zero_tol_mm = 0.55 * median_dx_mm

        left_local = np.where(x_row_mm < -0.5 * zero_tol_mm)[0]
        right_local = np.where(x_row_mm > 0.5 * zero_tol_mm)[0]
        center_local = int(np.argmin(np.abs(x_row_mm)))
        has_center = abs(float(x_row_mm[center_local])) <= zero_tol_mm

        left_near = left_local[np.argsort(np.abs(x_row_mm[left_local]))]
        right_near = right_local[np.argsort(np.abs(x_row_mm[right_local]))]
        n_pairs = int(min(len(left_near), len(right_near)))

        if n_pairs <= 0:
            raise RuntimeError("No symmetric receptor pairs found on centerline.")

        left_keep = left_near[:n_pairs]
        right_keep = right_near[:n_pairs]

        chosen_local = list(left_keep)
        if has_center:
            chosen_local.append(center_local)
        chosen_local.extend(list(right_keep))
        chosen_local = np.array(sorted(set(chosen_local), key=lambda k: x_row_mm[k]), dtype=int)

        selected_idx = row_idx[chosen_local]
        x_sel_mm = coords[selected_idx, 0] * 1000.0

        diag = {
            'center_y_mm': float(center_y * 1000.0),
            'row_count': int(row_idx.size),
            'selected_count': int(selected_idx.size),
            'has_center': bool(has_center),
            'nearest_center_x_mm': float(x_row_mm[center_local]),
            'selected_x_mm': x_sel_mm.copy(),
            'median_dx_mm': float(median_dx_mm),
        }
        return selected_idx, diag

    def _phase_fold_map(signal_xt: np.ndarray, t_rel_ms: np.ndarray, period_ms: float, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phase-fold signal_xt over one modulation cycle.
        signal_xt shape: (Nx, Nt)
        returns:
            folded_map shape: (Nx, n_bins)
            phase_centers_ms shape: (n_bins,)
        """
        signal_xt = np.asarray(signal_xt, dtype=np.float64)
        t_rel_ms = np.asarray(t_rel_ms, dtype=np.float64)

        phase_edges = np.linspace(0.0, period_ms, n_bins + 1)
        phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])

        phase = np.mod(t_rel_ms, period_ms)
        # Put phase==period exactly into last bin safely
        bin_idx = np.floor(phase / period_ms * n_bins).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        folded = np.full((signal_xt.shape[0], n_bins), np.nan, dtype=np.float64)
        counts = np.bincount(bin_idx, minlength=n_bins)

        print(f"[DEBUG][FIG3-FINAL] phase-fold counts | min={counts.min()}, max={counts.max()}, mean={counts.mean():.3f}")

        for b in range(n_bins):
            mask = bin_idx == b
            if np.any(mask):
                folded[:, b] = np.mean(signal_xt[:, mask], axis=1)

        # Fill occasional empty bins by linear interpolation along phase
        if np.any(~np.isfinite(folded)):
            print("[WARN ][FIG3-FINAL] Empty phase bins detected; applying linear interpolation along phase.")
            for i in range(folded.shape[0]):
                row = folded[i]
                good = np.isfinite(row)
                if np.sum(good) >= 2:
                    folded[i] = np.interp(np.arange(n_bins), np.where(good)[0], row[good])
                elif np.sum(good) == 1:
                    folded[i] = row[good][0]
                else:
                    folded[i] = 0.0

        return folded, phase_centers

    # -------------------------------------------------------------------------
    # Precompute per method
    # -------------------------------------------------------------------------
    cache = {}
    all_values = []
    all_profiles = []
    global_abs_x_mm = 0.0

    for method in METHOD_ORDER:
        print("\n" + "-" * 110)
        print(f"[DEBUG][FIG3-FINAL][{method}] START")
        print("-" * 110)

        method_data = kwave[method]
        coords = np.asarray(population[method]['receptor_coords_m'], dtype=np.float64)

        selected_idx, diag = _choose_symmetric_centerline_indices(coords)
        line_coords = coords[selected_idx]
        x_mm = line_coords[:, 0] * 1000.0
        global_abs_x_mm = max(global_abs_x_mm, float(np.max(np.abs(x_mm))))

        print(f"[DEBUG][FIG3-FINAL][{method}] centerline y ~= {diag['center_y_mm']:.3f} mm")
        print(f"[DEBUG][FIG3-FINAL][{method}] selected count = {diag['selected_count']}")
        print(f"[DEBUG][FIG3-FINAL][{method}] nearest center x = {diag['nearest_center_x_mm']:.3f} mm")
        print(f"[DEBUG][FIG3-FINAL][{method}] selected x_mm = {np.array2string(diag['selected_x_mm'], precision=2, separator=', ')}")

        # Time window: last 15 ms, but exclude duplicated endpoint at exactly t_end
        t_ms_full = np.asarray(method_data['t'], dtype=np.float64) * 1000.0
        t_end = float(t_ms_full.max())
        window_mask = (t_ms_full >= (t_end - RASTER_DURATION_MS - 0.5 * dt_ms)) & (t_ms_full < (t_end - 0.5 * dt_ms))
        t_win_ms = t_ms_full[window_mask]
        t_rel_ms = t_win_ms - t_win_ms[0]

        print(f"[DEBUG][FIG3-FINAL][{method}] window samples = {t_rel_ms.size}")
        print(f"[DEBUG][FIG3-FINAL][{method}] window range = [{t_win_ms.min():.6f}, {t_win_ms.max():.6f}] ms")
        print(f"[DEBUG][FIG3-FINAL][{method}] relative range = [{t_rel_ms.min():.6f}, {t_rel_ms.max():.6f}] ms")

        dyn = compute_dynamic_components(method_data)

        integrator = CoherentIntegrator(
            method_data['roi_x'],
            method_data['roi_y'],
            line_coords,
            SHEAR_SPEED,
            LAMBDA_SPACE,
            dt,
        )

        # Signed, band-passed component drives
        spike_array = np.asarray(population[method]['spikes'])
        if spike_array.ndim != 3 or spike_array.shape[0] != len(ORTHO_COMPONENTS):
            raise RuntimeError(
                f"Unexpected spike array shape for {method}: {spike_array.shape}; "
                f"expected ({len(ORTHO_COMPONENTS)}, N_receptors, N_time)"
            )
        if spike_array.shape[1] != coords.shape[0]:
            raise RuntimeError(
                f"Spike receptor count mismatch for {method}: "
                f"expected {coords.shape[0]}, got {spike_array.shape[1]}"
            )

        component_weight_maps = []
        component_profiles = []
        for comp_idx, comp in enumerate(ORTHO_COMPONENTS):
            m_comp = integrator.integrate(dyn[comp])
            u_comp = apply_pacinian_filter(m_comp, dt)
            u_win = u_comp[:, window_mask]
            r_comp = np.maximum(u_win, 0.0)

            spike_comp = spike_array[comp_idx]
            vs_all = np.array(
                [compute_vector_strength_from_spike_train(spike_comp[idx], dt, CARRIER_FREQ) for idx in range(spike_comp.shape[0])],
                dtype=np.float64,
            )
            vs_sel = vs_all[selected_idx]
            weight_map = r_comp * vs_sel[:, None]
            weight_profile = np.max(weight_map, axis=1)

            component_weight_maps.append(weight_map)
            component_profiles.append(weight_profile)

            _print_stats(f"[DEBUG][FIG3-FINAL][{method}][{comp}] u_comp win", u_win)
            _print_stats(f"[DEBUG][FIG3-FINAL][{method}][{comp}] vs_sel", vs_sel)
            _print_stats(f"[DEBUG][FIG3-FINAL][{method}][{comp}] weight_map=r*VS", weight_map)
            _print_stats(f"[DEBUG][FIG3-FINAL][{method}][{comp}] weight_profile=max_t(r*VS)", weight_profile)

        component_weight_maps = np.stack(component_weight_maps, axis=0)  # (3, Nx, Nt_win)
        component_profiles = np.stack(component_profiles, axis=0)        # (3, Nx)

        final_weight_map = np.maximum.reduce(component_weight_maps, axis=0)  # (Nx, Nt_win)
        final_profile = np.max(component_profiles, axis=0)                   # (Nx,)

        _print_stats(f"[DEBUG][FIG3-FINAL][{method}] final_weight_map win", final_weight_map)
        _print_stats(f"[DEBUG][FIG3-FINAL][{method}] final_profile w_i", final_profile)

        # Fold 15 ms -> 1 cycle (5 ms)
        folded_map, phase_centers_ms = _phase_fold_map(
            signal_xt=final_weight_map,
            t_rel_ms=t_rel_ms,
            period_ms=cycle_ms,
            n_bins=n_phase_bins,
        )

        centroid_mm = _weighted_centroid_mm(x_mm, final_profile)
        left_sum, right_sum, asym = _left_right_asymmetry(x_mm, final_profile)

        print(f"[DEBUG][FIG3-FINAL][{method}] folded_map shape = {folded_map.shape}")
        _print_stats(f"[DEBUG][FIG3-FINAL][{method}] folded_map", folded_map)
        _print_stats(f"[DEBUG][FIG3-FINAL][{method}] final_profile", final_profile)
        print(
            f"[DEBUG][FIG3-FINAL][{method}] final_profile centroid = {centroid_mm:.6f} mm | "
            f"left={left_sum:.6f}, right={right_sum:.6f}, asym={asym:.6f}"
        )

        # Optional weak sanity check against stored weights
        weights_all = np.asarray(population[method]['weights'], dtype=np.float64).reshape(-1)
        if weights_all.size == coords.shape[0]:
            weights_sel = weights_all[selected_idx]
            if np.std(final_profile) > 0 and np.std(weights_sel) > 0:
                try:
                    r, p = stats.pearsonr(final_profile, weights_sel)
                    print(f"[DEBUG][FIG3-FINAL][{method}] corr(final_profile, stored_weights_sel) = r={r:.6f}, p={p:.6g}")
                except Exception as e:
                    print(f"[WARN ][FIG3-FINAL][{method}] correlation failed: {repr(e)}")

        cache[method] = {
            'x_mm': x_mm,
            'phase_centers_ms': phase_centers_ms,
            'folded_map': folded_map,
            'final_profile': final_profile,
        }

        all_values.append(folded_map.ravel())
        all_profiles.append(final_profile.ravel())

    # -------------------------------------------------------------------------
    # Global scale
    # -------------------------------------------------------------------------
    all_values = np.concatenate(all_values)
    all_profiles = np.concatenate(all_profiles)

    vmax = float(np.nanpercentile(all_values, 99.5))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(all_values))
    if vmax <= 0:
        vmax = 1.0

    profile_max = float(np.nanpercentile(all_profiles, 99.5))
    if not np.isfinite(profile_max) or profile_max <= 0:
        profile_max = float(np.nanmax(all_profiles))
    if profile_max <= 0:
        profile_max = 1.0

    print("\n" + "=" * 110)
    print("[DEBUG][FIG3-FINAL] GLOBAL SUMMARY")
    print("=" * 110)
    _print_stats("[DEBUG][FIG3-FINAL] all folded values", all_values)
    _print_stats("[DEBUG][FIG3-FINAL] all mean profiles", all_profiles)
    print(f"[DEBUG][FIG3-FINAL] global_abs_x_mm = {global_abs_x_mm:.6f}")
    print(f"[DEBUG][FIG3-FINAL] heatmap vmax = {vmax:.6f}")
    print(f"[DEBUG][FIG3-FINAL] profile_max = {profile_max:.6f}")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12.8, 14.0))
    gs = gridspec.GridSpec(
        len(METHOD_ORDER),
        3,
        width_ratios=[6.8, 2.0, 0.18],
        hspace=0.10,
        wspace=0.12,
    )

    heat_axes = []
    profile_axes = []
    mappable = None

    for row, method in enumerate(METHOD_ORDER):
        ax_h = fig.add_subplot(
            gs[row, 0],
            sharex=heat_axes[0] if len(heat_axes) > 0 else None,
            sharey=heat_axes[0] if len(heat_axes) > 0 else None,
        )
        ax_p = fig.add_subplot(
            gs[row, 1],
            sharex=profile_axes[0] if len(profile_axes) > 0 else None,
            sharey=ax_h,
        )

        heat_axes.append(ax_h)
        profile_axes.append(ax_p)

        x_mm = cache[method]['x_mm']
        folded_map = cache[method]['folded_map']
        phase_centers_ms = cache[method]['phase_centers_ms']
        final_profile = cache[method]['final_profile']

        x_edges = _centers_to_edges(x_mm)
        phase_edges = np.linspace(0.0, cycle_ms, len(phase_centers_ms) + 1)

        mappable = ax_h.pcolormesh(
            phase_edges,
            x_edges,
            folded_map,
            shading='auto',
            cmap='magma',
            vmin=0.0,
            vmax=vmax,
        )

        ax_h.axhline(0.0, color='white', linestyle='--', lw=1.0, alpha=0.55)
        ax_h.set_ylim(-global_abs_x_mm, global_abs_x_mm)
        ax_h.grid(False)

        ax_h.text(
            0.015, 0.92, method,
            transform=ax_h.transAxes,
            ha='left', va='top',
            color='white',
            fontsize=LEGEND_SIZE,
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.20, edgecolor='none', pad=2.5)
        )

        if row < len(METHOD_ORDER) - 1:
            ax_h.tick_params(labelbottom=False)
        else:
            ax_h.set_xlabel('Phase within one 200 Hz cycle [ms]', fontweight='bold')

        ax_p.plot(final_profile, x_mm, color=METHOD_COLORS[method], lw=LINE_WIDTH)
        ax_p.fill_betweenx(x_mm, 0.0, final_profile, color=METHOD_COLORS[method], alpha=0.22)
        ax_p.axhline(0.0, color='0.35', linestyle='--', lw=0.9, alpha=0.60)
        ax_p.set_xlim(0.0, profile_max * 1.05)
        ax_p.grid(True, axis='x', linestyle='--', alpha=0.35)
        ax_p.tick_params(labelleft=False)

        if row < len(METHOD_ORDER) - 1:
            ax_p.tick_params(labelbottom=False)
        else:
            ax_p.set_xlabel('Final readout weight $w_i = \max(w_{xy}, w_{xz}, w_{yz})$ [a.u.]', fontweight='bold')

        if row == 0:
            ax_h.set_title(
                r'Phase-folded centerline weight map $r(x,\phi) \times VS_{200\,\mathrm{Hz}}$',
                fontweight='bold',
                pad=10,
            )
            ax_p.set_title(
                'Final readout\n'+r'$w_i = \max(w_{xy}, w_{xz}, w_{yz})$',
                fontweight='bold',
                pad=10,
            )

    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r'Phase-folded weight $r \times VS_{200\,\mathrm{Hz}}$ [a.u.]', fontweight='bold')
    cbar.outline.set_linewidth(1.0)

    fig.text(
        0.02, 0.5,
        'Centerline receptor position x [mm]',
        rotation=90,
        va='center',
        ha='center',
        fontweight='bold'
    )

    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_3_PhaseFolded_Centerline_DriveMagnitude')

    print("\n" + "=" * 110)
    print("[DEBUG][FIG3-FINAL] END | Figure 3 saved successfully")
    print("=" * 110 + "\n")
=======


def plot_figure3(data: Dict) -> None:
    print("\n" + "="*85)
    print("[DEBUG] STARTING PLOT_FIGURE_3 (3D POOLED FIRST-PRINCIPLE DRIVEN)")
    print("="*85)
    
    population = data['population']
    fig = plt.figure(figsize=(13, 14))
    outer = gridspec.GridSpec(3, 2, width_ratios=[2.2, 1.0], wspace=0.15, hspace=0.4)

    RASTER_COLORS = {
        'ULM_L': '#440154',
        'DLM_2': '#3b528b',
        'LM_L': '#b89500'  
    }

    for row, method in enumerate(RASTER_METHODS):
        print(f"\n---> [DEBUG] ================= Analyzing Method: {method} =================")
        method_pop = population[method]
        coords = method_pop['receptor_coords_m']
        
        # 1. 提取中心水平切线
        center_y = coords[np.argmin(np.abs(coords[:, 1])), 1]
        line_idx = np.where(np.isclose(coords[:, 1], center_y, atol=1e-6))[0]
        sorted_line = line_idx[np.argsort(coords[line_idx, 0])]
        
        if len(sorted_line) > N_RASTER_NEURONS:
            start = (len(sorted_line) - N_RASTER_NEURONS) // 2
            selected = sorted_line[start:start+N_RASTER_NEURONS]
        else:
            selected = sorted_line
            
        x_mm = coords[selected, 0] * 1000.0  
        print(f"[DEBUG - SPACE] Baseline Y={center_y*1000:.3f} mm. X bounds: [{x_mm.min():.2f}, {x_mm.max():.2f}] mm")
        
        # 2. RASTER PLOT DATA (Original Spike Pooling Logic)
        all_spikes = method_pop['spikes'][:, selected, :]  # Shape: (3, N_neurons, N_time)
        spikes_pooled = np.any(all_spikes, axis=0)         # Shape: (N_neurons, N_time)
        print(f"[DEBUG - PHYSICS] Pooled spikes across xy, xz, yz components. Shape: {spikes_pooled.shape}")

        # 3. 时间轴对齐
        t_vec_full = np.asarray(data['kwave']['methods'][method]['t'], dtype=np.float64)
        n_time = spikes_pooled.shape[1]
        t_trim_sec = t_vec_full[-n_time:]  
        t_trim_ms = t_trim_sec * 1000.0
        
        T_end_ms = float(t_trim_ms[-1])
        T_start_ms = T_end_ms - RASTER_DURATION_MS

        # ==========================================
        # 子图 1：XT-Spacetime Raster Plot
        # ==========================================
        ax_raster = fig.add_subplot(outer[row, 0])
        y_range = float(x_mm.max() - x_mm.min())
        dy = y_range / max(1, len(x_mm) - 1)
        
        total_raw_spikes_in_win = 0
            
        for n_idx in range(len(selected)):
            spike_idx = np.where(spikes_pooled[n_idx])[0]
            if len(spike_idx) == 0: continue
                
            spike_times_ms_full = t_trim_ms[spike_idx]
            
            # --- 核心科学修正：单神经轴突全局绝对不应期 (2.0 ms) ---
            axon_filtered_spikes = []
            last_t = -999.0
            for t in spike_times_ms_full:
                if t - last_t > 2.0:  # 严格锁定 2.0 ms
                    axon_filtered_spikes.append(t)
                    last_t = t
            spike_times_ms_full = np.array(axon_filtered_spikes)
            # --------------------------------------------------------
            valid_raw = (spike_times_ms_full >= T_start_ms) & (spike_times_ms_full <= T_end_ms)
            spike_times_win = spike_times_ms_full[valid_raw] - T_start_ms
            
            total_raw_spikes_in_win += np.sum(valid_raw)
            
            if len(spike_times_win) > 0:
                y_pos = float(x_mm[n_idx])
                ax_raster.vlines(x=spike_times_win, ymin=y_pos - dy * 0.45, ymax=y_pos + dy * 0.45, 
                                 color=RASTER_COLORS[method], lw=1.5, alpha=0.9, zorder=3)
                
        print(f"[DEBUG - RASTER] Pooled spikes plotted in window -> RAW: {total_raw_spikes_in_win}")
                
        ax_raster.set_xlim(0, float(RASTER_DURATION_MS))
        view_limit = 12.0
        ax_raster.set_ylim(-view_limit, view_limit)
        ax_raster.set_ylabel('Position X [mm]', fontweight='bold')
        
        if row == len(RASTER_METHODS) - 1:
            ax_raster.set_xlabel('Time [ms]', fontweight='bold')
        else:
            ax_raster.tick_params(labelbottom=False)
            
        ax_raster.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax_raster.set_title(f'{method} | XT spike raster', fontweight='bold', loc='left', pad=15)

        # ==========================================
        # 子图 2：局部感受野受体级频率保真度 (Delay-compensated Group Event VS)
        # ==========================================
        ax_polar = fig.add_subplot(outer[row, 1], projection='polar')
        
        # --- Compute pooled analog drive for polar plot ---
        method_data = data['kwave']['methods'][method]
        dyn_all = compute_dynamic_components(method_data)
        
        integrator = CoherentIntegrator(
            method_data['roi_x'], 
            method_data['roi_y'], 
            coords[selected], 
            SHEAR_SPEED, 
            LAMBDA_SPACE, 
            data['kwave']['dt']
        )
        
        u_components = []
        for comp in ORTHO_COMPONENTS:
            m_comp = integrator.integrate(dyn_all[comp])
            u_comp = apply_pacinian_filter(m_comp, data['kwave']['dt'])
            u_components.append(np.maximum(u_comp, 0.0))
        
        u_pool = np.maximum.reduce(u_components)   # shape: (N_selected, Nt)

        # Extract events
        dt = data['kwave']['dt']
        min_dist = max(1, int(round(0.002 / dt)))
        
        # Align u_pool time vector
        t_ms_full = method_data['t'] * 1000.0
        window_mask = (t_ms_full >= T_start_ms) & (t_ms_full <= T_end_ms)
        
        event_times_by_neuron = []
        for i in range(u_pool.shape[0]):
            trace = u_pool[i]
            win_trace = trace[window_mask]
            
            if len(win_trace) == 0:
                event_times_by_neuron.append(np.array([]))
                continue

            peaks, _ = find_peaks(
                win_trace, 
                distance=min_dist, 
                prominence=0.5 * np.std(win_trace)
            )
            
            if len(peaks) > 0:
                event_times_ms = t_ms_full[window_mask][peaks] - T_start_ms
                event_times_by_neuron.append(event_times_ms)
            else:
                event_times_by_neuron.append(np.array([]))
        
        ROI_RADIUS = 6.0 
        center_mask = np.abs(x_mm) <= ROI_RADIUS
        center_indices = np.where(center_mask)[0]
        
        absolute_phases = []
        
        for local_i in center_indices:
            t_ev_ms_rel = event_times_by_neuron[local_i]
            if len(t_ev_ms_rel) == 0:
                continue
                
            # Convert to seconds relative to start of window + start of window offset
            t_ev_sec = t_ev_ms_rel / 1000.0 + T_start_ms / 1000.0
            
            x_i_m = coords[selected[local_i], 0]
            
            # Delay compensation: t' = t - x/c
            t_comp = t_ev_sec - x_i_m / SHEAR_SPEED
            
            ph = (2.0 * np.pi * CARRIER_FREQ * t_comp) % (2.0 * np.pi)
            absolute_phases.extend(ph)

        # Calculate Group VS
        if len(absolute_phases) > 0:
            complex_ph = np.exp(1j * np.array(absolute_phases))
            local_vs = float(np.abs(np.mean(complex_ph)))
            mean_angle = np.angle(np.mean(complex_ph))
        else:
            local_vs = 0.0
            mean_angle = 0.0
            
        # Plot Histogram (aligned to mean angle = 0 for visualization)
        phases_array = np.array(absolute_phases)
        if len(phases_array) > 0:
            phases_plot = (phases_array - mean_angle) % (2.0 * np.pi)
        else:
            phases_plot = np.array([])
            
        n_bins = 24
        bin_width = 2.0 * np.pi / n_bins

        if len(phases_plot) > 0:
            bin_indices = np.floor(phases_plot / bin_width).astype(int) % n_bins
            counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
            if counts.max() > 0:
                counts = counts / counts.max()

            bin_centers = np.arange(n_bins) * bin_width + bin_width / 2.0
            ax_polar.bar(bin_centers, counts, width=bin_width, bottom=0.0, align='center',
                         color=METHOD_COLORS.get(method, '#333333'), alpha=0.85, edgecolor='white', linewidth=0.5, zorder=5)
        
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        
        # Annotate VS
        if local_vs > 0.01:
            ax_polar.annotate("",
                xy=(0, local_vs), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", facecolor='black', edgecolor='black', lw=3.0, mutation_scale=25), zorder=10
            )

        ax_polar.set_ylim(0, 1.0)
        ax_polar.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_polar.set_yticklabels(['', '0.5', '', '1.0'], fontsize=12, color='0.4')
        ax_polar.set_xticks(np.arange(0, 2.0 * np.pi, np.pi / 4.0))
        # Labels relative to aligned mean
        ax_polar.set_xticklabels(['0°', '+45°', '+90°', '+135°', '±180°', '-135°', '-90°', '-45°'], fontsize=14, zorder=1000)
        ax_polar.tick_params(axis='x', pad=-35, size=14)
        
        ax_polar.set_title(f'Central ROI (delay-compensated 200Hz VS) = {local_vs:.3f}', va='bottom', fontweight='bold', pad=20)

    fig.subplots_adjust(hspace=0.4)
    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_3_Spike_Raster_Phase_Locking')
    
    print("="*85)
    print("[DEBUG] END PLOT_FIGURE_3 ANALYSIS")
    print("="*85 + "\n")
>>>>>>> 1aab52592594dc4dde9dbed65fe7e1e4792c83ce

# =============================================================================
# Figure 4
# =============================================================================


def plot_figure4(data: Dict) -> None:
    kwave = data['kwave']['methods']
    population = data['population']
    fig, axes = plt.subplots(1, len(PRIMARY_COMPARE), figsize=(14, 6.4), sharey=True, gridspec_kw={'wspace': 0.1})

    for ax, method in zip(axes, PRIMARY_COMPARE):
        method_data = kwave[method]
        coords = population[method]['receptor_coords_m']
        lattice_center_idx = choose_centerline_receptors(coords, len(np.unique(np.round(coords[:, 0], 6))))
        x_coords = coords[lattice_center_idx, 0]
        sort_idx = np.argsort(x_coords)
        x_coords = x_coords[sort_idx]

        dyn = compute_dynamic_components(method_data)
        integrator = CoherentIntegrator(method_data['roi_x'], method_data['roi_y'], coords[lattice_center_idx], SHEAR_SPEED, LAMBDA_SPACE, data['kwave']['dt'])
        comp_weights = []
        for key in ORTHO_COMPONENTS:
            m_drive = integrator.integrate(dyn[key])
            u_drive = apply_pacinian_filter(m_drive, data['kwave']['dt'])
            positive = np.maximum(u_drive, 0.0)
            weight_line = positive.mean(axis=1)
            comp_weights.append(weight_line[sort_idx])
            ax.plot(x_coords * 1000.0, weight_line[sort_idx], color=COMPONENT_COLORS[key], lw=5, label=COMPONENT_LABELS[key])

        envelope = np.maximum.reduce(comp_weights)
        ax.plot(x_coords * 1000.0, envelope, color='black', lw=5.0, ls='--', label='Max-pooling envelope', alpha=0.8)
        ax.fill_between(x_coords * 1000.0, 0, envelope, color='0.7', alpha=0.15)
        ax.set_title(f'{method}', fontweight='bold')
        ax.set_xlabel('x [mm]', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.35)

    axes[0].set_ylabel('Neural drive weight [a.u.]', fontweight='bold')
    handles = [Line2D([0], [0], color=COMPONENT_COLORS[k], lw=LINE_WIDTH, label=COMPONENT_LABELS[k]) for k in ORTHO_COMPONENTS]
    handles.append(Line2D([0], [0], color='black', lw=3.0, ls='--', label='Max-pooling envelope'))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.02))
    # fig.suptitle('Figure 4 | Directional Max-Pooling Resolution', fontweight='bold', y=1.08)
    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_4_Directional_Max_Pooling')


# =============================================================================
# Figure 5
# =============================================================================

from scipy.interpolate import griddata

def plot_figure5(data: Dict) -> None:
    population = data['population']
    
    # 用于存储插值后的高分辨率连续场
    high_res_maps = []
    extents = None
    
    # 生成统一的高分辨率空间网格 (例如 100x100 像素，确保平滑)
    grid_res = 100 
    
    for m in METHOD_ORDER:
        method_pop = population[m]
        # 提取绝对物理坐标和对应的权重
        coords = method_pop['receptor_coords_m']
        weights = np.asarray(method_pop['weights'], dtype=np.float64)
        
        if extents is None:
            # 锁定物理坐标边界 (转换为毫米)
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            extents = [x_min * 1000, x_max * 1000, y_min * 1000, y_max * 1000]
            
            # 生成高分辨率的查询网格
            grid_x, grid_y = np.meshgrid(
                np.linspace(x_min, x_max, grid_res),
                np.linspace(y_min, y_max, grid_res)
            )
            
        # 核心修复：使用物理坐标进行三次样条插值 (Cubic Interpolation)
        # 这无视任何数组的内部排序，直接在 2D 物理空间上重建连续的感受曲面
        map_hr = griddata(
            points=(coords[:, 0], coords[:, 1]), 
            values=weights, 
            xi=(grid_x, grid_y), 
            method='cubic', 
            fill_value=np.min(weights) # 边缘外推使用最小值
        )
        
        high_res_maps.append(map_hr)
        
    # 计算全局颜色对齐的阈值
    vmax = max(np.nanmax(sm) for sm in high_res_maps)
    vmin = min(np.nanmin(sm) for sm in high_res_maps)
    
    # 使用 gridspec_kw 控制子图间水平间距 (wspace)
    # 去除 constrained_layout=True 以便更自由地控制布局，wspace=0.05 表示子图间距为子图宽度的 5%
<<<<<<< HEAD
    fig, axes = plt.subplots(1, 5, figsize=(30, 4), gridspec_kw={'wspace': 0.3})
=======
    fig, axes = plt.subplots(1, 5, figsize=(25, 4), gridspec_kw={'wspace': 0.2})
>>>>>>> 1aab52592594dc4dde9dbed65fe7e1e4792c83ce
    
    for ax, method, sm in zip(axes, METHOD_ORDER, high_res_maps):
        
        # 绘制极度平滑的连续空间触觉场
        im = ax.imshow(sm, cmap='Greens', vmin=0, vmax=vmax, 
                       origin='lower', extent=extents, aspect='equal')
        
        # 修正等高线逻辑：基于局部对比度的相对半高全宽
        local_min = np.nanmin(sm)
        local_max = np.nanmax(sm)
        relative_thr = local_min + 0.5 * (local_max - local_min)
        
        # 只有当对比度足够时才绘制等高线 (避免平坦区域画出杂乱线)
        if (local_max - local_min) > (0.1 * vmax):
            X_hr, Y_hr = np.meshgrid(
                np.linspace(extents[0], extents[1], grid_res),
                np.linspace(extents[2], extents[3], grid_res)
            )
            ax.contour(X_hr, Y_hr, sm, levels=[relative_thr], colors='white', linewidths=2.0, alpha=0.9)
            
        ax.set_title(method, fontweight='bold', fontsize=TITLE_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])

    # 全局 Colorbar
    # pad: 控制 colorbar 与子图的间距，0.01 表示非常靠近
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.75, aspect=25, pad=0.05)
    cbar.set_label('Population weight [a.u.]', fontweight='bold', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.outline.set_linewidth(1.0)
    
    # fig.suptitle('Figure 5 | Population Weight Maps', fontweight='bold', fontsize=TITLE_SIZE+2, y=1.05)
    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_5_Population_Weight_Maps')


# =============================================================================
# Figure 6
# =============================================================================


def plot_figure6(data: Dict) -> None:
    summary = data['summary']
    experiment = data['experiment']
    methods = METHOD_ORDER
    x = np.array([experiment['scores'][m] for m in methods], dtype=np.float64)
    xerr = np.array([experiment['se'][m] for m in methods], dtype=np.float64)
    y = np.array([summary['intensity_zscore'][m] for m in methods], dtype=np.float64)

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_fit = np.linspace(x.min() - 0.2, x.max() + 0.2, 200)
    y_fit = intercept + slope * x_fit

    n = len(x)
    x_mean = x.mean()
    s_err = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / max(n - 2, 1))
    ssx = np.sum((x - x_mean) ** 2)
    t_val = stats.t.ppf(0.975, max(n - 2, 1))
    conf = t_val * s_err * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / max(ssx, 1e-12))

    # --- Figure 6a: Regression ---
    fig_reg, ax_reg = plt.subplots(figsize=(6, 6), constrained_layout=True)

    for m, xi, yi, xe in zip(methods, x, y, xerr):
        ax_reg.errorbar(xi, yi, xerr=xe, fmt='o', ms=10, capsize=6, color=METHOD_COLORS[m], mec='black', mew=1.2, elinewidth=3, label=m)
    
    ax_reg.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=True, fancybox=True, framealpha=0.8)

    ax_reg.plot(x_fit, y_fit, color='black', lw=2.5, alpha=0.8)
    ax_reg.fill_between(x_fit, y_fit - conf, y_fit + conf, color='0.75', alpha=0.2)
    ax_reg.set_xlabel('Subjective BT score (log-odds ± SE)', fontweight='bold', fontsize=20)
    ax_reg.set_ylabel('Predicted intensity z-score', fontweight='bold', fontsize=20)
    ax_reg.grid(True, linestyle='--', alpha=0.35)
    
    # R2 and p-value annotation
    ax_reg.text(0.03, 0.97, f'$R^2$ = {r_value ** 2:.3f}\n$p$ = {p_value:.3g}**', 
                transform=ax_reg.transAxes, ha='left', va='top', fontsize=20, 
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.8'))

    # Ensure square shape for the regression plot
    ax_reg.set_box_aspect(1)
    save_figure(fig_reg, OUTPUT_DIR / 'Figure_Neural_6a_Model_vs_Psychophysics_Regression')

    # --- Figure 6b: Symmetric Pairwise Matrix ---
    fig_mat, ax_mat = plt.subplots(figsize=(6, 6), constrained_layout=True)

    exp_mat = build_experiment_win_fraction_matrix(methods, experiment['win_matrix_csv'])
    model_mat = build_pairwise_matrix(methods, summary['pairwise']['intensity'])
    
    combo = np.full_like(exp_mat, np.nan, dtype=np.float64)
    for i in range(len(methods)):
        for j in range(len(methods)):
            if i < j:
                # Upper triangle: Model P(Row > Col)
                combo[i, j] = model_mat[i, j]
            elif i > j:
                # Lower triangle: Exp P(Col > Row)
                combo[i, j] = exp_mat[j, i]
            else:
                # Diagonal
                combo[i, j] = 0.5
    im = ax_mat.imshow(combo, cmap='Greens', vmin=0, vmax=1)
    
    ax_mat.set_xticks(range(len(methods)))
    ax_mat.set_yticks(range(len(methods)))
    ax_mat.set_xticklabels(methods, rotation=0, ha='center', va='top', fontsize=16, fontweight='bold')
    ax_mat.set_yticklabels(methods, fontsize=16, ha='right', va='center', fontweight='bold')
    
    ax_mat.set_title('Upper: Model P(Row>Col) | Lower: Exp P(Col>Row)', fontsize=20, fontweight='bold', pad=15)
    
    for i in range(len(methods)):
        for j in range(len(methods)):
            if np.isfinite(combo[i, j]):
                # text_color = 'white' if abs(combo[i, j] - 0.5) > 0.25 else 'black'
                text_color = 'white'
                ax_mat.text(j, i, f'{combo[i, j]:.2f}', ha='center', va='center', 
                            color=text_color, fontsize=18, fontweight='bold')
    ax_mat.set_aspect('equal')
    ax_mat.set_box_aspect(1)
    save_figure(fig_mat, OUTPUT_DIR / 'Figure_Neural_6b_Model_vs_Psychophysics_Matrix')



# =============================================================================
# Saving / main
# =============================================================================


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{stem}.png', bbox_inches='tight')
    fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all_data()
    plot_figure1(data)
    plot_figure2(data)
    plot_figure3(data)
    plot_figure4(data)
    plot_figure5(data)
    plot_figure6(data)
    print('Neural dynamics figures generated successfully.')


if __name__ == '__main__':
    main()
