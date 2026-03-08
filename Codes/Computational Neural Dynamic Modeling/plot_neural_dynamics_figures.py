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
from scipy.signal import butter, get_window, sosfilt


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
WATERFALL_RECEPTORS = 13
FREQ_MAX = 800.0

METHOD_COLORS = {
    'ULM_L': '#440154',
    'DLM_2': '#3b528b',
    'DLM_3': '#21918c',
    'LM_C': '#5ec962',
    'LM_L': '#fde725',
}
SPECTRUM_COLORS = {'ULM_L': '#c73e1d', 'LM_L': '#2a6fbb'}
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

    # 第一遍遍历：计算全局最大值，保证左右两图的Y轴比例尺完全一致 (修复 2)
    global_max_tau = 0.0
    global_max_m = 0.0
    plot_data_cache = {}

    for method in PRIMARY_COMPARE:
        method_data = kwave[method]
        dyn = compute_dynamic_components(method_data)['xy'] # 这里可以保留xy，展示特定波前
        lattice = build_receptor_lattice(method_data['roi_x'], method_data['roi_y'])
        receptor_idx = choose_centerline_receptors(lattice['coords_m'], WATERFALL_RECEPTORS)
        integrator = CoherentIntegrator(method_data['roi_x'], method_data['roi_y'], 
                                        lattice['coords_m'][receptor_idx], 
                                        SHEAR_SPEED, LAMBDA_SPACE, data['kwave']['dt'])
        m_drive = integrator.integrate(dyn)

        y_idx = int(np.argmin(np.abs(method_data['roi_y'])))
        x_indices = [int(np.argmin(np.abs(method_data['roi_x'] - x0))) for x0 in lattice['coords_m'][receptor_idx, 0]]
        tau_traces = np.stack([dyn[y_idx, xi, :] for xi in x_indices], axis=0)

        t_ms = method_data['t'] * 1000.0
        window_mask = t_ms >= (t_ms.max() - RASTER_DURATION_MS)
        
        tau_win = tau_traces[:, window_mask]
        m_win = m_drive[:, window_mask]
        
        global_max_tau = max(global_max_tau, np.max(np.abs(tau_win)))
        global_max_m = max(global_max_m, np.max(np.abs(m_win)))
        
        plot_data_cache[method] = (t_ms[window_mask], tau_win, m_win)

    # 设定全局统一的 offset
    offset_tau = global_max_tau * 1.5 + 1e-6
    offset_m = global_max_m * 1.2 + 1e-6

    for col, method in enumerate(PRIMARY_COMPARE):
        t_win, tau_win, m_win = plot_data_cache[method]
        
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[col], height_ratios=[1, 1], hspace=0.15)
        ax_top = fig.add_subplot(inner[0])
        ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)

        for ridx in range(tau_win.shape[0]):
            base = ridx * offset_tau
            smooth = gaussian_filter1d(tau_win[ridx], sigma=1.0)
            ax_top.fill_between(t_win, base, base + np.clip(smooth, 0, None), color='#b40426', alpha=0.45, linewidth=0)
            ax_top.fill_between(t_win, base, base + np.clip(smooth, None, 0), color='#3b4cc0', alpha=0.45, linewidth=0)
            ax_top.plot(t_win, base + smooth, color='0.35', lw=0.9, alpha=0.9)

        for ridx in range(m_win.shape[0]):
            ax_bottom.plot(t_win, m_win[ridx] + ridx * offset_m, color='black', lw=1.6, alpha=0.9)

        # 修复：合理的 Gain 计算（该方法的 m 极值 / tau 极值）
        method_max_tau = np.max(np.abs(tau_win))
        method_max_m = np.max(np.abs(m_win))
        gain = method_max_m / max(method_max_tau, 1e-12)

        ax_top.set_title(f'{method} | Raw shear wavefronts', fontweight='bold')
        ax_bottom.set_title(f'{method} | Delayed coherent drive m(t)', fontweight='bold', pad=6)
        ax_top.text(0.02, 0.95, f'Peak Gain ≈ {gain:.1f}×', transform=ax_top.transAxes, 
                    ha='left', va='top', fontsize=LEGEND_SIZE, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
        
        ax_top.set_ylabel('Virtual receptors', fontweight='bold')
        ax_bottom.set_ylabel('Integrated traces', fontweight='bold')
        ax_bottom.set_xlabel('Time [ms]', fontweight='bold')
        
        # 统一 Y 轴边界
        ax_top.set_ylim(-offset_tau, tau_win.shape[0] * offset_tau)
        ax_bottom.set_ylim(-offset_m, m_win.shape[0] * offset_m)
        
        ax_top.set_yticks(np.arange(tau_win.shape[0]) * offset_tau)
        ax_top.set_yticklabels([str(i + 1) for i in range(tau_win.shape[0])])
        ax_bottom.set_yticks([])
        ax_top.grid(False)
        ax_bottom.grid(True, linestyle='--', alpha=0.35)
        ax_top.tick_params(labelbottom=False)

    fig.suptitle('Figure 1 | Spatiotemporal Coherent Integration Dynamics', fontweight='bold', y=0.995)
    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_1_Coherent_Integration_Dynamics')


# =============================================================================
# Figure 2
# =============================================================================

def plot_figure2(data: Dict) -> None:
    kwave = data['kwave']['methods']
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
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
        
        ax.plot(freqs_m, spec_m_db, color=color, lw=LINE_WIDTH, label=f'{method} | m(t)')
        ax.plot(freqs_u, spec_u_db, color=color, lw=2.2, ls='--', alpha=0.8, label=f'{method} | u(t)')

    ax.set_xlim(0, FREQ_MAX)
    ax.set_ylim(-200, 80) # 建议固定Y轴范围，避免数据抖动导致留白过大
    ax.set_xlabel('Frequency [Hz]', fontweight='bold')
    ax.set_ylabel('PSD [dB]', fontweight='bold')
    ax.set_title('Figure 2 | Frequency Fidelity & Receptor Tuning', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(frameon=True, ncol=1, loc='lower right')
    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_2_Frequency_Fidelity')


# =============================================================================
# Figure 3
# =============================================================================


def plot_figure3(data: Dict) -> None:
    population = data['population']
    fig = plt.figure(figsize=(13, 10))
    outer = gridspec.GridSpec(3, 2, width_ratios=[1.8, 1.0], wspace=0.15, hspace=0.4)

    for row, method in enumerate(RASTER_METHODS):
        method_pop = population[method]
        coords = method_pop['receptor_coords_m']
        selected = choose_central_neurons(coords, N_RASTER_NEURONS)
        spikes = method_pop['spikes'][0, selected, :]
        n_time = spikes.shape[1]
        dt = 0.015 / n_time
        t_ms = np.arange(n_time) * dt * 1000.0
        time_mask = t_ms <= RASTER_DURATION_MS

        ax_raster = fig.add_subplot(outer[row, 0])
        for n_idx, neuron_spikes in enumerate(spikes[:, time_mask], start=1):
            spike_times = t_ms[time_mask][neuron_spikes.astype(bool)]
            if spike_times.size:
                ax_raster.vlines(spike_times, n_idx - 0.38, n_idx + 0.38, color=METHOD_COLORS[method], lw=2)
        ax_raster.set_xlim(0, RASTER_DURATION_MS)
        ax_raster.set_ylim(0.5, len(selected) + 0.5)
        ax_raster.set_ylabel(f'{method}\nNeuron #', fontweight='bold')
        if row == len(RASTER_METHODS) - 1:
            ax_raster.set_xlabel('Time [ms]', fontweight='bold')
        else:
            ax_raster.tick_params(labelbottom=False)
        ax_raster.grid(True, linestyle='--', alpha=0.25)
        ax_raster.set_title(f'{method} | Spike raster', fontweight='bold', loc='left',y=1.2)

        ax_polar = fig.add_subplot(outer[row, 1], projection='polar')
        spike_times_all = np.where(spikes[:, time_mask])[1] * dt
        phases = (2.0 * np.pi * CARRIER_FREQ * spike_times_all) % (2.0 * np.pi)
        if phases.size == 0:
            phases = np.array([0.0])
        bins = np.linspace(0, 2 * np.pi, 17)
        ax_polar.hist(phases, bins=bins, color=METHOD_COLORS[method], alpha=0.8, edgecolor='white')
        vs = compute_vector_strength_from_spike_times(spike_times_all)
        mean_angle = np.angle(np.mean(np.exp(1j * phases))) if phases.size else 0.0
        ax_polar.plot([mean_angle, mean_angle], [0, ax_polar.get_rmax() if ax_polar.get_rmax() > 0 else 1], color='black', lw=2.2)
        ax_polar.set_title(f'{method} | Phase-locking | VS={vs:.3f}', va='bottom', fontweight='bold',y=1.2)
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)

    fig.suptitle('Figure 3 | Spike Raster & Phase-Locking', fontweight='bold', y=0.995)
    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_3_Spike_Raster_Phase_Locking')


# =============================================================================
# Figure 4
# =============================================================================


def plot_figure4(data: Dict) -> None:
    kwave = data['kwave']['methods']
    population = data['population']
    fig, axes = plt.subplots(1, len(PRIMARY_COMPARE), figsize=FIGSIZE_Figure_4, sharey=True)

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
            ax.plot(x_coords * 1000.0, weight_line[sort_idx], color=COMPONENT_COLORS[key], lw=LINE_WIDTH, label=COMPONENT_LABELS[key])

        envelope = np.maximum.reduce(comp_weights)
        ax.plot(x_coords * 1000.0, envelope, color='black', lw=3.0, ls='--', label='Max-pooling envelope')
        ax.fill_between(x_coords * 1000.0, 0, envelope, color='0.7', alpha=0.15)
        ax.set_title(f'{method}', fontweight='bold')
        ax.set_xlabel('x [mm]', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.35)

    axes[0].set_ylabel('Neural drive weight [a.u.]', fontweight='bold')
    handles = [Line2D([0], [0], color=COMPONENT_COLORS[k], lw=LINE_WIDTH, label=COMPONENT_LABELS[k]) for k in ORTHO_COMPONENTS]
    handles.append(Line2D([0], [0], color='black', lw=3.0, ls='--', label='Max-pooling envelope'))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Figure 4 | Directional Max-Pooling Resolution', fontweight='bold', y=1.08)
    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_4_Directional_Max_Pooling')


# =============================================================================
# Figure 5
# =============================================================================


def plot_figure5(data: Dict) -> None:
    population = data['population']
    maps = [population[m]['population_map'] for m in METHOD_ORDER]
    vmax = max(np.max(m) for m in maps)
    fig, axes = plt.subplots(1, 5, figsize=FIGSIZE_MATRIX, sharex=True, sharey=True)
    last_im = None
    for ax, method, pmap in zip(axes, METHOD_ORDER, maps):
        arr = np.asarray(pmap, dtype=np.float64)
        last_im = ax.imshow(arr, origin='lower', cmap=POPULATION_CMAP, vmin=0, vmax=vmax, aspect='auto')
        thr = 0.5 * np.max(arr)
        if np.any(arr >= thr):
            ax.contour(arr, levels=[thr], colors='white', linewidths=1.7)
        ax.set_title(method, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.022, pad=0.012)
    cbar.set_label('Population weight [a.u.]', fontweight='bold')
    fig.suptitle('Figure 5 | Population Weight Maps', fontweight='bold', y=0.98)
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

    fig, ax = plt.subplots(figsize=FIGSIZE_REGRESSION)
    for m, xi, yi, xe in zip(methods, x, y, xerr):
        ax.errorbar(xi, yi, xerr=xe, fmt='o', ms=9, capsize=4, color=METHOD_COLORS[m], mec='black', mew=0.7)
        ax.text(xi + 0.03, yi + 0.03, m, fontsize=LEGEND_SIZE, fontweight='bold')
    ax.plot(x_fit, y_fit, color='black', lw=2.5)
    ax.fill_between(x_fit, y_fit - conf, y_fit + conf, color='0.75', alpha=0.45)
    ax.set_xlabel('Subjective BT score [log-odds ± SE]', fontweight='bold')
    ax.set_ylabel('Predicted intensity z-score', fontweight='bold')
    ax.set_title('Figure 6 | Model vs. Psychophysics Alignment', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.text(0.03, 0.97, f'$R^2$ = {r_value ** 2:.3f}\n$p$ = {p_value:.3g}', transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_SIZE, bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.8'))

    inset = inset_axes(ax, width='42%', height='42%', loc='lower right', borderpad=1.2)
    exp_mat = build_experiment_win_fraction_matrix(methods, experiment['win_matrix_csv'])
    model_mat = build_pairwise_matrix(methods, summary['pairwise']['intensity'])
    combo = np.full_like(exp_mat, np.nan, dtype=np.float64)
    for i in range(len(methods)):
        for j in range(len(methods)):
            if i > j:
                combo[i, j] = exp_mat[i, j]
            elif i < j:
                combo[i, j] = model_mat[i, j]
            else:
                combo[i, j] = 0.5
    im = inset.imshow(combo, cmap='viridis', vmin=0, vmax=1)
    inset.set_xticks(range(len(methods)))
    inset.set_yticks(range(len(methods)))
    inset.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    inset.set_yticklabels(methods, fontsize=9)
    inset.set_title('Pairwise matrix\nlower: experiment / upper: model', fontsize=10, fontweight='bold')
    for i in range(len(methods)):
        for j in range(len(methods)):
            if np.isfinite(combo[i, j]):
                inset.text(j, i, f'{combo[i, j]:.2f}', ha='center', va='center', color='white', fontsize=7)
    cax = inset_axes(inset, width='5%', height='100%', loc='center right', borderpad=-2.4)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)

    save_figure(fig, OUTPUT_DIR / 'Figure_Neural_6_Model_vs_Psychophysics')


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
