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
    fig, axes = plt.subplots(1, 5, figsize=(25, 4), gridspec_kw={'wspace': 0.2})
    
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
