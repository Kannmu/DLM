import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'PWM_2_Intensity.csv')
OUTPUT_BASENAME = os.path.join(BASE_DIR, 'Figure_PWM_Duty_to_Acoustic_Output')
REFERENCE_DUTY = 0.5
REFERENCE_VPP = 14.1


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, sep='\t')
    df.columns = [col.strip() for col in df.columns]

    for col in ['Strength', 'Duty Ratio', 'Vpp Measured']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Duty Ratio', 'Vpp Measured']).copy()
    df = df.sort_values('Duty Ratio').reset_index(drop=True)

    df['Normalized Acoustic Pressure'] = df['Vpp Measured'] / REFERENCE_VPP
    df['Normalized Acoustic Intensity/Force'] = df['Normalized Acoustic Pressure'] ** 2
    df['Theoretical Normalized Pressure'] = np.sin(np.pi * df['Duty Ratio'])
    df['Theoretical Normalized Intensity'] = df['Theoretical Normalized Pressure'] ** 2
    df['Absolute Intensity Error'] = np.abs(
        df['Normalized Acoustic Intensity/Force'] - df['Theoretical Normalized Intensity']
    )
    df['Pressure Residual'] = (
        df['Normalized Acoustic Pressure'] - df['Theoretical Normalized Pressure']
    )
    df['Intensity Residual'] = (
        df['Normalized Acoustic Intensity/Force'] - df['Theoretical Normalized Intensity']
    )

    return df


def build_theoretical_curve():
    duty = np.linspace(0.0, 0.5, 600)
    pressure = np.sin(np.pi * duty)
    intensity = pressure ** 2
    return pd.DataFrame({
        'Duty Ratio': duty,
        'Theoretical Normalized Pressure': pressure,
        'Theoretical Normalized Intensity': intensity,
    })


def compute_summary(df):
    pressure_rmse = float(np.sqrt(np.mean(df['Pressure Residual'] ** 2)))
    intensity_rmse = float(np.sqrt(np.mean(df['Intensity Residual'] ** 2)))
    pressure_mae = float(np.mean(np.abs(df['Pressure Residual'])))
    intensity_mae = float(np.mean(np.abs(df['Intensity Residual'])))
    pressure_r = float(np.corrcoef(df['Normalized Acoustic Pressure'], df['Theoretical Normalized Pressure'])[0, 1])
    intensity_r = float(np.corrcoef(df['Normalized Acoustic Intensity/Force'], df['Theoretical Normalized Intensity'])[0, 1])
    max_error_row = df.loc[df['Absolute Intensity Error'].idxmax()]

    low_mask = df['Duty Ratio'] <= 0.15
    mid_mask = (df['Duty Ratio'] > 0.15) & (df['Duty Ratio'] <= 0.35)
    high_mask = df['Duty Ratio'] > 0.35

    def mean_if_any(mask, column):
        if mask.sum() == 0:
            return np.nan
        return float(df.loc[mask, column].mean())

    return {
        'n_points': int(len(df)),
        'pressure_rmse': pressure_rmse,
        'intensity_rmse': intensity_rmse,
        'pressure_mae': pressure_mae,
        'intensity_mae': intensity_mae,
        'pressure_r': pressure_r,
        'intensity_r': intensity_r,
        'max_error_duty': float(max_error_row['Duty Ratio']),
        'max_error_value': float(max_error_row['Absolute Intensity Error']),
        'mean_intensity_error_low': mean_if_any(low_mask, 'Absolute Intensity Error'),
        'mean_intensity_error_mid': mean_if_any(mid_mask, 'Absolute Intensity Error'),
        'mean_intensity_error_high': mean_if_any(high_mask, 'Absolute Intensity Error'),
    }


def add_primary_plot(ax, data_df, theory_df, palette):
    ax.plot(
        theory_df['Duty Ratio'],
        theory_df['Theoretical Normalized Pressure'],
        linestyle='--',
        linewidth=2,
        color=palette[2],
        alpha=0.95
    )
    ax.plot(
        theory_df['Duty Ratio'],
        theory_df['Theoretical Normalized Intensity'],
        linestyle='--',
        linewidth=3,
        color=palette[4],
        alpha=0.95
    )

    sc_pressure = ax.scatter(
        data_df['Duty Ratio'],
        data_df['Normalized Acoustic Pressure'],
        c=data_df['Strength'],
        cmap='Greens',
        s=100,
        marker='o',
        edgecolor='black',
        linewidth=1,
        alpha=0.9,
        zorder=4
    )
    ax.plot(
        data_df['Duty Ratio'],
        data_df['Normalized Acoustic Pressure'],
        color=palette[2],
        linewidth=3,
        alpha=0.85,
        zorder=3
    )

    ax.scatter(
        data_df['Duty Ratio'],
        data_df['Normalized Acoustic Intensity/Force'],
        c=data_df['Strength'],
        cmap='Greens',
        s=100,
        marker='s',
        edgecolor='black',
        linewidth=1,
        alpha=0.9,
        zorder=5
    )
    ax.plot(
        data_df['Duty Ratio'],
        data_df['Normalized Acoustic Intensity/Force'],
        color=palette[4],
        linewidth=3,
        alpha=0.9,
        zorder=4
    )

    # for _, row in data_df.iterrows():
    #     if row['Duty Ratio'] in [0.10, 0.25, 0.50]:
    #         ax.annotate(
    #             f"D={row['Duty Ratio']:.2f}",
    #             xy=(row['Duty Ratio'], row['Normalized Acoustic Intensity/Force']),
    #             xytext=(6, 8),
    #             textcoords='offset points',
    #             fontsize=9.5,
    #             color='black',
    #             bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='0.72', alpha=0.9)
    #         )

    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 1.03)
    ax.set_xticks(np.arange(0, 0.51, 0.05))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel('PWM duty ratio (D)')
    ax.set_ylabel('Normalized output')
    # ax.set_title('PWM duty ratio vs. acoustic output')
    ax.grid(True, linestyle='--', alpha=0.28)

    legend_handles = [
        Line2D([0], [0], color=palette[2], lw=3, ls='--', label='Theory pressure'),
        Line2D([0], [0], color=palette[4], lw=3, ls='--', label='Theory intensity'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[2], markeredgecolor='black', markersize=15, label='Measured pressure'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=palette[4], markeredgecolor='black', markersize=15, label='Measured intensity'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', frameon=True, borderpad=0.4, handlelength=1.8)
    return sc_pressure


def add_error_inset(ax, data_df, palette):
    inset = ax.inset_axes([0.565, 0.06, 0.4, 0.25])
    inset.bar(
        data_df['Duty Ratio'],
        data_df['Absolute Intensity Error'],
        width=0.02,
        color=palette[3],
        edgecolor='black',
        linewidth=0.5,
        alpha=0.9
    )
    inset.set_title('|Δ intensity|', fontsize=14, pad=5)
    inset.set_xlim(0.0, 0.5)
    inset.set_ylim(0.0, max(0.08, data_df['Absolute Intensity Error'].max() * 1.2))
    inset.tick_params(axis='both', labelsize=12)
    inset.grid(True, linestyle=':', alpha=0.25)


def add_summary_box(ax, summary):
    text = (
        f"n = {summary['n_points']}\n"
        f"Pressure: r = {summary['pressure_r']:.3f}, RMSE = {summary['pressure_rmse']:.3f}\n"
        f"Intensity: r = {summary['intensity_r']:.3f}, RMSE = {summary['intensity_rmse']:.3f}\n"
        f"Max |ΔI| at D = {summary['max_error_duty']:.2f}"
    )
    ax.text(
        0.965,
        0.38,
        text,
        transform=ax.transAxes,
        fontsize=14,
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='0.62', alpha=0.95)
    )


def add_equation_box(ax):
    text = (
        "Vpp ∝ P\n"
        "Pnorm = Vpp(D) / Vpp(0.5)\n"
        "Inorm = Pnorm²\n"
        "Theory: sin(πD), sin²(πD)"
    )
    ax.text(
        0.985,
        0.985,
        text,
        transform=ax.transAxes,
        fontsize=9.8,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='0.62', alpha=0.95)
    )


def add_top_axis(ax):
    top_ax = ax.secondary_xaxis('top', functions=(lambda x: x / 0.5 * 100, lambda s: s / 100 * 0.5))
    top_ax.set_xlabel('PWM strength parameter (%)', fontsize=13, fontweight='bold')
    top_ax.set_xticks(np.arange(0, 101, 10))
    top_ax.tick_params(axis='x', labelsize=11)


def save_outputs(fig, output_basename):
    fig.savefig(f'{output_basename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_basename}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_basename}.svg', dpi=300, bbox_inches='tight')


def main():
    data_df = load_and_prepare_data(CSV_PATH)
    theory_df = build_theoretical_curve()
    summary = compute_summary(data_df)
    palette = sns.color_palette('Greens', 6)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = add_primary_plot(ax, data_df, theory_df, palette)
    add_error_inset(ax, data_df, palette)
    add_summary_box(ax, summary)
    # add_equation_box(ax)
    # add_top_axis(ax)

    cbar = fig.colorbar(sc, ax=ax, pad=0.014, fraction=0.045, shrink=0.94)
    cbar.set_label('Strength (%)', fontweight='bold', fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout(rect=[0.0, 0.01, 0.98, 1.0])
    save_outputs(fig, OUTPUT_BASENAME)
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
