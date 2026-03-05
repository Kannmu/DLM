import os
import glob
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
except Exception:
    BinomialBayesMixedGLM = None

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13


def _to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient='records')
    return obj


def save_structured_results(output_dir, filename, payload):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_to_serializable)
    print(f"Saved structured analysis results: {path}")


def load_data(data_dir):
    """
    Load all CSV files from the data directory and concatenate them.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []

    if not all_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame(), []

    load_logs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
            msg = f"Loaded {filename}, shape: {df.shape}"
            print(msg)
            load_logs.append({
                'file': filename,
                'shape': {'rows': int(df.shape[0]), 'cols': int(df.shape[1])},
                'status': 'ok'
            })
        except Exception as e:
            msg = f"Error loading {filename}: {e}"
            print(msg)
            load_logs.append({'file': filename, 'status': 'error', 'error': str(e)})

    if not df_list:
        return pd.DataFrame(), load_logs

    return pd.concat(df_list, ignore_index=True), load_logs


def build_long_format(subset, winner_col):
    rows = []
    if 'ParticipantID' not in subset.columns:
        subset = subset.copy()
        subset['ParticipantID'] = 'P0'
    for _, row in subset.iterrows():
        winner = row[winner_col]
        if pd.isna(winner):
            continue
        stim_a = row['StimulusA']
        stim_b = row['StimulusB']
        if pd.isna(stim_a) or pd.isna(stim_b):
            continue
        if winner == stim_a:
            outcome = 1
        elif winner == stim_b:
            outcome = 0
        else:
            continue
        rows.append({
            'ParticipantID': row['ParticipantID'],
            'StimulusA': stim_a,
            'StimulusB': stim_b,
            'Outcome': outcome
        })
    return pd.DataFrame(rows)


def compute_zlog_rt(df, rt_col='ReactionTime', min_rt=0.2, max_rt=10.0):
    df = df.copy()
    df[rt_col] = pd.to_numeric(df[rt_col], errors='coerce')
    if 'ParticipantID' not in df.columns:
        df['ParticipantID'] = 'P0'
    valid = df[rt_col].between(min_rt, max_rt)
    df['LogRT'] = np.where(valid, np.log(df[rt_col]), np.nan)
    stats = df.groupby('ParticipantID')['LogRT'].agg(rt_mean='mean', rt_std='std')
    df = df.join(stats, on='ParticipantID')
    df['ZLogRT'] = (df['LogRT'] - df['rt_mean']) / df['rt_std']
    df.loc[df['rt_std'].isna() | (df['rt_std'] == 0), 'ZLogRT'] = 0.0
    df.loc[df['LogRT'].isna(), 'ZLogRT'] = np.nan

    rt_summary = {
        'rt_column': rt_col,
        'min_rt': float(min_rt),
        'max_rt': float(max_rt),
        'n_total': int(len(df)),
        'n_valid_range': int(valid.sum()),
        'n_invalid_range': int((~valid).sum()),
        'n_logrt_non_nan': int(df['LogRT'].notna().sum()),
        'n_zlogrt_non_nan': int(df['ZLogRT'].notna().sum()),
        'participant_stats': stats.reset_index().replace({np.nan: None}).to_dict(orient='records')
    }

    return df, rt_summary


def build_rt_matrix(df, stimuli, z_col='ZLogRT'):
    matrix = pd.DataFrame(np.nan, index=stimuli, columns=stimuli)
    if df.empty or not stimuli:
        return matrix, {'pair_means': [], 'n_pairs': 0}
    valid = df[['StimulusA', 'StimulusB', z_col]].dropna()
    if valid.empty:
        return matrix, {'pair_means': [], 'n_pairs': 0}
    pair_keys = valid.apply(lambda r: tuple(sorted([r['StimulusA'], r['StimulusB']])), axis=1)
    valid = valid.assign(Pair=pair_keys)
    pair_means = valid.groupby('Pair')[z_col].mean()
    for (a, b), mean_val in pair_means.items():
        matrix.loc[a, b] = mean_val
        matrix.loc[b, a] = mean_val

    pair_means_payload = [
        {'stimulus_1': a, 'stimulus_2': b, 'mean_zlogrt': float(v)}
        for (a, b), v in pair_means.items()
    ]
    return matrix, {'pair_means': pair_means_payload, 'n_pairs': int(len(pair_means_payload))}


def plot_rt_heatmap(ax, rt_matrix, title, title_y=1.02):
    mask = rt_matrix.isna()
    sns.heatmap(
        rt_matrix,
        cmap="Greens",
        ax=ax,
        square=True,
        mask=mask,
        annot=True,
        fmt=".2f",
        annot_kws={'fontsize': 12},
        cbar_kws={'label': 'Z-Log-RT', 'pad': 0.02},
        linewidths=0.5,
        linecolor='white'
    )
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    ax.set_title(title, fontsize=20, fontweight='bold', y=title_y)
    ax.set_xlabel("Stimulus", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold', labelpad=8)
    ax.set_ylabel("Stimulus", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold', labelpad=8)

    current_x = [item.get_text() for item in ax.get_xticklabels()]
    current_y = [item.get_text() for item in ax.get_yticklabels()]

    ax.set_xticklabels([format_method_name(label) for label in current_x], rotation=0, ha='center', va='top')
    ax.set_yticklabels([format_method_name(label) for label in current_y], rotation=0, ha='right', va='center')

    ax.tick_params(axis='both', labelsize=plt.rcParams['xtick.labelsize'])


def fit_bradley_terry_glmm(long_df, methods):
    if BinomialBayesMixedGLM is None:
        raise ImportError("statsmodels is required for Bradley-Terry mixed model. Install statsmodels to proceed.")
    methods = list(methods)
    reference = methods[-1]
    method_cols = [m for m in methods if m != reference]
    data = long_df.copy()
    for m in method_cols:
        data[m] = (data['StimulusA'] == m).astype(int) - (data['StimulusB'] == m).astype(int)
    formula = "Outcome ~ 0 + " + " + ".join(method_cols)
    re_formula = {"ParticipantID": "0 + C(ParticipantID)"}
    model = BinomialBayesMixedGLM.from_formula(formula, re_formula, data)
    result = model.fit_map()
    k_fe = getattr(model, "k_fe", None)
    if k_fe is None:
        k_fe = model.k_fep
    fe_params = result.params[:k_fe]
    cov = result.cov_params()
    if isinstance(cov, pd.DataFrame):
        cov = cov.to_numpy()
    if cov.shape[0] > k_fe:
        cov = cov[:k_fe, :k_fe]
    scores = pd.Series(0.0, index=methods)
    for i, m in enumerate(method_cols):
        scores[m] = fe_params[i]
    scores = scores - scores.mean()
    n = len(methods)
    M = np.zeros((n, k_fe))
    col_index = {m: i for i, m in enumerate(method_cols)}
    for i, m in enumerate(methods):
        if m in col_index:
            M[i, col_index[m]] = 1.0
    C = np.eye(n) - np.ones((n, n)) / n
    T = C @ M
    cov_scores = T @ cov @ T.T
    se = pd.Series(np.sqrt(np.diag(cov_scores)), index=methods)

    model_details = {
        'reference_method': reference,
        'method_columns': method_cols,
        'formula': formula,
        'n_observations': int(len(data)),
        'k_fixed_effects': int(k_fe),
        'fixed_effect_params': {method_cols[i]: float(fe_params[i]) for i in range(len(method_cols))},
        'cov_fixed_effects': cov.tolist(),
        'scores_centered': scores.to_dict(),
        'se_scores': se.to_dict(),
        'cov_scores': cov_scores.tolist()
    }

    return scores, se, cov_scores, result, model_details


def pairwise_wald(scores, cov_scores):
    methods = list(scores.index)
    pairs = []
    n = len(methods)
    for i in range(n):
        for j in range(i + 1, n):
            var = cov_scores[i, i] + cov_scores[j, j] - 2 * cov_scores[i, j]
            if var <= 0 or np.isnan(var):
                continue
            diff = scores.iloc[i] - scores.iloc[j]
            z = diff / np.sqrt(var)
            p = 2 * norm.sf(abs(z))
            odds_ratio = float(np.exp(diff))
            pairs.append({
                'method_1': methods[i],
                'method_2': methods[j],
                'score_diff': float(diff),
                'wald_z': float(z),
                'p_value': float(p),
                'odds_ratio': odds_ratio,
                'effect_size_log_odds': float(diff),
                'significance': significance_label(float(p))
            })
    return pairs


def significance_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def format_method_name(name):
    """
    Formats method names like 'ULM_L' to LaTeX-style subscript.
    Uses \mathregular to maintain the font style (Arial).
    """
    if '_' in name:
        parts = name.split('_', 1)
        base = parts[0]
        sub = parts[1]
        return rf"$\mathregular{{{base}}}_{{\mathregular{{{sub}}}}}$"
    return name


def plot_on_axis(ax, scores, se, cov_scores, y_label):
    scores_sorted = scores.sort_values(ascending=False)
    se_sorted = se.reindex(scores_sorted.index)
    ci = 1.96 * se_sorted
    plot_df = pd.DataFrame({
        'Method': scores_sorted.index,
        'Score': scores_sorted.values,
        'CI': ci.values
    })
    sns.barplot(x='Method', y='Score', data=plot_df, hue='Method', palette="Greens", errorbar=None, ax=ax, zorder=2, legend=False)
    ax.errorbar(x=range(len(plot_df)), y=plot_df['Score'], yerr=plot_df['CI'], fmt='none', c='black', capsize=6, elinewidth=1.5, zorder=5, clip_on=False)
    ax.set_xlabel("Method", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold', labelpad=8)
    ax.set_ylabel(y_label, fontsize=plt.rcParams['axes.labelsize'], fontweight='bold', labelpad=8)

    current_labels = [item.get_text() for item in ax.get_xticklabels()]
    formatted_labels = [format_method_name(label) for label in current_labels]
    ax.set_xticklabels(formatted_labels)

    ax.tick_params(axis='both', labelsize=plt.rcParams['xtick.labelsize'])
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    order = list(scores_sorted.index)
    idx = [scores.index.get_loc(m) for m in order]
    cov_order = cov_scores[np.ix_(idx, idx)]
    pairs = pairwise_wald(scores_sorted, cov_order)
    index_map = {m: i for i, m in enumerate(order)}
    sig_pairs = []
    for item in pairs:
        p = item['p_value']
        if p < 0.05:
            i = index_map[item['method_1']]
            j = index_map[item['method_2']]
            if i > j:
                i, j = j, i
            sig_pairs.append((i, j, p))
    sig_pairs.sort(key=lambda x: (x[0], x[1]))
    y_min = (plot_df['Score'] - plot_df['CI']).min()
    y_max = (plot_df['Score'] + plot_df['CI']).max()
    span = y_max - y_min
    if span == 0:
        span = 1.0
    line_height = span * 0.02
    step = span * 0.08
    y = y_max + step
    for i, j, p in sig_pairs:
        ax.plot([i, i, j, j], [y, y + line_height, y + line_height, y], c='black', lw=1.5, zorder=6, clip_on=False)
        ax.text((i + j) / 2, y + line_height, significance_label(p), ha='center', va='center', fontsize=13)
        y += step
    if sig_pairs:
        ax.set_ylim(y_min - span * 0.05, y + step)
    else:
        ax.set_ylim(y_min - span * 0.05, y_max + span * 0.1)


def save_combined_plot(intensity_res, clarity_res, rt_matrix, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(32, 10), sharey=False, gridspec_kw={'width_ratios': [1, 1, 1.3]})
    title_y = 1.02

    if intensity_res:
        scores, se, cov_scores = intensity_res
        plot_on_axis(axes[0], scores, se, cov_scores, "Intensity Score (Log-odds ± 95% CI)")
        axes[0].set_title("Intensity Preference", fontsize=20, fontweight='bold', y=title_y)

    if clarity_res:
        scores, se, cov_scores = clarity_res
        plot_on_axis(axes[1], scores, se, cov_scores, "Clarity Score (Log-odds ± 95% CI)")
        axes[1].set_title("Spatial Clarity Preference", fontsize=20, fontweight='bold', y=title_y)

    if rt_matrix is not None:
        plot_rt_heatmap(axes[2], rt_matrix, "Reaction Time (Z-Log-RT)", title_y=title_y)

    fig.align_ylabels(axes[:2])
    fig.align_xlabels(axes)
    fig.subplots_adjust(top=0.6, bottom=0.18, wspace=0.35)
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def analyze_block(df, block_type, winner_col, output_dir, file_prefix):
    print(f"\nAnalyzing {block_type}...")

    subset = df[df['BlockType'] == block_type].copy()

    if subset.empty:
        subset = df[df[winner_col].notna()].copy()
        if subset.empty:
            print(f"No data found for {block_type}")
            return None, None, None, {'status': 'no_data', 'block_type': block_type}

    stimuli = pd.unique(subset[['StimulusA', 'StimulusB']].values.ravel('K'))
    stimuli = [s for s in stimuli if pd.notna(s)]
    stimuli = sorted(stimuli)

    if not stimuli:
        print("No stimuli found.")
        return None, None, None, {'status': 'no_stimuli', 'block_type': block_type}

    print(f"Stimuli found: {stimuli}")

    win_matrix = pd.DataFrame(0, index=stimuli, columns=stimuli)

    for _, row in subset.iterrows():
        winner = row[winner_col]
        if pd.isna(winner):
            continue

        stim_a = row['StimulusA']
        stim_b = row['StimulusB']

        if winner == stim_a:
            loser = stim_b
        elif winner == stim_b:
            loser = stim_a
        else:
            continue

        win_matrix.loc[winner, loser] += 1

    print("Win Matrix:")
    print(win_matrix)

    win_matrix_path = os.path.join(output_dir, f"{file_prefix}_win_matrix.csv")
    win_matrix.to_csv(win_matrix_path)

    long_df = build_long_format(subset, winner_col)
    if long_df.empty:
        print("No valid trials for modeling.")
        return None, None, None, {
            'status': 'no_valid_trials',
            'block_type': block_type,
            'stimuli': stimuli,
            'win_matrix_csv': win_matrix_path
        }

    scores, se, cov_scores, result, model_details = fit_bradley_terry_glmm(long_df, stimuli)
    print("BT Mixed Model Scores:")
    print(scores)

    pairs = pairwise_wald(scores, cov_scores)
    p_table = pd.DataFrame(np.nan, index=stimuli, columns=stimuli)
    for item in pairs:
        a = item['method_1']
        b = item['method_2']
        p = item['p_value']
        p_table.loc[a, b] = p
        p_table.loc[b, a] = p

    print("Pairwise p-values (Wald):")
    print(p_table)

    block_payload = {
        'status': 'ok',
        'block_type': block_type,
        'winner_column': winner_col,
        'file_prefix': file_prefix,
        'n_rows_subset': int(len(subset)),
        'n_valid_trials_model': int(len(long_df)),
        'stimuli': stimuli,
        'win_matrix_csv': win_matrix_path,
        'win_matrix': win_matrix.reset_index().rename(columns={'index': 'stimulus'}).to_dict(orient='records'),
        'long_format_preview': long_df.head(20).to_dict(orient='records'),
        'long_format_n_rows': int(len(long_df)),
        'score_by_method': scores.to_dict(),
        'se_by_method': se.to_dict(),
        'ci95_by_method': (1.96 * se).to_dict(),
        'cov_scores': cov_scores.tolist(),
        'pairwise_wald': pairs,
        'pairwise_p_table': p_table.reset_index().rename(columns={'index': 'stimulus'}).to_dict(orient='records'),
        'model_details': model_details,
        'optimizer_converged': getattr(result, 'optim_retvals', None)
    }

    save_structured_results(output_dir, f"{file_prefix}_detailed_results.json", block_payload)
    return scores, se, cov_scores, block_payload


def main():
    base_dir = r"d:\Data\OneDrive\Papers\SWIM\Codes\Experiment 1"
    data_dir = os.path.join(base_dir, "Data")
    output_dir = os.path.join(base_dir, "Analysis")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    df, load_logs = load_data(data_dir)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    df_rt, rt_summary = compute_zlog_rt(df)
    stimuli = pd.unique(df[['StimulusA', 'StimulusB']].values.ravel('K'))
    stimuli = [s for s in stimuli if pd.notna(s)]
    stimuli = sorted(stimuli)
    rt_matrix, rt_matrix_details = build_rt_matrix(df_rt, stimuli)

    res_intensity = analyze_block(
        df,
        block_type='Intensity',
        winner_col='Chosen_Intensity',
        output_dir=output_dir,
        file_prefix='Intensity'
    )

    res_clarity = analyze_block(
        df,
        block_type='Spatial',
        winner_col='Chosen_Clarity',
        output_dir=output_dir,
        file_prefix='Clarity'
    )

    intensity_for_plot = None if res_intensity[0] is None else res_intensity[:3]
    clarity_for_plot = None if res_clarity[0] is None else res_clarity[:3]

    save_combined_plot(intensity_for_plot, clarity_for_plot, rt_matrix, os.path.join(output_dir, "Experiment 1 Combined"))

    overall_payload = {
        'data_loading': {
            'data_dir': data_dir,
            'n_rows': int(len(df)),
            'n_cols': int(df.shape[1]),
            'columns': list(df.columns),
            'load_logs': load_logs
        },
        'reaction_time_analysis': {
            'summary': rt_summary,
            'stimuli': stimuli,
            'rt_matrix': rt_matrix.reset_index().rename(columns={'index': 'stimulus'}).to_dict(orient='records'),
            'rt_matrix_details': rt_matrix_details
        },
        'block_analyses': {
            'intensity': None if res_intensity is None else res_intensity[3],
            'clarity': None if res_clarity is None else res_clarity[3]
        },
        'output_files': {
            'combined_plot_pdf': os.path.join(output_dir, "Experiment 1 Combined.pdf"),
            'combined_plot_svg': os.path.join(output_dir, "Experiment 1 Combined.svg"),
            'combined_plot_png': os.path.join(output_dir, "Experiment 1 Combined.png"),
            'intensity_detail_json': os.path.join(output_dir, "Intensity_detailed_results.json"),
            'clarity_detail_json': os.path.join(output_dir, "Clarity_detailed_results.json"),
            'overall_detail_json': os.path.join(output_dir, "Experiment1_overall_detailed_results.json")
        }
    }
    save_structured_results(output_dir, "Experiment1_overall_detailed_results.json", overall_payload)


if __name__ == "__main__":
    main()
