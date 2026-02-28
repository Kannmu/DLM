import os
import glob
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

def load_data(data_dir):
    """
    Load all CSV files from the data directory and concatenate them.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    
    if not all_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame()
        
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
            print(f"Loaded {filename}, shape: {df.shape}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

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
    return scores, se, cov_scores, result

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
            pairs.append((methods[i], methods[j], p))
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
    ax.set_xlabel("Method", fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')
    ax.set_ylabel(y_label, fontsize=plt.rcParams['axes.labelsize'], fontweight='bold')

    # Format x-axis labels with subscripts
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
    for m1, m2, p in pairs:
        if p < 0.05:
            i = index_map[m1]
            j = index_map[m2]
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
        ax.text((i + j) / 2, y + line_height, significance_label(p), ha='center', va='bottom', fontsize=13)
        y += step
    if sig_pairs:
        ax.set_ylim(y_min - span * 0.05, y + step)
    else:
        ax.set_ylim(y_min - span * 0.05, y_max + span * 0.1)

def save_combined_plot(intensity_res, clarity_res, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    
    if intensity_res:
        scores, se, cov_scores = intensity_res
        plot_on_axis(axes[0], scores, se, cov_scores, "Relative Intensity Score (Log-odds ± 95% CI)")
        axes[0].set_title("Intensity Preference", fontsize=20, fontweight='bold')
    
    if clarity_res:
        scores, se, cov_scores = clarity_res
        plot_on_axis(axes[1], scores, se, cov_scores, "Relative Clarity Score (Log-odds ± 95% CI)")
        axes[1].set_title("Spatial Clarity Preference", fontsize=20, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def analyze_block(df, block_type, winner_col, output_dir, file_prefix):
    print(f"\nAnalyzing {block_type}...")
    
    subset = df[df['BlockType'] == block_type].copy()
    
    if subset.empty:
        subset = df[df[winner_col].notna()].copy()
        if subset.empty:
            print(f"No data found for {block_type}")
            return

    stimuli = pd.unique(subset[['StimulusA', 'StimulusB']].values.ravel('K'))
    stimuli = [s for s in stimuli if pd.notna(s)]
    stimuli = sorted(stimuli)
    
    if not stimuli:
        print("No stimuli found.")
        return

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
    
    win_matrix.to_csv(os.path.join(output_dir, f"{file_prefix}_win_matrix.csv"))
    
    long_df = build_long_format(subset, winner_col)
    if long_df.empty:
        print("No valid trials for modeling.")
        return
    scores, se, cov_scores, result = fit_bradley_terry_glmm(long_df, stimuli)
    print("BT Mixed Model Scores:")
    print(scores)
    pairs = pairwise_wald(scores, cov_scores)
    p_table = pd.DataFrame(np.nan, index=stimuli, columns=stimuli)
    for a, b, p in pairs:
        p_table.loc[a, b] = p
        p_table.loc[b, a] = p
    print("Pairwise p-values (Wald):")
    print(p_table)
    return scores, se, cov_scores

def main():
    base_dir = r"d:\Data\OneDrive\Papers\SWIM\Codes\Pilot Experiment"
    data_dir = os.path.join(base_dir, "Data")
    output_dir = os.path.join(base_dir, "Analysis")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading data...")
    df = load_data(data_dir)
    
    if df.empty:
        print("No data loaded. Exiting.")
        return

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

    save_combined_plot(res_intensity, res_clarity, os.path.join(output_dir, "Combined_Ranking.png"))

if __name__ == "__main__":
    main()
