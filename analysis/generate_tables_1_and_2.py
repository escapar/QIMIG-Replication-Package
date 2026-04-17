import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, norm, kruskal
import os
import warnings

warnings.filterwarnings('ignore')

INPUT_FILE = '../data/main_experiment_results.csv'
TARGET_ALGO = 'QIMIG'
BASELINE_ALGO = 'AUTHOR_UNSGA3'

ALGO_NAME_MAP = {
    'AUTHOR_UNSGA3': 'Deshpande et al.',
    'NSGA2': 'NSGA-II',
    'MOEAD': 'MOEA/D',
    'random': 'Random Search'
}

RULE_MAP = {
    'migrationRule1': r'\texttt{c-logging} $\to$ \texttt{slf4j}',
    'migrationRule2': r'\texttt{slf4j} $\to$ \texttt{log4j}',
    'migrationRule3': r'\texttt{easymock} $\to$ \texttt{mockito}',
    'migrationRule4': r'\texttt{g-collect} $\to$ \texttt{guava}',
    'migrationRule5': r'\texttt{gson} $\to$ \texttt{jackson}',
    'migrationRule6': r'\texttt{json} $\to$ \texttt{gson}',
    'migrationRule7': r'\texttt{c-lang} $\to$ \texttt{guava}',
    'migrationRule8': r'\texttt{testng} $\to$ \texttt{junit}',
    'migrationRule9': r'\texttt{json-simple} $\to$ \texttt{gson}'
}

METRICS = ['F1', 'Recall', 'Precision', 'HV', 'ED', 'Time']

def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    m, n = len(x), len(y)
    if m == 0 or n == 0: return 0.0
    mat = np.sign(np.subtract.outer(x, y))
    return np.sum(mat) / (m * n)

def get_cliffs_label(delta):
    d = abs(delta)
    if d < 0.147: return 'Negligible'
    if d < 0.33:  return 'Small'
    if d < 0.474: return 'Medium'
    return 'Large'

def calculate_rosenthal_r(p_val, n_pairs):
    if p_val >= 1.0: return 0.0
    z_stat = abs(norm.ppf(p_val / 2))
    return z_stat / np.sqrt(2 * n_pairs)

def load_data(file_path):
    df_check = pd.read_csv(file_path, nrows=5)
    if not any(col in df_check.columns for col in ['Algorithm', 'F1', 'Rule']):
        names = ['Rule', 'Instance', 'Algorithm', 'Run', 'Type', 'Precision', 'Recall', 'F1', 'HV', 'ED', 'Time']
        df = pd.read_csv(file_path, header=None, names=names[:len(df_check.columns)])
    else:
        df = pd.read_csv(file_path)

    df['Algorithm'] = df['Algorithm'].replace(ALGO_NAME_MAP)

    for col in ['Time', 'HV', 'ED']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    return df

def generate_paper_report(df):
    target_name = ALGO_NAME_MAP.get(TARGET_ALGO, TARGET_ALGO)
    base_name = ALGO_NAME_MAP.get(BASELINE_ALGO, BASELINE_ALGO)

    global_means = df.groupby('Algorithm')[METRICS].mean()
    qimig_f1 = global_means.loc[target_name, 'F1'] if target_name in global_means.index else 0
    base_f1 = global_means.loc[base_name, 'F1'] if base_name in global_means.index else 0

    print("=" * 60)
    print("1. TEXT PLACEHOLDERS (For Section 4.1.1)")
    print("=" * 60)
    print(f"-> QIMIG peak Global Mean F1-score: {qimig_f1:.4f}")
    print(f"-> Baseline ({base_name}) Global Mean F1-score: {base_f1:.4f}")

    print("\n" + "=" * 60)
    print("2. LATEX CODE: TABLE 1 (Global Ranking with Multi-Objective Metrics)")
    print("=" * 60)

    global_ranked = global_means.sort_values('F1', ascending=False)

    best_hv = global_ranked['HV'].max()
    valid_ed = global_ranked['ED'][global_ranked['ED'] > 0]
    best_ed = valid_ed.min() if not valid_ed.empty else 0.0

    latex_tab1 = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Global Performance of Algorithms.}",
        r"\label{tab:global_ranking}",
        r"\small",
        r"\begin{tabular}{l c c c c c r}",
        r"\toprule",
        r"\textbf{Algorithm} & \textbf{Mean F1} & \textbf{Recall} & \textbf{Precision} & \textbf{HV} & \textbf{ED} & \textbf{Time (s)} \\",
        r"\midrule"
    ]

    for algo, row in global_ranked.iterrows():
        f1_str = f"\\textbf{{{row['F1']:.4f}}}" if algo == target_name else f"{row['F1']:.4f}"
        prec_str = f"\\textbf{{{row['Precision']:.4f}}}" if algo == target_name else f"{row['Precision']:.4f}"
        
        hv_str = f"\\textbf{{{row['HV']:.4f}}}" if row['HV'] == best_hv and best_hv > 0 else f"{row['HV']:.4f}"
        ed_str = f"\\textbf{{{row['ED']:.4f}}}" if row['ED'] == best_ed and best_ed > 0 else f"{row['ED']:.4f}"

        if algo == target_name:
            latex_tab1.append(f"\\textbf{{{algo} (Ours)}} & {f1_str} & {row['Recall']:.4f} & {prec_str} & {hv_str} & {ed_str} & {row['Time']:.1f} \\\\")
        else:
            latex_tab1.append(f"{algo} & {f1_str} & {row['Recall']:.4f} & {prec_str} & {hv_str} & {ed_str} & {row['Time']:.1f} \\\\")

    latex_tab1.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    print("\n".join(latex_tab1))

    print("\n" + "=" * 60)
    print("3. LATEX CODE: TABLE 2 (Per Rule Challenger)")
    print("=" * 60)

    latex_tab2 = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{QIMIG vs. The Best Challenger (Per Migration Task)}",
        r"\label{tab:best_challenger}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{l l c c c}",
        r"\toprule",
        r"\textbf{Migration Task} & \textbf{Challenger Method} & \textbf{QIMIG} & \textbf{Challenger} & \textbf{Gap} \\",
        r"\midrule"
    ]

    rules = [r for r in df['Rule'].unique() if pd.notna(r)]
    for rule in sorted(rules):
        rule_data = df[df['Rule'] == rule]
        algo_means = rule_data.groupby('Algorithm')['F1'].mean()

        if target_name not in algo_means.index: continue
        q_score = algo_means[target_name]

        others = algo_means.drop(target_name, errors='ignore')
        if others.empty: continue

        challenger_name = others.idxmax()
        c_score = others.max()
        gap = q_score - c_score

        task_name = RULE_MAP.get(rule, str(rule).replace('_', r'\_'))
        q_str = f"\\textbf{{{q_score:.4f}}}" if q_score >= c_score else f"{q_score:.4f}"
        c_str = f"\\textbf{{{c_score:.4f}}}" if c_score > q_score else f"{c_score:.4f}"
        gap_str = f"+{gap:.4f}" if gap > 0 else f"{gap:.4f}"

        latex_tab2.append(f"{task_name} & {challenger_name} & {q_str} & {c_str} & {gap_str} \\\\")

    latex_tab2.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    print("\n".join(latex_tab2))

    print("\n" + "=" * 60)
    print("4. STATISTICAL SIGNIFICANCE (QIMIG vs Baseline)")
    print("=" * 60)

    for metric in ['F1', 'Precision', 'Recall', 'HV', 'ED']:
        df_pivot = df.groupby(['Instance', 'Algorithm'])[metric].mean().unstack().dropna()
        if target_name not in df_pivot.columns or base_name not in df_pivot.columns:
            continue

        target_vals = df_pivot[target_name].values
        base_vals = df_pivot[base_name].values

        alt_hyp = 'less' if metric == 'ED' else 'greater'

        try:
            w_stat, p_wilcoxon = wilcoxon(target_vals, base_vals, alternative=alt_hyp)
            r_val = calculate_rosenthal_r(p_wilcoxon, len(target_vals))
            k_stat, p_kruskal = kruskal(target_vals, base_vals)
            c_delta = cliffs_delta(target_vals, base_vals)
            c_label = get_cliffs_label(c_delta)

            print(f"--- Metric: {metric} {'(Smaller is better)' if metric == 'ED' else '(Larger is better)'} ---")
            print(f"Wilcoxon P-val : {p_wilcoxon:.4e} (Rosenthal r: {r_val:.4f})")
            print(f"Kruskal P-val  : {p_kruskal:.4e}")
            print(f"Cliff's Delta  : {c_delta:.4f} ({c_label})")
            print("-" * 30)
        except ValueError as e:
            print(f"--- Metric: {metric} ---")
            print(f"Cannot compute stats: {e}")
            print("-" * 30)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    df = load_data(INPUT_FILE)
    if 'Type' in df.columns:
        df = df[df['Type'] != 'SENS'].copy()

    generate_paper_report(df)

if __name__ == '__main__':
    main()
