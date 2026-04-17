import pandas as pd
import numpy as np
import scipy.stats as stats
import re
import os

QF_FILE = "../data/qimig_30runs_results.csv"
BASE_FILE = "../data/baselines_30runs_results.csv"

EXCLUDE_KEYWORDS = ["NSGA", "nsga"]

def normalize_rule(x):
    s = str(x)
    m = re.search(r'(\d+)', s)
    if m: return f"migrationRule{m.group(1)}"
    return s

def get_significance_mark(p_val, mean_diff):
    if p_val >= 0.05: return "$\\approx$ (ns)"
    if mean_diff > 0:
        return "$\\blacktriangle$"
    else:
        return "$\\triangledown$"

def main():
    print("STATISTICAL ANALYSIS (NO NSGA MODE)...")

    if not os.path.exists(QF_FILE) or not os.path.exists(BASE_FILE):
        print("Error: Files not found.")
        return

    df_qf = pd.read_csv(QF_FILE)
    df_base = pd.read_csv(BASE_FILE)

    df_qf['Rule'] = df_qf['Rule'].apply(normalize_rule)
    df_base['Rule'] = df_base['Rule'].apply(normalize_rule)

    print(f"   Original Baseline Rows: {len(df_base)}")
    for kw in EXCLUDE_KEYWORDS:
        df_base = df_base[~df_base['Algorithm'].str.contains(kw, case=False, na=False)]
    print(f"   Rows after removing '{EXCLUDE_KEYWORDS}': {len(df_base)}")

    if df_base.empty:
        print("Error: Baseline is empty after filtering! Check your CSV.")
        return

    qf_metric = next((c for c in df_qf.columns if 'f1' in c.lower()), "F1")
    base_metric = next((c for c in df_base.columns if 'f1' in c.lower()), "F1")

    def get_rule_num(r_str):
        m = re.search(r'(\d+)', r_str)
        return int(m.group(1)) if m else 999

    rules = sorted(df_qf['Rule'].unique(), key=get_rule_num)

    means = df_base.groupby(['Rule', 'Algorithm'])[base_metric].max().reset_index()

    latex_rows = []

    print(f"\n{'Rule':<10} | {'QF':<6} | {'Best Alt. Alg':<12} | {'Base':<6} | {'Status'}")
    print("-" * 60)

    for rule in rules:
        qf_vals = df_qf[df_qf['Rule'] == rule][qf_metric].values
        qf_mean = np.mean(qf_vals)

        rule_means = means[means['Rule'] == rule]
        if rule_means.empty: continue

        best_row = rule_means.loc[rule_means[base_metric].idxmax()]
        best_alg = best_row['Algorithm']
        base_val = best_row[base_metric]

        try:
            diffs = qf_vals - base_val
            if np.all(diffs == 0):
                p_val = 1.0
            elif np.unique(qf_vals).size == 1:
                p_val = 0.0 if qf_mean > base_val else 1.0
            else:
                stat_res = stats.wilcoxon(diffs, alternative='greater')
                p_val = stat_res.pvalue
        except:
            p_val = 0.0 if qf_mean > base_val else 1.0

        wins = np.sum(qf_vals > base_val)
        ties = np.sum(qf_vals == base_val)
        sr = (wins + 0.5 * ties) / len(qf_vals)

        sig_mark = get_significance_mark(p_val, qf_mean - base_val)
        short_rule = rule.replace("migrationRule", "Rule")

        print(f"{short_rule:<10} | {qf_mean:.3f}  | {best_alg:<12} | {base_val:.3f}  | {sig_mark}")

        row = f"{short_rule} & {qf_mean:.3f} & {best_alg} & {base_val:.3f} & {p_val:.2e} & {sr:.2f} & {sig_mark} \\\\"
        latex_rows.append(row)

    print("\n" + "=" * 60)
    print("LATEX TABLE (NO NSGA)")
    print("=" * 60)
    print(r"\begin{table*}[ht]")
    print(r"\centering")
    print(r"\caption{Statistical Comparison: QIMIG vs. Best Alternative Baseline (Excluding NSGA variants)}")
    print(r"\label{tab:stats}")
    print(r"\begin{tabular}{l c l c c c c}")
    print(r"\toprule")
    print(r"\textbf{Rule} & \textbf{QIMIG} & \textbf{Best Baseline} & \textbf{Base F1} & \textbf{$p$-value} & \textbf{SR} & \textbf{Sig.} \\")
    print(r"\midrule")
    for row in latex_rows:
        print(row)
    print(r"\bottomrule")
    print(r"\multicolumn{7}{l}{\footnotesize Sig.: $\blacktriangle$=Win ($p<0.05$), $\triangledown$=Loss, $\approx$=Not Sig. Baseline excludes NSGA-II/III.} \\")
    print(r"\end{tabular}")
    print(r"\end{table*}")

if __name__ == "__main__":
    main()
