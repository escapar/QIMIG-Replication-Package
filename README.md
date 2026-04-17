# Replication Package for QIMIG

This repository contains the dataset, source code, and statistical analysis scripts required to reproduce the experiments and figures presented in the QIMIG (Quantum-Inspired Migration) manuscript.

## Detailed Directory Structure

Below is a granular breakdown of every file provided in this replication package and its specific function:

```text
replication/
├── generate_tables_and_figures.bat    # [Windows] 1-Click execution script to automatically generate all tables and figures.
├── generate_tables_and_figures.sh     # [Linux/Mac] 1-Click execution script to automatically generate all tables and figures.
├── requirements.txt                   # Dependency list specifying required Python libraries (e.g., pymoo, pandas).
│
├── data/                              # Datasets and Offline CSV Metrics
│   ├── dataset/                       # Raw API dependency graph datasets describing topological constraints.
│   ├── dataset2/                      # Refined/Normalized migration objective instances loaded by the core algorithms.
│   ├── main_experiment_results.csv    # Consolidated benchmarking logs capturing primary F1, Recall, Precision metrics.
│   ├── qimig_30runs_results.csv       # Granular output of QIMIG tested over 30 independent runs for statistical stability.
│   ├── baselines_30runs_results.csv   # Granular output of baseline algorithms over 30 runs for head-to-head comparison.
│   └── sensitivity_analysis_results.csv # Metrics collected using varying Alpha and N hyperparameters (used for RQ3 plots). 
│
├── src/                               # Algorithm Implementations
│   ├── qimig_main.py                  # Core implementation of the Quantum-Inspired Migration (QIMIG) logic.
│   └── evaluate_baselines.py          # Parallelized suite running state-of-the-art literature competitors (NSGA, AGE-MOEA, etc.).
│
├── analysis/                          # Data Interpretation and LaTeX Generators
│   ├── generate_tables_1_and_2.py     # Parses `main_experiment_results.csv` to format LaTeX matrices for Table 1 and Table 2.
│   ├── test_statistical_significance.py # Performs Wilcoxon Rank-Sum tests analyzing variance between QIMIG & baselines.
│   └── plot_sensitivity.py            # Generates `sensitivity_analysis_hybrid_pop.pdf` exposing correlation limits across scales.
│
└── tables_and_figures/                # Final Outputs
    # (Dynamically generated after running scripts. Below are the expected artifacts:)
    ├── sensitivity_analysis_hybrid_pop.pdf # High-resolution parameter variance vector graphic (RQ3).
    ├── output_table1_table2.txt       # LaTeX strings corresponding to global capability metrics.
    ├── output_significance.txt        # LaTeX significance comparison output matrices.
    └── output_sensitivity_text.txt    # Corpus/discourse texts answering hyper-variance limits.
```

## Environment Setup

The code is developed in Python 3.9+. We strongly recommend using a virtual environment. Install the necessary dependencies via:

```bash
pip install numpy pandas scipy scikit-learn pymoo matplotlib seaborn tqdm
```

*(You can also use `pip install -r requirements.txt` if you populate the root with specific package versions).*

## Reproduction Instructions

The reproduction is split into two phases: **1. Fast Offline Reproduction** (reproducing textual tables and figures from paper data) and **2. Core Algorithm Execution** (running the actual quantum-inspired search).

### Phase 1: Fast Output Verification (Tables & Figures)

**⭐ 1-Click Automated Generation (Recommended):**
For maximum convenience, you can generate all figures, tables, and statistics automatically using the provided batch/shell scripts at the root of `replication/`:

**Windows:**
```cmd
generate_tables_and_figures.bat
```
**Linux / macOS:**
```bash
./generate_tables_and_figures.sh
```

> **Expected Outcome:** All generated text files (containing ready-to-copy LaTeX syntax) and PDFs will be saved directly to the `tables_and_figures/` directory seamlessly.

#### Alternative: Manual Step-by-Step Execution
If you prefer running modules individually, navigate to the `analysis` directory:
*Note: Depending on your terminal, if you encounter layout misalignment, try prefixing your execute commands with `python -X utf8`.*

**1. Reproduce Table 1 (Global Performance) and Table 2 (Per Rule Challenger):**
```bash
cd analysis
python generate_tables_1_and_2.py
```
**2. Reproduce Statistical Significance Results:**
```bash
python test_statistical_significance.py
```
**3. Generate RQ3 Parameter Sensitivity Plot (Figure):**
```bash
python plot_sensitivity.py
```

### Phase 2: Re-running the Search Algorithms (Smoke Testing)

To execute the quantum-inspired optimization algorithms and rebuild the CSV raw metrics computationally from scratch:

**1. Run QIMIG Solo:**
```bash
cd src
python qimig_main.py
```
> Evaluates standalone performance limits of QIMIG via multi-point generation tracking.

**2. Run Comprehensive Baselines:**
```bash
python evaluate_baselines.py
```
> Evaluates benchmarking techniques parallelized securely over available cores.
