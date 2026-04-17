#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import warnings
import numpy as np
import pandas as pd
import concurrent.futures
import time
from tqdm import tqdm

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.population import Population
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.indicators.hv import HV
    from sklearn.metrics import f1_score, precision_score, recall_score
except ImportError:
    print("Error: Missing dependencies.")
    print("Run: pip install pymoo scikit-learn pandas tqdm")
    sys.exit(1)

warnings.filterwarnings("ignore")

DATASET_ROOT = "../data/dataset2"
OUTPUT_FILE = "../tables_and_figures/qimig_final_results.csv"

HYPERPARAMS = {
    'POP_SIZE': 100,
    'N_EVALS': 50000,
    'Q_LEARNING_RATE': 0.05,
}
N_RUNS = 30

class QuantumModule:
    def __init__(self, n_var):
        self.n_var = n_var
        self.theta = np.full(n_var, np.pi / 4)

    def observe(self, n_samples):
        probabilities = np.sin(self.theta) ** 2
        rand_matrix = np.random.random((n_samples, self.n_var))
        return (rand_matrix < probabilities).astype(bool)

    def update_rotation_gate(self, best_solution):
        target = best_solution.astype(int)
        current_p = np.sin(self.theta) ** 2
        diff = target - current_p
        delta = HYPERPARAMS['Q_LEARNING_RATE'] * np.sign(diff)
        self.theta += delta
        self.theta = np.clip(self.theta, 0.01, np.pi / 2 - 0.01)

class QIMIG_Algorithm(NSGA2):
    def __init__(self, problem, **kwargs):
        super().__init__(
            pop_size=HYPERPARAMS['POP_SIZE'],
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(prob=0.9),
            mutation=BitflipMutation(prob=0.05),
            eliminate_duplicates=True,
            **kwargs
        )
        self.quantum_engine = QuantumModule(problem.n_var)
        self.problem_instance = problem

    def _initialize_infill(self):
        X = self.quantum_engine.observe(self.pop_size)
        X = self._repair(X)
        return Population.new(X=X)

    def _infill(self):
        n_quantum = int(self.pop_size * 0.7)
        n_genetic = self.pop_size - n_quantum

        X_q = self.quantum_engine.observe(n_quantum)
        X_q = self._repair(X_q)
        off_q = Population.new(X=X_q)

        if n_genetic > 0:
            off_g = super()._infill()
            if len(off_g) > n_genetic: off_g = off_g[:n_genetic]
            off_g.set("X", self._repair(off_g.get("X")))
            return Population.merge(off_q, off_g)
        return off_q

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills, **kwargs)
        if len(self.opt) > 0:
            best_idx = np.argmin(self.opt.get("F")[:, 0])
            leader = self.opt[best_idx].X
            self.quantum_engine.update_rotation_gate(leader)

    def _repair(self, X):
        W = self.problem_instance.W
        P = self.problem_instance.P
        Cap = self.problem_instance.Capacity
        W_safe = np.where(W == 0, 1e-6, W)
        ratio = P / W_safe

        for i in range(len(X)):
            current_w = np.sum(X[i] * W)
            if current_w > Cap:
                selected = np.where(X[i])[0]
                sorted_idx = selected[np.argsort(ratio[selected])]
                for idx in sorted_idx:
                    if current_w <= Cap: break
                    X[i, idx] = False
                    current_w -= W[idx]
        return X

class CustomMultiObjKnapsack(Problem):
    def __init__(self, n_items, W, P):
        super().__init__(n_var=n_items, n_obj=2, n_ieq_constr=1, vtype=bool)
        self.W = np.array(W)
        self.P = np.array(P)
        self.Capacity = np.sum(self.W) * 0.5

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = -np.sum(self.P * x, axis=1)
        f2 = np.sum(x, axis=1)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = (np.sum(self.W * x, axis=1) - self.Capacity).reshape(-1, 1)

def create_problem(filename):
    if filename is None: return None
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            n_items_m = re.search(r"(\d+)\s+items", content)
            if not n_items_m: return None
            n_items = int(n_items_m.group(1))

            profits = np.zeros(n_items)
            weights = np.ones(n_items)

            lines = content.split('\n')
            curr_idx = -1
            for line in lines:
                im = re.search(r"item:\s*(\d+)", line)
                if im: curr_idx = int(im.group(1)) - 1

                pm = re.search(r"profit:\s*\+?\-?(\d+\.?\d*)", line)
                if pm and 0 <= curr_idx < n_items: profits[curr_idx] = float(pm.group(1))

                wm = re.search(r"weight:\s*(\d+\.?\d*)", line)
                if wm and 0 <= curr_idx < n_items: weights[curr_idx] = float(wm.group(1))

            return CustomMultiObjKnapsack(n_items, weights, profits)
    except:
        return None

def find_ground_truth(input_path):
    curr = os.path.dirname(os.path.abspath(input_path))
    for _ in range(3):
        p = os.path.join(curr, "knapsack_file_grndtrth.txt")
        if os.path.exists(p): return p
        p2 = os.path.join(curr, "groundtruth", "knapsack_file_grndtrth.txt")
        if os.path.exists(p2): return p2
        curr = os.path.dirname(curr)
    return None

def calc_metrics(res, y_true, problem, time_elapsed):
    solutions = np.atleast_2d(res.X)
    limit = min(len(y_true), problem.n_var)
    y_true_trunc = y_true[:limit]

    precs, recs, f1s = [], [], []
    for sol in solutions:
        y_pred = sol.astype(int)[:limit]
        precs.append(precision_score(y_true_trunc, y_pred, zero_division=0))
        recs.append(recall_score(y_true_trunc, y_pred, zero_division=0))
        f1s.append(f1_score(y_true_trunc, y_pred, average='macro', zero_division=0))

    mean_p = np.mean(precs)
    mean_r = np.mean(recs)
    mean_f1 = np.mean(f1s)

    gt_f1 = -np.sum(problem.P[:limit] * y_true_trunc)
    gt_f2 = np.sum(y_true_trunc)

    F = res.F
    ideal = np.min(F, axis=0)
    nadir = np.max(F, axis=0)
    denom = nadir - ideal
    denom[denom == 0] = 1e-9

    F_norm = (F - ideal) / denom
    gt_norm = (np.array([gt_f1, gt_f2]) - ideal) / denom

    ed = np.mean(np.sqrt(np.sum((F_norm - gt_norm) ** 2, axis=1)))

    try:
        ref_point = np.array([1.1, 1.1])
        hv = HV(ref_point=ref_point)(F_norm)
    except:
        hv = 0.0

    return mean_p, mean_r, mean_f1, ed, hv, time_elapsed

def process_single_run(args):
    file_path, rule_id, run_id = args
    try:
        problem = create_problem(file_path)
        gt_path = find_ground_truth(file_path)
        if problem is None or not gt_path: return None

        with open(gt_path, 'r') as f:
            clean = re.sub(r"[\n\s\[\]]", "", f.read())
            y_true = np.array([int(c) for c in clean if c in '01'])

        algorithm = QIMIG_Algorithm(problem)
        termination = get_termination("n_eval", HYPERPARAMS['N_EVALS'])

        start_time = time.time()
        res = minimize(problem, algorithm, termination, seed=42 + run_id, verbose=False)
        end_time = time.time()
        duration = end_time - start_time

        if res.X is None: return None

        p, r, f1, ed, hv, t = calc_metrics(res, y_true, problem, duration)

        return {
            "Rule": rule_id,
            "Run": run_id,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "ED": ed,
            "HV": hv,
            "Time": t
        }
    except Exception as e:
        return None

def main():
    print(f"QIMIG SOLO RUN | Metrics: P, R, F1, ED, HV, Time")

    files = glob.glob(os.path.join(DATASET_ROOT, "**", "knapsack_file"), recursive=True)
    target_files = [f for f in files if "input_CO" in f and "groundtruth" not in f]

    tasks = []
    for f in target_files:
        m = re.search(r"migrationRule(\d+)", f)
        if not m: continue
        rid = f"migrationRule{m.group(1)}"

        for r in range(N_RUNS):
            tasks.append((f, rid, r))

    print(f"Scheduled {len(tasks)} runs. Running QIMIG...")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(2, os.cpu_count())) as exc:
        futures = {exc.submit(process_single_run, t): t for t in tasks}

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Running", ncols=100):
            res = fut.result()
            if res:
                results.append(res)

    if len(results) > 0:
        df = pd.DataFrame(results)
        df.sort_values(by=["Rule", "Run"], inplace=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nDone! Results saved to: {OUTPUT_FILE}")
        print(f"Preview:\n{df.head()}")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    main()
