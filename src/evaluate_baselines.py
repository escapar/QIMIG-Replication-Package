#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import warnings
import time
import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing
import sklearn.metrics
from pathlib import Path

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATASET_ROOT = "../data/dataset2"
OUTPUT_FILE = "../tables_and_figures/baselines_evaluation_results.csv"
HEADERS = ['Rule', 'Instance', 'Algorithm', 'Run', 'Type', 'Precision', 'Recall', 'F1', 'HV', 'ED', 'Time']

N_RUNS = 30
N_EVALS = 100000
MAX_CORES = 30

TPE_POP = 200
TPE_LR = 0.43636187731135556
TPE_RATIO = 0.5099379433242419
TPE_GREEDY_RATIO = 0.41505435376646915


class CustomMultiObjKnapsack(Problem):
    def __init__(self, n_items, W, P):
        super().__init__(n_var=n_items, n_obj=2, vtype=bool)
        self.W = np.array(W)
        self.P = np.array(P)
        self.Capacity = np.sum(self.W) * 0.5

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = - np.sum(self.P * x, axis=1)
        f2 = np.sum(x, axis=1)
        out["F"] = np.column_stack([f1, f2])


class QuantumEngine:
    def __init__(self, n_var, lr):
        self.n_var = n_var
        self.lr = lr
        self.theta = np.full(n_var, np.pi / 4)

    def observe(self, n_samples):
        probs = np.sin(self.theta) ** 2
        return (np.random.random((n_samples, self.n_var)) < probs).astype(bool)

    def update(self, best_sol):
        target = best_sol.astype(int)
        diff = target - (np.sin(self.theta) ** 2)
        self.theta += self.lr * np.sign(diff)
        self.theta = np.clip(self.theta, 0.01, np.pi / 2 - 0.01)


class HybridQuantumUNSGA3(UNSGA3):
    def __init__(self, problem, pop_size, q_lr, q_ratio, greedy_ratio, **kwargs):
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        super().__init__(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
            **kwargs
        )
        self.q_ratio = q_ratio
        self.greedy_ratio = greedy_ratio
        self.problem_inst = problem
        self.qe = QuantumEngine(problem.n_var, lr=q_lr)

    def _initialize_infill(self):
        n_ratio = int(self.pop_size * (self.greedy_ratio / 2.0))
        n_profit = int(self.pop_size * (self.greedy_ratio / 2.0))
        n_quantum = self.pop_size - n_ratio - n_profit

        W = self.problem_inst.W
        P = self.problem_inst.P
        Cap = self.problem_inst.Capacity

        ratio = P / (np.where(W == 0, 1e-6, W) + 1e-9)
        X_ratio = self._generate_greedy(n_ratio, ratio, W, Cap)
        X_profit = self._generate_greedy(n_profit, P, W, Cap)
        X_quantum = self._repair(self.qe.observe(n_quantum))

        return Population.new(X=np.vstack([X_ratio, X_profit, X_quantum]))

    def _generate_greedy(self, n, score_metric, W, Cap):
        X = np.zeros((n, self.problem_inst.n_var), dtype=bool)
        for i in range(n):
            noisy_score = score_metric * np.random.uniform(0.95, 1.05, size=len(score_metric))
            order = np.argsort(noisy_score)[::-1]
            cw = 0
            for idx in order:
                if cw + W[idx] <= Cap:
                    X[i, idx] = True
                    cw += W[idx]
        return X

    def _infill(self):
        n_q = int(self.pop_size * self.q_ratio)
        X_q = self._repair(self.qe.observe(n_q))
        off_g = super()._infill()
        if off_g is not None and len(off_g) > 0:
            n_g = self.pop_size - n_q
            return Population.merge(Population.new(X=X_q), off_g[:n_g])
        return Population.new(X=X_q)

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills, **kwargs)
        if len(self.opt) > 0:
            best_idx = np.argmin(self.opt.get("F")[:, 0])
            self.qe.update(self.opt[best_idx].X)

    def _repair(self, X):
        W, P, Cap = self.problem_inst.W, self.problem_inst.P, self.problem_inst.Capacity
        ratio = P / (np.where(W == 0, 1e-6, W) + 1e-9)
        mean_ratio = np.mean(ratio)

        for i in range(len(X)):
            cw = np.sum(X[i] * W)

            if cw > Cap:
                sel = np.where(X[i])[0]
                order = np.argsort(ratio[sel])
                for idx_rel in order:
                    idx = sel[idx_rel]
                    if cw <= Cap: break
                    X[i, idx] = False
                    cw -= W[idx]

            if cw < Cap:
                unsel = np.where(~X[i])[0]
                order_fill = np.argsort(ratio[unsel])[::-1]
                for idx_rel in order_fill:
                    idx = unsel[idx_rel]

                    if ratio[idx] < mean_ratio:
                        continue

                    if cw + W[idx] <= Cap:
                        X[i, idx] = True
                        cw += W[idx]

                    if Cap - cw < 1e-5: break
        return X


def create_mo_problem_from_file_with_log(filename):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    n = 0
    for l in lines:
        if "knapsacks" in l:
            match = re.findall(r"\d+", l)
            if len(match) > 1: n = int(match[1])
            break

    if n == 0:
        raise ValueError(f"Parse error: {filename}")

    wp = pd.DataFrame(np.zeros((n, 2)), columns=["p", "w"])
    wp.loc[:, "w"] = 1.0

    profit_sum = 0
    for l in lines:
        if "item" in l:
            match = re.findall(r"\d+", l)
            if match: idx = int(match[0]) - 1
        if "profit" in l:
            val_match = re.findall(r"\+?\-?\d+\.?\d*", l)
            if val_match:
                val = float(val_match[0])
                wp.iloc[idx, 0] = val
                profit_sum += val

    return CustomMultiObjKnapsack(n, wp["w"].tolist(), wp["p"].tolist()), profit_sum


def load_ground_truth_aligned(fpath):
    p = Path(fpath).resolve()
    gt_path = None
    for parent in p.parents:
        cand = parent / "groundtruth" / "knapsack_file_grndtrth.txt"
        if cand.exists(): gt_path = cand; break
        if parent.name == "dataset2": break
    if gt_path is None:
        raise FileNotFoundError(f"Groundtruth not found for: {fpath}")
    with open(gt_path, "r") as f:
        return np.array([int(x) for x in re.sub(r"[^01]", "", f.read())])


def normalize_path(p):
    return str(p).replace("\\", "/").strip()


def collect_tasks():
    sys.stdout.write("Starting file collection and task construction phase...\n")
    all_files = glob.glob(os.path.join(DATASET_ROOT, "**", "knapsack_file"), recursive=True)
    target_files = [f for f in all_files if "input_CO" in f and "grndtrth" not in f and "output" not in f]

    sys.stdout.write(f"Original files scanned: {len(target_files)}\n")

    tasks = []
    for f in target_files:
        rel = normalize_path(os.path.relpath(os.path.dirname(f), DATASET_ROOT))
        m_rule = re.search(r"(migrationRule\d+)", f)
        rid = m_rule.group(1) if m_rule else "Unknown"
        m_run = re.search(r"run_(\d+)", f)
        if not m_run:
            continue
        run_idx = int(m_run.group(1))
        tasks.append((f, rid, rel, run_idx, "QIMIG"))

    tasks.sort(key=lambda x: x[0])
    sys.stdout.write(f"File collection complete. Total tasks built: {len(tasks)}\n")
    return tasks


def worker(args):
    fpath, rid, inst_id, run_id, algo = args

    prob, total_profit = create_mo_problem_from_file_with_log(fpath)
    if total_profit == 0:
        raise ValueError("Profit is 0")

    sys.stdout.write(f"Starting instance: {inst_id} | Run: {run_id}\n")

    yt = load_ground_truth_aligned(fpath)
    term = get_termination("n_eval", N_EVALS)
    seed = 42 + run_id

    algo_inst = HybridQuantumUNSGA3(
        prob,
        pop_size=TPE_POP,
        q_lr=TPE_LR,
        q_ratio=TPE_RATIO,
        greedy_ratio=TPE_GREEDY_RATIO
    )

    ts = time.time()
    res = minimize(prob, algo_inst, term, seed=seed, verbose=False)
    dur = time.time() - ts

    if res.X is None:
        raise RuntimeError(f"Optimization failed for {inst_id}")

    F_vals = np.atleast_2d(res.F)
    X_vals = np.atleast_2d(res.X)

    sorted_idx = np.argsort(F_vals[:, 0])
    keep_n = max(1, int(len(sorted_idx) * 0.05))
    best_X = X_vals[sorted_idx[:keep_n]]

    sols = best_X.astype(bool)

    p_l = [sklearn.metrics.precision_score(yt, s.astype(float), zero_division=0) for s in sols]
    r_l = [sklearn.metrics.recall_score(yt, s.astype(float), zero_division=0) for s in sols]
    f1_l = [sklearn.metrics.f1_score(yt, s.astype(float), average='macro', zero_division=0) for s in sols]

    mp = np.mean(p_l)
    mr = np.mean(r_l)
    mf1 = np.mean(f1_l)
    ed = np.sqrt((1 - mp) ** 2 + (1 - mr) ** 2)
    hv = mp * mr

    sys.stdout.write(
        f"Completed instance: {inst_id} | Run: {run_id} | F1: {mf1:.4f} | Precision: {mp:.4f} | Recall: {mr:.4f}\n")

    return {
        "Rule": rid, "Instance": inst_id, "Algorithm": "QIMIG", "Run": run_id,
        "Type": "INTEGRATED", "Precision": mp, "Recall": mr, "F1": mf1,
        "HV": hv, "ED": ed, "Time": dur
    }


def main():
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=HEADERS).to_csv(OUTPUT_FILE, index=False)

    tasks = collect_tasks()

    if len(tasks) > 0:
        sys.stdout.write('Starting parallel computing pool...\n')
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CORES) as ex:
            futures = {ex.submit(worker, args): args for args in tasks}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc='Processing'):
                res = fut.result()
                if res:
                    pd.DataFrame([res]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
