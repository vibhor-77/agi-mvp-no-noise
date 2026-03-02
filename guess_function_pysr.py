"""
Symbolic Regression via PySR
-----------------------------
PySR wraps a high-performance Julia backend (SymbolicRegression.jl) and
handles complexity penalties, algebraic simplification, and parallelism
automatically. It returns a Pareto front: one best expression per complexity
level, so you can see the accuracy/simplicity tradeoff clearly.

SETUP (one-time, on your Mac):
    pip install pysr
    python -c "import pysr; pysr.install()"   # downloads Julia + packages (~2 min)

Usage:
    python guess_function_pysr.py data.csv [options]

Options:
    --workers   <n>    parallel processes (default: all CPU cores)
    --iters     <n>    search iterations (default: 100)
    --maxsize   <n>    max expression complexity (default: 20)
    --lam       <f>    complexity penalty weight (default: 0.001)
"""

import csv, sys, argparse
from multiprocessing import cpu_count

import numpy as np
from pysr import PySRRegressor

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Symbolic regression via PySR')
    p.add_argument('csv',           help='CSV file with x,y columns')
    p.add_argument('--workers',     type=int,   default=cpu_count(), help=f'Parallel workers (default {cpu_count()})')
    p.add_argument('--iters',       type=int,   default=100,         help='Search iterations (default 100)')
    p.add_argument('--maxsize',     type=int,   default=20,          help='Max expression complexity (default 20)')
    p.add_argument('--lam',         type=float, default=0.001,       help='Complexity penalty λ (default 0.001)')
    return p.parse_args()

# ── Data ───────────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = [list(map(float, r)) for r in reader]
    var_names = header[:-1]           # everything except the last column ('y')
    X = np.array([[r[i] for i in range(len(var_names))] for r in rows])
    y = np.array([r[-1] for r in rows])
    return X, y, var_names

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args            = parse_args()
    X, y, var_names = load_csv(args.csv)

    print(f"Loaded {len(y)} points from '{args.csv}'  "
          f"(variables: {var_names})")
    print(f"workers={args.workers}  iters={args.iters}  "
          f"maxsize={args.maxsize}  λ={args.lam}\n")

    model = PySRRegressor(
        # ── Primitives (same set as our hand-rolled version) ───────────────────
        binary_operators  = ["+", "-", "*", "/"],
        unary_operators   = ["sin", "cos", "square", "cube", "sqrt", "log", "exp"],

        # ── Occam / MDL ────────────────────────────────────────────────────────
        # parsimony sets the per-node complexity penalty — equivalent to our λ
        parsimony         = args.lam,
        maxsize           = args.maxsize,

        # ── Search budget ──────────────────────────────────────────────────────
        niterations       = args.iters,
        populations       = args.workers * 3,   # more populations = more diversity
        procs             = args.workers,

        # ── Output ─────────────────────────────────────────────────────────────
        verbosity         = 1,
        progress          = True,
    )

    model.fit(X, y, variable_names=var_names)

    # PySR returns the full Pareto front — best expression at each complexity level
    print("\n" + "═" * 65)
    print("  Pareto front (accuracy vs. simplicity):\n")
    print(model)

    best = model.get_best()
    print("\n" + "─" * 65)
    print(f"  Best expression : {model.sympy()}")
    print(f"  Complexity      : {best['complexity']}")
    print(f"  Loss (MSE)      : {best['loss']:.2e}")
    print(f"  R² score        : {model.score(X, y):.10f}  (1.0 = perfect)")
    print("═" * 65)
