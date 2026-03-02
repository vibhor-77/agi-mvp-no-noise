"""
Generate a CSV of (x0, ..., xn, y) points from a secret function.
The number of input variables is auto-detected from SECRET_FN's signature.

Single variable  → column named  x   (backward compatible)
Multiple variables → columns named  x0, x1, x2, ...

Edit SECRET_FN to change what the guesser has to find.

Usage:
    python make_data.py             # writes data.csv
    python make_data.py out.csv
"""

import csv, math, random, sys, inspect

# ── Define your secret function here ─────────────────────────────────────────
#
# Single variable (original):
def SECRET_FN(x):
    return math.sin(x ** 2) + 2 * x

# Multivariate example — uncomment to try:
# def SECRET_FN(x0, x1):
#     return math.sin(x0 ** 2) + 2 * x0 * x1
#
# def SECRET_FN(x0, x1, x2):
#     return x0 ** 2 + math.cos(x1 * x2)

# ── Config ────────────────────────────────────────────────────────────────────

N      = 40       # number of data points
X_MIN  = -3.0
X_MAX  =  3.0
SEED   = 42

# ── Auto-detect arity from function signature ─────────────────────────────────

n_vars = len(inspect.signature(SECRET_FN).parameters)

# Single-variable keeps column name 'x' for backward compatibility with
# guess_function.py; multivariate uses x0, x1, ...
if n_vars == 1:
    var_names = ['x']
else:
    var_names = [f'x{i}' for i in range(n_vars)]

# ── Generate points ───────────────────────────────────────────────────────────

random.seed(SEED)
path = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'

with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(var_names + ['y'])
    for _ in range(N):
        xs = [round(X_MIN + random.random() * (X_MAX - X_MIN), 6) for _ in range(n_vars)]
        y  = SECRET_FN(*xs)
        writer.writerow(xs + [round(y, 6)])

fn_src = inspect.getsource(SECRET_FN).strip().splitlines()[1].strip()
print(f"Wrote {N} points to '{path}'")
print(f"  variables : {var_names}")
print(f"  function  : {fn_src}")
