"""
Generate a CSV of (x, y) points from a secret function.
Edit SECRET_FN to change what the guesser has to find.

Usage:
    python make_data.py          # writes data.csv
    python make_data.py out.csv
"""

import csv, math, sys

# ── Change this to whatever you want the guesser to discover ──────────────────
def SECRET_FN(x):
    return math.sin(x ** 2) + 2 * x

# ─────────────────────────────────────────────────────────────────────────────

N      = 40          # number of points
X_MIN  = -3.0
X_MAX  =  3.0
path   = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'

xs = [X_MIN + (X_MAX - X_MIN) * i / (N - 1) for i in range(N)]

with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for x in xs:
        writer.writerow([round(x, 6), round(SECRET_FN(x), 6)])

print(f"Wrote {N} points to '{path}'  (secret: sin(x²) + 2x)")
