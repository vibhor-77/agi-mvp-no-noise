"""
Symbolic Regression MVP  —  Beam Search + Parallel edition
-----------------------------------------------------------
Fitness = MSE  +  λ · tree_size   (Occam / MDL)

Beam search: maintain the top BEAM_SIZE candidates at all times.
Each generation every beam member spawns OFFSPRING_PER_BEAM mutations;
all candidates are evaluated in parallel across N_WORKERS processes.

Usage:
    python guess_function.py data.csv [options]

Options  (all optional, defaults shown):
    --beam      20       number of elite candidates to keep
    --offspring 60       mutations generated per beam member per generation
    --workers   <ncpus>  parallel worker processes
    --gens      600      max generations
    --lambda    0.001    complexity penalty per node (Occam weight)
"""

import csv, math, random, copy, sys, operator, argparse, time
from multiprocessing import Pool, cpu_count

# ══════════════════════════════════════════════════════════════════════════════
# Primitives — must be named (not lambda) so multiprocessing can pickle them
# ══════════════════════════════════════════════════════════════════════════════

def _div (a, b): return a / b if abs(b) > 1e-10 else 1.0
def _sqrt(x):    return math.sqrt(abs(x))
def _log (x):    return math.log(abs(x)) if abs(x) > 1e-10 else 0.0
def _exp (x):    return math.exp(min(x, 50))
def _sq  (x):    return x * x
def _cube(x):    return x * x * x

BINARY      = {'add': operator.add, 'sub': operator.sub,
               'mul': operator.mul, 'div': _div}
UNARY       = {'sin': math.sin, 'cos': math.cos, 'sq': _sq,
               'cube': _cube,   'sqrt': _sqrt,   'log': _log, 'exp': _exp}
BINARY_KEYS = list(BINARY.keys())
UNARY_KEYS  = list(UNARY.keys())

# ══════════════════════════════════════════════════════════════════════════════
# Function tree
# ══════════════════════════════════════════════════════════════════════════════

class Node:
    __slots__ = ('op', 'children', 'value')

    def __init__(self, op, children=None, value=None):
        self.op       = op
        self.children = children or []
        self.value    = value

    def eval(self, x):
        op = self.op
        if   op == 'x':     return x
        elif op == 'const': return self.value
        elif op in UNARY:   return UNARY[op](self.children[0].eval(x))
        else:               return BINARY[op](self.children[0].eval(x),
                                              self.children[1].eval(x))

    def size(self):
        """Node count — proxy for Kolmogorov complexity."""
        return 1 + sum(c.size() for c in self.children)

    def __str__(self):
        op = self.op
        if   op == 'x':     return 'x'
        elif op == 'const': return f'{self.value:.3f}'
        elif op in UNARY:   return f'{op}({self.children[0]})'
        else:
            sym = {'add':'+', 'sub':'-', 'mul':'*', 'div':'/'}
            return f'({self.children[0]} {sym[op]} {self.children[1]})'

# ══════════════════════════════════════════════════════════════════════════════
# Random tree generation & mutation
# ══════════════════════════════════════════════════════════════════════════════

def random_tree(depth=0, max_depth=4):
    if depth >= max_depth or (depth > 1 and random.random() < 0.35):
        return (Node('x') if random.random() < 0.5
                else Node('const', value=random.uniform(-5, 5)))
    if random.random() < 0.4:
        return Node(random.choice(UNARY_KEYS),
                    [random_tree(depth + 1, max_depth)])
    return Node(random.choice(BINARY_KEYS),
                [random_tree(depth + 1, max_depth),
                 random_tree(depth + 1, max_depth)])

def mutate(tree, prob=0.15):
    if random.random() < prob:
        return random_tree(max_depth=3)          # swap entire subtree
    if tree.op == 'const':
        tree.value += random.gauss(0, 0.3)       # nudge constant
    tree.children = [mutate(c, prob) for c in tree.children]
    return tree

# ══════════════════════════════════════════════════════════════════════════════
# Fitness (Occam / MDL) — worker-process side
# ══════════════════════════════════════════════════════════════════════════════

_POINTS     = None   # set once per worker via initializer — avoids re-sending
_LAMBDA     = None

def _worker_init(points, lam):
    """Called once when each worker process starts."""
    global _POINTS, _LAMBDA
    _POINTS = points
    _LAMBDA = lam
    random.seed()           # each worker needs its own RNG seed

def _eval_fitness(tree):
    """Runs inside a worker process."""
    total = 0.0
    for x, y in _POINTS:
        try:
            pred = tree.eval(x)
            if not math.isfinite(pred):
                return float('inf')
            total += (pred - y) ** 2
        except Exception:
            return float('inf')
    error = total / len(_POINTS)
    return error + _LAMBDA * tree.size()

def _mse_only(tree, points):
    total = 0.0
    for x, y in points:
        try:
            pred = tree.eval(x)
            if not math.isfinite(pred): return float('inf')
            total += (pred - y) ** 2
        except Exception:
            return float('inf')
    return total / len(points)

# ══════════════════════════════════════════════════════════════════════════════
# Beam search
# ══════════════════════════════════════════════════════════════════════════════

def evolve(points, beam_size, offspring_per, n_workers, n_gens, lam):
    print(f"  workers={n_workers}  beam={beam_size}  "
          f"offspring/beam={offspring_per}  "
          f"candidates/gen={beam_size * offspring_per + beam_size}  "
          f"λ={lam}\n")

    # Seed beam with random trees
    beam = [random_tree() for _ in range(beam_size)]

    best_tree     = None
    best_fitness  = float('inf')
    best_mse      = float('inf')
    t0            = time.time()

    with Pool(n_workers, initializer=_worker_init,
              initargs=(points, lam)) as pool:

        for gen in range(n_gens):

            # ── Expand: each beam member spawns offspring_per mutations ────────
            candidates = list(beam)                             # keep parents too
            for parent in beam:
                for _ in range(offspring_per):
                    candidates.append(mutate(copy.deepcopy(parent)))

            # ── Evaluate all candidates in parallel ────────────────────────────
            scores = pool.map(_eval_fitness, candidates)

            # ── Select top beam_size by Occam fitness ──────────────────────────
            ranked = sorted(zip(scores, candidates), key=lambda z: z[0])
            beam   = [t for _, t in ranked[:beam_size]]

            # ── Track global best ──────────────────────────────────────────────
            top_fitness, top_tree = ranked[0]
            if top_fitness < best_fitness:
                best_fitness = top_fitness
                best_tree    = top_tree
                best_mse     = _mse_only(best_tree, points)
                elapsed      = time.time() - t0
                print(f"  gen {gen:4d} │ {elapsed:6.1f}s │ "
                      f"nodes {best_tree.size():2d} │ "
                      f"MSE {best_mse:.8f} │ "
                      f"f(x) = {best_tree}")

            if best_mse < 1e-8:
                print(f"\n  ✓ Converged at generation {gen}")
                break

            # ── Show beam diversity every 50 gens ─────────────────────────────
            if gen % 50 == 49:
                print(f"\n  ── Beam at gen {gen+1} ──")
                for rank, (sc, t) in enumerate(ranked[:5]):
                    print(f"    #{rank+1}  nodes={t.size():2d}  fitness={sc:.6f}  {t}")
                print()

    return best_tree, best_mse

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        return [(float(r[0]), float(r[1])) for r in reader]

def parse_args():
    p = argparse.ArgumentParser(description='Symbolic regression via beam search')
    p.add_argument('csv',            help='Input CSV file with x,y columns')
    p.add_argument('--beam',         type=int,   default=20,          help='Beam size (default 20)')
    p.add_argument('--offspring',    type=int,   default=60,          help='Offspring per beam member (default 60)')
    p.add_argument('--workers',      type=int,   default=cpu_count(), help=f'Worker processes (default {cpu_count()})')
    p.add_argument('--gens',         type=int,   default=600,         help='Max generations (default 600)')
    p.add_argument('--lam',          type=float, default=0.001,       help='Complexity penalty λ (default 0.001)')
    return p.parse_args()

if __name__ == '__main__':
    args   = parse_args()
    points = load_csv(args.csv)

    print(f"Loaded {len(points)} points from '{args.csv}'\n")

    best, mse_val = evolve(
        points,
        beam_size    = args.beam,
        offspring_per= args.offspring,
        n_workers    = args.workers,
        n_gens       = args.gens,
        lam          = args.lam,
    )

    print(f"\n{'═'*60}")
    print(f"  Best guess : f(x) = {best}")
    print(f"  Nodes      : {best.size()}  (Kolmogorov proxy)")
    print(f"  Final MSE  : {mse_val:.10f}")
    print(f"{'═'*60}")
