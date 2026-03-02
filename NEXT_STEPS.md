# Next Steps

*Concrete follow-on work, roughly ordered by proximity to the current code.*

---

## Short term — improve the hand-rolled version

**1. Decouple structure search from coefficient optimisation**

Currently constants are evolved by Gaussian mutation, which is slow. Instead:
after each structural mutation, run a fast local optimiser (e.g.
`scipy.optimize.minimize`) to fit the constants for that fixed structure. This
would make coefficient convergence near-instant and let the beam focus entirely
on structural search.

**2. Add algebraic simplification**

After evaluating a candidate, pass it through SymPy to canonicalise before
scoring its size. This fixes the `cos(π/2 + x²) ≡ sin(x²)` problem where
different syntactic forms of the same function get different node counts.

```python
import sympy
expr = sympy.sympify(str(tree))
simplified = sympy.simplify(expr)
size = count_nodes(simplified)
```

**3. Add crossover**

Current mutation only modifies one tree at a time. Genetic programming
traditionally includes crossover — swapping subtrees between two parent trees.
This lets good sub-expressions discovered in different lineages combine,
which is more powerful than mutation alone.

**4. Smarter exploration — UCB over operators**

Rather than choosing operators uniformly at random when generating or mutating
trees, use a UCB (Upper Confidence Bound) bandit to track which operators have
historically appeared in high-fitness trees. This biases exploration towards
likely-useful primitives without eliminating the less common ones.

---

## Medium term — close the loop

**5. Implement the closed-loop extension**

The next level described in the original prompt: the AI's output influences the
next input. Concretely:

- Generate a time series where `x_{t+1} = f(x_t) + AI_output_t`
- The AI discovers `f` and simultaneously learns what output minimises
  prediction error
- This is now reinforcement learning with a symbolic policy

This is the most interesting and least explored direction.

**6. Stream mode — online learning**

Currently the code takes a fixed batch of points. Change it to process a
stream: receive one (x, y) pair at a time, update candidate fitness
incrementally, and track how the beam evolves as more data arrives. This maps
directly to the original "time series" framing and to the feedback loop pillar.

---

## Longer term — scaling and priors

**7. Learned structural prior**

The search currently treats all expression structures as equally likely a
priori. Train a small model (or use a pre-trained LLM with prompting) on a
corpus of known mathematical functions to predict which structural compositions
are plausible before evaluating them. Use this as a prior to bias the search.

**8. Multi-target symbolic regression**

Rather than fitting one function, discover a *system* of equations that jointly
explain multiple related time series. For example, given `(x(t), y(t))` from a
predator-prey simulation, recover the Lotka-Volterra equations. PySR has some
support for this; a hand-rolled version would require a shared expression tree
with multiple outputs.

**9. Noise robustness**

Add Gaussian noise to the generated data and evaluate how robust the search
is. Real data is always noisy. MDL helps here — the complexity penalty
naturally prevents overfitting to noise — but the search may need a
noise-aware fitness function (e.g. robust regression loss instead of MSE).

---

## File structure when these are implemented

```
guess_function.py          # univariate, hand-rolled (current)
guess_function_pysr.py     # uni/multivariate, PySR backend (current)
guess_function_v2.py       # + scipy coefficient optimisation + simplification
guess_stream.py            # online/streaming mode
guess_closed_loop.py       # closed-loop / policy discovery
make_data.py               # data generator (current, uni/multivariate)
make_data_stream.py        # streaming data generator
make_data_system.py        # system of equations (e.g. Lotka-Volterra)
```
