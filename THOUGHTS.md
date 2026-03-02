# Thoughts & Commentary on the AGI MVP

*Analysis and observations from building this, in roughly chronological order.*

---

## What you described is symbolic regression

The core MVP — maintain a set of candidate compositional functions, search over
structures and coefficients, score by fit — is a well-studied field called
**symbolic regression**. The key difference from standard machine learning is
that the output is a human-readable formula, not a weight matrix. This matters
enormously: a formula can be inspected, reasoned about, generalised, and
composed with other formulas. A weight matrix cannot.

What makes your framing distinctive is the *motivation*: you're not using
symbolic regression as a tool to fit curves. You're proposing it as a primitive
for general intelligence — the hypothesis that cognition is, at its core,
discovering compact compositional descriptions of experience.

---

## The size-accuracy tradeoff is Kolmogorov complexity / MDL

Your observation that "if we guess well, the size of the AI will be very small
and it will also be very accurate" is a precise restatement of two foundational
ideas:

**Kolmogorov complexity** — the true complexity of an object is the length of
the shortest program that produces it. `sin(x²) + 2x` has low Kolmogorov
complexity; a lookup table of 10,000 (x, y) pairs has high complexity, even if
both describe the same function perfectly.

**Minimum Description Length (MDL)** — the best model is the one that
minimises `description_length(model) + description_length(data | model)`. In
our fitness function, `MSE + λ · nodes`, MSE measures how well the model
explains the data and `λ · nodes` is the cost of the model itself. These are
the same thing.

The implication is profound: a system that genuinely understands a phenomenon
will be *smaller* than one that merely memorises it. Neural networks, for all
their power, are mostly memorising — their weights encode patterns that could
often be expressed in far fewer symbols.

---

## The existing landscape worth knowing

**PySR** (Miles Cranmer, Oxford) — the most capable open-source symbolic
regression tool. Uses a Julia backend (`SymbolicRegression.jl`), does algebraic
simplification, returns a Pareto front. This is essentially what you described,
built by someone who had the same instinct.

**Bayesian Program Learning** (Josh Tenenbaum, MIT) — "learning to learn" by
building probabilistic programs from primitives. Philosophically very aligned
with the 4 pillars, especially abstraction and exploration. Famous for the
one-shot handwriting learning paper (2015, Science).

**MCTS + function grammars** — using Monte Carlo Tree Search to navigate the
space of compositions. A natural fit for the exploration pillar.

**Eureqa** — the original commercial symbolic regression tool (2009), which
caused a minor sensation by rediscovering physical laws from raw data. Now
largely superseded by PySR.

---

## The unsolved problem we ran into: syntactic ≠ semantic complexity

Our hand-rolled beam search found `cos(π/2 + x²)` when the true answer was
`sin(x²)`. These are algebraically identical — `cos(π/2 + θ) = -(-sin(θ)) =
sin(θ)` — but the search assigned them different node counts and couldn't see
through the equivalence.

This is a deep problem. The syntactic size of an expression tree is only a
proxy for its true Kolmogorov complexity. Two programs with the same semantics
can have wildly different sizes depending on which normal form they happen to be
written in. A complete solution requires either:

1. **Algebraic simplification** — rewrite expressions to a canonical form
   before scoring (e.g. using SymPy). PySR does this.
2. **Semantic hashing** — evaluate candidates on a large set of points and
   deduplicate by output, not by structure.
3. **Grammar design** — choose primitives carefully so that fewer algebraic
   identities exist. Harder than it sounds.

This is one of the central open problems in symbolic regression, and arguably
in program synthesis more broadly.

---

## The beam search adds something real

The shift from a single evolving population to a beam of K elite candidates is
not just an engineering optimisation. It changes the character of the search:

- A single population tends to collapse — one lineage dominates and diversity
  dies. This is genetic drift.
- A beam maintains K independent lineages simultaneously. You can watch them
  explore different structural hypotheses (e.g. one beam member pursues a
  `sin`-based explanation while another tries a `polynomial` one).
- The beam diversity printout every 50 generations makes this visible: you can
  literally see the search holding multiple competing hypotheses.

This maps directly onto your pillar of **exploration** — the system shouldn't
commit prematurely to one explanation.

---

## The next level: output token that influences next input

You mentioned this as a "next level of complexity" — an AI whose outputs feed
back into its inputs. This is no longer passive curve-fitting. It's closer to:

- **Active inference** (Karl Friston) — an agent that acts on the world to
  reduce surprise, not just to predict it
- **Closed-loop symbolic control** — the discovered formula becomes a policy,
  not just a description
- **Reinforcement learning with symbolic policies** — instead of a neural
  policy, the agent's behaviour is described by a compact formula that it
  discovers and refines over time

This is much less explored than pure symbolic regression and is genuinely novel
territory. The symbolic representation gives you something RL with neural
policies lacks: interpretability and composability. You could inspect the
policy, understand why it acts as it does, and compose it with other known
policies.

---

## The broader thesis

The 4-pillar framework is coherent and points at something real. The standard
deep learning paradigm is strong on approximability (universal function
approximators) and feedback loops (gradient descent) but weak on abstraction
and composability (weights don't compose) and exploration (gradient descent
is local and greedy).

Symbolic regression addresses the abstraction and composability gap directly.
The question is whether it can scale — the search space grows combinatorially
with expression depth, and real-world phenomena may require expressions too
large to search exhaustively. The answer is probably that the right architecture
combines both: neural networks for fast perceptual processing, symbolic systems
for high-level reasoning and generalisation. This is roughly the direction of
neurosymbolic AI, which is an active research front.

The MVP you've built demonstrates the core idea cleanly. It finds `sin(x²) + 2x`
from 40 data points in seconds, with no supervision and no prior knowledge
beyond the primitive set. That's the abstraction and composability pillar
working exactly as intended.
