# Thoughts & Commentary on the AGI MVP

*Analysis, observations, and open questions from building this project.*

---

## What you described is symbolic regression

The core MVP — maintain a set of candidate compositional functions, search over
structures and coefficients, score by fit — is a well-studied field called
**symbolic regression**. The key difference from standard machine learning is
that the output is a human-readable formula rather than a weight matrix. This
matters enormously: a formula can be inspected, reasoned about, generalised,
and composed with other formulas. A weight matrix cannot.

What makes the framing here distinctive is the *motivation*. You are not using
symbolic regression as a tool to fit curves. You are proposing it as a
primitive for general intelligence — the hypothesis that cognition is, at its
core, discovering compact compositional descriptions of experience. That is a
much stronger claim, and a more interesting one.

---

## The size-accuracy insight is Kolmogorov complexity / MDL

The observation that "if we guess well, the size of the AI will be very small
and it will also be very accurate" is a precise restatement of two foundational
ideas from theoretical computer science and statistics:

**Kolmogorov complexity** — the true complexity of an object is the length of
the shortest program that produces it. `sin(x²) + 2x` has low Kolmogorov
complexity; a lookup table of 10,000 (x, y) pairs has high complexity, even if
both describe the same function perfectly over the training range. Crucially,
the compact description generalises — it will correctly predict `x = 100`. The
lookup table will not.

**Minimum Description Length (MDL)** — the best model is the one that
minimises `description_length(model) + description_length(data | model)`. In
our fitness function `MSE + λ · nodes`, MSE measures how well the model
explains the data and `λ · nodes` is the cost of the model itself. These are
the same thing stated differently.

The implication for AI is significant: a system that genuinely *understands* a
phenomenon will be *smaller* than one that merely memorises it. Current neural
networks are mostly memorising — their billions of weights encode patterns that
could often be expressed in far fewer symbols if the right abstractions were
found. The bet here is that finding those abstractions is the right path.

---

## The existing landscape

**PySR** (Miles Cranmer, Oxford/Princeton) — the most capable open-source
symbolic regression tool as of 2025. Uses a Julia backend
(`SymbolicRegression.jl`), does algebraic simplification, returns a Pareto
front. Cranmer developed it partly to rediscover physical laws from simulation
data (Lagrangian mechanics, cosmological structure). It is essentially the same
idea, built by someone who had the same instinct.

**Bayesian Program Learning** (Josh Tenenbaum, MIT) — "learning to learn" by
building probabilistic programs from primitives. Philosophically very aligned
with the 4 pillars, especially abstraction and exploration. The landmark 2015
Science paper showed one-shot handwriting recognition by learning a generative
program for each character class. The program was tiny; the generalisation was
human-level.

**Eureqa** (Schmidt & Lipson, Cornell, 2009) — the original commercial symbolic
regression tool. Caused a minor sensation by rediscovering Newton's second law
and conservation of energy from raw pendulum data. Now largely superseded by
PySR, but historically important as proof of concept.

**MCTS + function grammars** — using Monte Carlo Tree Search to navigate the
space of compositions rather than evolutionary search. Closer in spirit to
AlphaGo-style planning and a natural fit for the exploration pillar.

**Neurosymbolic AI** (broad field) — hybrid architectures that combine neural
networks for perception/embedding with symbolic systems for reasoning. This is
the direction many researchers are moving: neural nets are good at pattern
recognition, symbolic systems are good at composable reasoning.

---

## The unsolved problem we ran into: syntactic ≠ semantic complexity

Our hand-rolled beam search found `cos(π/2 + x²)` when the true answer was
`sin(x²)`. These are algebraically identical — `cos(π/2 + θ) = sin(θ)` — but
the search assigned them different node counts and couldn't see through the
equivalence.

This is a deep problem. The syntactic size of an expression tree is only a
proxy for Kolmogorov complexity. Two programs with the same semantics can have
wildly different sizes depending on which normal form they happen to be written
in. There are three known approaches:

1. **Algebraic simplification** — rewrite expressions to a canonical form
   before scoring, using a computer algebra system like SymPy. PySR does this,
   which is why it gave us the clean `sin(x²) + 2x` rather than the aliased
   form. The cost is runtime; simplification is itself non-trivial.

2. **Semantic hashing** — evaluate candidates on a large held-out set of
   points and deduplicate by their output vectors rather than their structure.
   Two trees that produce the same outputs are the same function. This is exact
   where algebraic simplification is approximate.

3. **Grammar design** — choose primitives carefully to minimise the number of
   algebraic identities in the language. Harder than it sounds. Every
   reasonable primitive set contains redundancies.

This is one of the central open problems in symbolic regression and program
synthesis more broadly.

---

## Why beam search is more than an optimisation

The shift from a single evolving population to a beam of K elite candidates
changes the character of the search in a meaningful way, not just in speed.

A single population tends to collapse: one lineage dominates through fitness
pressure, genetic drift kills diversity, and the search gets stuck in a local
optimum. This is a well-documented failure mode in genetic programming.

A beam maintains K independent lineages simultaneously. You can observe this
in the diversity printout every 50 generations: beam members may be exploring
structurally different hypotheses — one pursuing a `sin`-based explanation,
another trying a polynomial, a third using `exp`. They compete not through
reproduction but through ranking, so no lineage can crowd out the others until
it genuinely dominates on fitness.

This directly embodies the **exploration** pillar. The system is not committing
prematurely to one explanation. It is holding multiple competing hypotheses and
letting the data decide.

There is also a practical benefit: the beam gives you a ranked shortlist rather
than a single answer. In a real use case you might want the top 5 candidates
to show a human, rather than picking one algorithmically.

---

## Coefficient search is a hidden hard problem

The code treats constants as evolved values, mutated by Gaussian noise. This
works well enough for simple cases but is brittle for deep compositions. The
loss landscape over coefficients for a fixed structure — say
`a · sin(b · x² + c) + d · x` — is non-convex with many local minima, and
gradient-free mutation takes many generations to converge.

A cleaner approach for a fixed structure is to optimise coefficients with a
local numerical optimiser (e.g. `scipy.optimize.minimize`) after each
structural mutation. This decouples structure search (evolutionary/beam) from
coefficient optimisation (gradient-based), and the coefficient step can
converge in milliseconds. PySR does something similar internally.

This is also where the **approximability** pillar comes in from Vibhor's
original framing: given a fixed structure, coefficient fitting is a
well-understood approximation problem. The hard part is the structure.

---

## The closed-loop extension is genuinely novel

The "next level of complexity" — an AI whose outputs feed back as inputs — is
where this goes from curve-fitting to something more interesting. It is no
longer passive regression. The discovered formula becomes a *policy*.

This maps to several active research areas:

**Active inference** (Karl Friston, UCL) — an agent that acts on the world to
minimise surprise, not just to predict it. The agent's model of the world and
its policy are unified: both are described by the same generative model, and
action is inference. A symbolic generative model would be dramatically more
interpretable than Friston's neural implementations.

**Reinforcement learning with symbolic policies** — instead of a neural policy
function, the agent's behaviour is described by a compact formula discovered
by symbolic regression on its own experience. This would give you interpretable
RL policies that can be inspected, modified, and composed. No existing RL
framework does this cleanly.

**Closed-loop symbolic control** — the discovered formula is a control law: it
takes the current state as input and outputs an action that changes the state.
This is classical control theory (e.g. PID controllers), but the controller is
*learned* from data rather than hand-designed. Symbolic regression on
trajectories to discover control laws is an underexplored direction.

---

## The four pillars, audited against what we built

| Pillar | Status in this MVP |
|--------|-------------------|
| **Feedback loops** | Present — fitness score drives selection; each generation's survivors seed the next |
| **Approximability** | Present — MSE measures how well the candidate approximates the data; constant mutation is a local approximation step |
| **Abstraction & composability** | The main focus — expression trees over a primitive grammar are the mechanism |
| **Exploration** | Partially addressed — beam search maintains diversity; random subtree replacement explores new structures; but search is still largely local |

The weakest pillar is exploration. The current search is not fundamentally
different from random walk with selection. A more principled exploration
strategy — curiosity-driven search, UCB-style bandit selection over structural
choices, or learned priors over likely compositions — would significantly
improve coverage of the hypothesis space.

---

## The broader thesis

The 4-pillar framework is coherent and points at something real. The standard
deep learning paradigm is strong on approximability (universal approximators)
and feedback loops (gradient descent on massive data) but weak on abstraction
and composability (weights do not compose cleanly) and exploration (gradient
descent is local and exploits rather than explores).

Symbolic regression addresses the abstraction and composability gap directly.
The question is whether it can scale. The search space grows combinatorially
with expression depth, and real phenomena may require expressions too large to
search exhaustively with current methods.

The most credible path to scaling is probably a learned prior over the grammar:
rather than searching uniformly over all compositions, use a model trained on
many past symbolic regression problems to predict which structures are likely
to be useful. This is what large language models happen to be surprisingly good
at — they have seen millions of mathematical expressions and have implicit
priors about which compositions tend to appear in nature. A hybrid where an LLM
proposes candidate structures and a classical evaluator scores and refines them
is plausible and underexplored.

The MVP here demonstrates the core idea cleanly. It finds `sin(x²) + 2x` from
40 data points in seconds, with no supervision and no prior knowledge beyond
the primitive set. That is the abstraction and composability pillar working
exactly as intended. The path from here to general intelligence is long, but
the direction is clear.
