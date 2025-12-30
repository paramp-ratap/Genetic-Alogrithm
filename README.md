# Genetic-Alogrithm
---

## Overview

You are given a propositional logic formula in **3-CNF** (conjunctive normal form, each clause has exactly 3 literals). The goal is to search for a model (an assignment of True/False to the variables) that **maximizes the percentage of satisfied clauses**.

Example 3‑CNF (for illustration):

```
(a ∨ ¬b ∨ c) ∧ (¬a ∨ b ∨ ¬c) ∧ (a ∨ ¬e ∨ ¬d) ∧ (¬b ∨ c ∨ d) ∧ (¬c ∨ ¬d ∨ e)
```

A model is an assignment to all variables — for example `(a = T, b = T, c = F, d = F, e = T)` — and a clause is satisfied if it evaluates to True under that model.

**Note on CNF Creator format:** the provided `CNF Creator.py` generates 3‑CNF formulas as lists of lists of integers. Each inner list contains 3 integers where a positive integer `k` denotes variable `x_k` and a negative integer `-k` denotes `¬x_k`:

```
[ -50, 37, -23 ]  # represents (¬x50 ∨ x37 ∨ ¬x23)
```

---

## Task

- The formula will use **n = 50** variables (state space size = 2^50 possible models).
- Use a **Genetic Algorithm (GA)** to find a model that maximizes the number of satisfied clauses.
- Define the fitness function as:

```
Fitness = (Number of satisfied clauses) / (Total number of clauses) × 100
```

---

## Requirements

1. **Baseline GA** (textbook standard implementation):
   - Population size = **20**.
   - Initial population sampled uniformly at random.

2. **Improved GA** (your implementation to outperform the baseline):
   - Design and implement improvements to find the best model as quickly as possible (increase fitness and reduce runtime).
   - You may modify any aspect of the GA (selection, crossover, mutation, elitism, local search, adaptive rates, etc.).
   - Ensure your program **terminates within 45 seconds**. You may also terminate earlier when either:
     1. fitness has not improved for several generations, or
     2. fitness reaches **100%**.

3. **Data generation:** Use the provided function `CreateRandomSentence()` to generate 3‑CNF sentences.
   - Number of variables: `n = 50` (fixed).
   - Number of clauses: `m ∈ {100, 120, 140, ..., 300}`.

---

## Reporting

1. **Average fitness graph:** for the improved algorithm, plot the average best fitness found (y‑axis) vs different values of `m` (x‑axis).
   - For each `m`, generate at least **10 random sentences** and report the average best fitness.
   - Stop execution at **45 seconds** if needed.

2. **Running time graph:** plot average running time vs different values of `m`.
   - For each `m`, use at least **10 random sentences**.
   - Running time must be **capped at 45 seconds** per run.

3. **Description of improvements:** explain the GA modifications you implemented and discuss approaches that failed.

4. **Observations:** discuss types of problems where GA struggles to find good solutions.

5. **Insights on 3‑CNF difficulty:** comment on when these problems become hardest to satisfy (empirical or theoretical observations).

---

## Useful reminders for implementation

- Fitness evaluation: evaluating a model on `m` clauses should be implemented efficiently — this is the dominant cost if you evaluate many individuals and generations.
- Keep careful control of runtime (45s limit). Consider using early stopping, elitism to preserve best individuals, and lightweight local improvements (e.g., greedy bit flips) if useful.
- Produce the required figures (fitness vs m, runtime vs m) for the report and save raw results so you can compute averages across multiple random instances.

---

## ANSWER 1 — Average best fitness vs m

* **Summary:** The average best fitness declines slowly as the number of clauses `m` increases. More clauses add restrictions, forcing tradeoffs between clauses and reducing the achievable fraction of satisfied clauses.
* **Figure:** (see image on page 1) "Average best fitness vs m" — shows near-100% fitness for small `m` and a gradual decline toward ~98.2% by `m = 300`.
* **Notes:** The slight uptick at `m = 300` is likely due to randomness in instance generation or sampling variation rather than a real improvement.

---

## ANSWER 2 — Average runtime vs m

* **Summary:** Runtime increases with `m`, with a sharp inflection around `m = 160–200` clauses where per-instance work balloons. Early on the solver can satisfy most clauses quickly, but as constraints accumulate, evaluating and locally refining candidates becomes more expensive.
* **Figure:** (see image on page 2) "Average runtime vs m" — runtime rises from near 0s at small `m` to ~3.3s at `m = 300`.

---

## ANSWER 3 — Improvements made and failed approaches

**Successful improvements implemented:**

1. **Heuristic seeding** — initialize some individuals by setting each variable to the majority sign observed across clauses; improves early fitness.
2. **Larger but balanced population (~60)** — increases diversity without excessive per-generation cost.
3. **Elitism + light local search** — preserve best individuals and perform a quick single-bit hill climb on elites.
4. **Uniform crossover** — per-bit mixing to spread useful assignments.
5. **Adaptive mutation rate** — reduce mutation when improving, increase when stagnating.
6. **Practical termination rules** — 45s cap, stop on perfect fitness, or stop on long stagnation.
7. **Occasional restarts** — when flatlining, restart with fresh seeded population.

**Approaches that underperformed:**

* Heavy local search (too slow), very large populations (>200), high mutation rates (>0.05), pure heuristic initialization (lost diversity), complex niching (overhead), multi-parent crossover (overhead).

---

## ANSWER 4 — Problem types where GA struggles

* **Highly contradictory clauses** (balanced positive/negative occurrences).
* **Near phase-transition region** (α ≈ 4.2), where good assignments are rare.
* **Very dense clause sets** (many clauses per variable).
* **Deceptive fitness landscapes** and **low diversity populations**.
* **Long-range dependencies** requiring coordinated flips.

**Practical signs:** early/long plateaus, population variance collapse, repeated similar local optima.

**Mitigations:** adaptive mutation, occasional restarts, light targeted local search on elites, hybrid seeding of heuristics + randoms, use diversity-preserving methods when time allows.

---

## ANSWER 5 — When 3-CNF problems become hardest

* **Phase transition:** Random 3‑SAT shows peak difficulty near the clause-to-variable ratio α ≈ 4.2 (for `n = 50`, this corresponds to `m ≈ 200–230`).
* **Why:** near-critical instances are marginally satisfiable, landscapes are rugged, solutions are isolated, and local changes often break many clauses.

**Experimental signatures:** downward trend of best fitness with `m`, spikes/dips near the transition, higher variance, and longer runtimes to reach high satisfaction thresholds.

**Advice for hard instances:** hybridize with SAT-local searches (e.g., WalkSAT), use many short runs (restarts/portfolios), sweep parameters by regime, or use specialized SAT solvers (DPLL/CDCL) when guarantees are needed.


