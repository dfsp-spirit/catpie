# Concepts (for non-experts)

This page explains the ideas behind catpie in plain language, with the
**expected value ranges** for everything you will pass in or get back. It is
written for first-year PhD students who may be new to both Item Response
Theory (IRT) and Python — no prior knowledge is assumed.

---

## 1. Ability (theta, $\theta$)

In IRT, every person has a hidden **ability** (or trait level) we want to
measure — e.g. working memory capacity. We call it **theta** ($\theta$).

- It is a single number, on a **z-score-like scale**, typically in **$[-4, 4]$**.
- $\theta = 0$ is the "average" ability; positive is above average, negative is
  below average.
- We can never observe $\theta$ directly — we only **estimate** it from the
  person's answers.

**In catpie:** every function that takes or returns ability uses this scale.
The default estimation grid is 33 points from `-4` to `4`.

## 2. Items and their four parameters

Every question ("item") is described by four numbers, stored as a tuple
`(a, b, c, d)`:

| Parameter | Name | Meaning | Typical range |
|-----------|------|---------|---------------|
| `a` | discrimination | How well the item tells apart people of similar ability | `> 0`, usually **0.5 – 2.5** |
| `b` | difficulty | The ability level at which the item is "50/50" | usually **−3 – 3** |
| `c` | guessing | Chance of a correct answer by pure guessing (lower asymptote) | **0 – 0.3** |
| `d` | inattention | Upper asymptote — even the best person won't get it right 100% of the time | usually **0.9 – 1** |

For example, the item `(1.0, -1.0, 0.2, 0.95)` is:

- moderately discriminating (`a = 1.0`),
- on the easy side (`b = -1.0`, so even people with below-average ability have
  a good chance),
- guessable (`c = 0.2`: 20% chance by guessing),
- slightly "inattention-prone" (`d = 0.95`: the ceiling is 95%).

## 3. The item response function (the 4PL model)

The **item response function** gives the probability that a person with ability
$\theta$ answers an item `(a, b, c, d)` correctly:

$$P(\theta) = c + (d - c) \cdot \frac{\exp(a (\theta - b))}{1 + \exp(a (\theta - b))}$$

This is the *four-parameter logistic* (4PL) model. In words:

- For a person much **more** able than the difficulty ($\theta \gg b$), the
  probability approaches the ceiling `d`.
- For a person much **less** able ($\theta \ll b$), it approaches the guessing
  floor `c`.
- The curve is steepest at $\theta = b$; a larger `a` makes it steeper (more
  discriminating).

**In catpie:** [`pi(th, item)`](api.md#catpie.irf.pi) computes this probability
(and its derivatives). Example:

```python
from catpie import pi
item = (1.0, -1.0, 0.2, 0.95)
print(pi(-1.0, item).P)   # ~0.57 : at theta == b, P is midway between c and d
print(pi( 2.0, item).P)   # ~0.92 : high ability -> near the ceiling
```

## 4. Item information

**Fisher information** measures how much a single item can tell you about a
person's ability *at a given level* $\theta$. Intuitively: a question is most
informative where you are most unsure about the answer — i.e. around its
difficulty.

- It is a **non-negative** number.
- An item's information is highest near $\theta = b$ and lower elsewhere.
- In an adaptive test, the next item is chosen to be the one with the **highest
  information at the current estimate** (the MFI criterion, see below).

**In catpie:** [`ii(th, item)`](api.md#catpie.irf.ii) returns the information.
For typical items it ranges from near `0` (far from difficulty) up to a few
units (near difficulty); with strong discrimination it can be larger.

## 5. Estimating ability

After each answer, we re-estimate the person's ability. catpie implements four
methods (the `method` argument of [`thetaEst`](api.md#catpie.estimators.thetaEst)):

| Method | Name | Idea | Notes |
|--------|------|------|-------|
| **EAP** (default) | Expected A Posteriori | Bayesian: combines the answers (likelihood) with a **prior** belief about ability | **Most robust.** Always returns a value inside the grid (default `[-4, 4]`). **Recommended default.** |
| **BM** | Bayesian Modal | The ability at which the posterior is highest | Uses the prior; good with few answers |
| **ML** | Maximum Likelihood | The ability that makes the observed answers most likely | No prior; can be extreme/unstable on hard banks (catR behaves the same) |
| **WL** | Weighted Likelihood | A bias-reduced version of ML | Good compromise; a bit more robust than ML |

**Priors** (used by EAP and BM, argument `priorDist`):

- `"norm"` (default): normal prior `N(mean, sd)` with default `(0, 1)` — the
  standard choice.
- `"unif"`: uniform prior over an interval — "no prior opinion".
- `"Jeffreys"`: a non-informative prior based on the items themselves.

**Standard error (SE):** [`semTheta`](api.md#catpie.estimators.semTheta)
returns the uncertainty of an estimate. Smaller SE = more confidence. Values
below ~`0.3` are often treated as "precise enough". You will also use the SE to
build confidence intervals (e.g. $\theta \pm 1.96 \cdot \mathrm{SE}$ for a 95%
interval).

## 6. Choosing the next item

Adaptive testing's key step: pick the item that will teach you the most next.
Two criteria are implemented (the `criterion` argument of
[`nextItem`](api.md#catpie.selection.nextItem)):

- **MFI** (Maximum Fisher Information, default): pick the item with the
  **highest information at the current estimate** — usually an item whose
  difficulty is close to the current $\theta$.
- **bOpt**: pick the item whose **difficulty `b` is closest to the current
  $\theta$**. Simpler and often used early in a test.

Both avoid items already administered (`out` / `administered`). When several
items tie for the best value, catpie (and catpie) picks one at **random** — the
so-called `randomesque = 1` behaviour.

## 7. Stopping rules

You rarely want to administer the whole bank. Rules tell the test when to stop
(the `rule`/`thr` arguments of [`checkStopRule`](api.md#catpie.simulation.checkStopRule)):

| Rule | Stop when ... | Typical threshold |
|------|---------------|-------------------|
| `"length"` | a fixed number of items has been administered | e.g. `20` items |
| `"precision"` | the standard error drops to the threshold or below | e.g. `0.3` |
| `"classification"` | the confidence interval of $\theta$ lies entirely above/below a cutoff (e.g. "is ability > 0?") | a cutoff like `0` |
| `"minInfo"` | even the best remaining item carries little information | e.g. `0.2` |

Rules are combined with **OR** (the test stops as soon as *any* rule fires).

## 8. A quick reference of expected value ranges

| Quantity | Where | Expected range |
|----------|-------|----------------|
| Ability estimate `theta` | outputs of `thetaEst`, `estimateTheta`, `randomCAT` | usually `[-4, 4]` (EAP always within the grid) |
| Standard error `se` | outputs of `semTheta`, `estimateTheta`, `randomCAT` | `> 0`, typically `0.05 – 1.5` |
| Probability `P` | output of `pi` | between `c` and `d` (clamped like catR) |
| Item information `Ii` | output of `ii` | `>= 0`, typically `0 – ~5` |
| `a` (discrimination) | item tuples | `> 0`, usually `0.5 – 2.5` |
| `b` (difficulty) | item tuples | usually `-3 – 3` |
| `c` (guessing) | item tuples | `0 – 0.3` |
| `d` (inattention) | item tuples | usually `0.9 – 1` |
| Responses | `x`, `responses` arguments | `0` (wrong) or `1` (correct) |
| Item indices | `out`, `administered` | integers, **0-based** (first item is `0`) |

!!! tip "Remember"
    All item indices in catpie are **0-based** (like Python, unlike R). The
    first item in the bank is index `0`.
