# catpie

**Computerized Adaptive Testing (CAT) in pure Python.**

catpie is a faithful Python translation of the most relevant parts of the R
package [**catR**](https://cran.r-project.org/package=catR). It lets you build
"smart" questionnaires that adapt to each person while they answer — the
standard tool for measuring latent traits such as working memory capacity in
experimental psychology.

It has **zero runtime dependencies** (pure Python standard library), so it is
easy to install and plays nicely inside large experiment stacks such as
PsychoPy.

---

## What is adaptive testing? (30 seconds)

Imagine a test with hundreds of questions of different difficulty. Instead of
asking everyone the same questions, an *adaptive* test does this:

1. **Ask one question** that is well matched to the person's current estimated
   ability (so it tells you the most about them).
2. **Use their answer** to update the estimate of their ability, and how sure
   you are about it.
3. **Repeat** until you are sure enough (or ran out of time/questions).

This gives you the same measurement precision with far fewer questions than a
fixed test — which matters a lot when each question is a demanding cognitive
task.

## What catpie provides

| Area | Function(s) | What it does |
|------|-------------|--------------|
| Item response model | [`pi`](api.md#catpie.irf.pi), [`ii`](api.md#catpie.irf.ii), [`ji`](api.md#catpie.irf.ji) | Probability of a correct answer; item information (how informative a question is) |
| Ability estimation | [`thetaEst`](api.md#catpie.estimators.thetaEst), [`estimateTheta`](api.md#catpie.estimateTheta) | Estimate a person's ability from their answers |
| Uncertainty | [`semTheta`](api.md#catpie.estimators.semTheta) | Standard error of the ability estimate (how confident we are) |
| Item selection | [`nextItem`](api.md#catpie.selection.nextItem), [`selectNextItem`](api.md#catpie.selectNextItem) | Pick the next, most informative question |
| Simulation | [`genPattern`](api.md#catpie.simulation.genPattern), [`randomCAT`](api.md#catpie.simulation.randomCAT), [`checkStopRule`](api.md#catpie.simulation.checkStopRule) | Simulate participants and whole adaptive sessions offline |
| Math helpers | [`dnorm`](api.md#catpie.math.dnorm), [`qnorm`](api.md#catpie.math.qnorm), [`uniroot`](api.md#catpie.math.uniroot), ... | The numerical building blocks (mirror R's base functions) |

## Why trust the numbers?

catpie is not an approximation — it is a line-by-line translation of catR
(validated against the *real* R package). The EAP ability estimates match catR
to **floating-point precision** (~1e-15), item selection matches on every
compared case, and even the root-finding estimation methods (BM/ML/WL) match
to catR's own numerical tolerance. See [Parity with catR](validation.md) for
the full measured results.

## Where to go next

- **[Installation](install.md)** — set it up in two minutes with `uv`.
- **[Quickstart](quickstart.md)** — a complete, beginner-friendly example.
- **[Concepts](concepts.md)** — what ability, item parameters, information and
  the estimation methods mean, in plain language, with the expected value
  ranges.
- **[API Reference](api.md)** — every function documented with its arguments,
  return values and value ranges.

!!! note "A note on attribution"
    catpie is a **translation of the R package `catR`** by David Magis, Gilles
    Raîche and Juan Ramón Barrada (maintained by Cheng Hua). It is **not**
    written or endorsed by the catR authors. Please cite catR if you use
    catpie in academic work — see [Attribution](attribution.md).
