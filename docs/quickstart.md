# Quickstart

This page walks you through a complete adaptive test with catpie, step by
step. If you are new to adaptive testing, you may want to read
[Concepts](concepts.md) first — but the example is understandable on its own.

## The idea

You have an **item bank**: a list of questions, each described by four numbers
`(a, b, c, d)` that say how discriminating (`a`), how difficult (`b`), how
guessable (`c`) and how "inattentive-friendly" (`d`) the question is. For each
person you want to find their **ability** (usually called *theta*, written
$\theta$).

## Step 1 — define an item bank

A tiny bank of five items (you would normally have dozens or hundreds):

```python
from catpie import estimateTheta, genPattern, selectNextItem, randomCAT

# (a, b, c, d) = (discrimination, difficulty, guessing, inattention)
bank = [
    (1.0, -1.0, 0.20, 0.95),   # easy-ish question
    (1.5,  1.0, 0.10, 0.98),   # harder question
    (0.9,  0.0, 0.15, 0.97),   # medium question
    (1.2,  0.5, 0.05, 0.99),
    (1.1, -0.5, 0.25, 0.96),
]
```

## Step 2 — run a live adaptive session

This is the core loop of a real experiment. For each step:

1. **select** the next item with [`selectNextItem`](api.md#catpie.selectNextItem),
2. **administer** it and record the person's 0/1 answer,
3. **estimate** ability and standard error with
   [`estimateTheta`](api.md#catpie.estimateTheta).

```python
administered = []   # indices of items already used
responses = []      # the person's 0/1 answers, same order

# --- First item: we don't know ability yet, so start at theta = 0 ---
sel = selectNextItem(bank, theta=0.0, administered=administered)
administered.append(sel.item)
print("First item:", sel.item, "info =", round(sel.info, 4))
# First item: 2 info = 0.2818   (the most informative item at theta = 0)

# --- Score the answer (in a real experiment this comes from the participant) ---
answer = 1          # correct
responses.append(answer)

# --- Update the ability estimate ---
est = estimateTheta(bank, administered, responses)
print("theta =", round(est.theta, 4), " se =", round(est.se, 4))
# theta = 0.5071  se = 0.9145

# --- Next item, at the updated estimate ---
sel = selectNextItem(bank, theta=est.theta, administered=administered)
administered.append(sel.item)
responses.append(0)   # wrong this time
est = estimateTheta(bank, administered, responses)
print("theta =", round(est.theta, 4), " se =", round(est.se, 4))
# theta = -0.0395  se = 0.6224     <- se went down (more confidence)
```

That is the whole loop: select, administer, estimate, repeat.

!!! tip "Reading the numbers"
    - `theta` is the ability estimate. A higher value = higher estimated
      ability. It usually stays within about `[-4, 4]`.
    - `se` is the standard error. Smaller = more confident. Values below
      ~0.3 are often considered "precise enough".

## Step 3 — simulate a whole test offline

To validate your item bank and stopping rules *before* deploying, simulate
thousands of fake participants with
[`randomCAT`](api.md#catpie.simulation.randomCAT). It runs the full loop
(selection + response + estimation) and stops when a rule triggers:

```python
run = randomCAT(
    trueTheta=0.7,                  # this "person's" true ability
    itemBank=bank,
    method="EAP",                   # robust Bayesian estimation
    stop={"rule": ["precision", "length"], "thr": [0.3, 10]},
    rng=lambda: 0.37,               # fixed seed for reproducibility
)
print("items administered:", run.nItems)
print("final theta:", round(run.finalTheta, 4))
print("final se:", round(run.finalSe, 4))
print("stopping rule:", run.stopRule)
# items administered: 5
# final theta: 1.2332
# final se: 0.7564
# stopping rule: None        (ran out of items before reaching precision 0.3)
```

The result object also stores the full history (`run.thetaHist`, `run.seHist`,
`run.administered`, ...) so you can plot how the estimate evolved.

## Step 4 — stop early with rules

In a real experiment you don't always want a fixed length. Use
[`checkStopRule`](api.md#catpie.simulation.checkStopRule) to stop when, say,
the standard error drops below `0.3`:

```python
from catpie import checkStopRule

stop = checkStopRule(th=0.4, se=0.25, n=12,
                     rule=["precision", "length"], thr=[0.3, 20])
print(stop.decision, stop.rule)   # True ['precision']
```

## Where to go next

- [Concepts](concepts.md) — the theory behind these numbers, in plain language.
- [API Reference](api.md) — full documentation of every function.
- [examples/demo.py](https://github.com/dfsp-spirit/catpie/blob/main/examples/demo.py) — a runnable copy of this example.
