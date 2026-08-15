"""
Simulation helpers: response generation, a simple CAT runner and stopping
rules. These mirror the *pieces* of catR (``genPattern``, ``checkStopRule``)
and provide a minimal, catR-inspired ``randomCAT`` loop.

Note: ``randomCAT`` here is deliberately a lightweight, documented loop - it is
NOT a bit-for-bit port of catR's ``randomCAT()`` (which has a much larger
option surface). Its building blocks (selection, estimation, SE) are the same
functions that validate to machine precision against catR.

This mirrors catjs-irt's ``src/simulation.js``.
"""

from __future__ import annotations

import math
import random
from typing import Callable, NamedTuple, Optional, Sequence, Union

from .estimators import semTheta, thetaEst
from .irf import Item, ii, pi
from .math import qnorm
from .selection import nextItem

Rng = Callable[[], float]


class StopRuleResult(NamedTuple):
    """Result of a stopping-rule check (catR ``checkStopRule()``)."""

    decision: bool
    rule: list  # list of rules that triggered


class RandomCatResult(NamedTuple):
    """Result of a minimal CAT run."""

    administered: list  # 0-indexed item indices
    responses: list  # 0/1 responses in administered order
    selected: list  # 0-indexed selections per step
    infoHist: list  # criterion value at each selection
    thetaHist: list  # ability estimate after each step
    seHist: list  # standard error after each step
    stopRule: Optional[list]  # stopping rule(s) that triggered, or None
    finalTheta: float
    finalSe: float
    nItems: int
    method: str
    itemSelect: str


def genPattern(
    theta: Union[float, Sequence[float]],
    items: Sequence[Item],
    D: float = 1.0,
    rng: Optional[Rng] = None,
) -> Union[list, list[list]]:
    """
    Generate a dichotomous 0/1 response pattern for one or more abilities,
    mirroring catR's ``genPattern(th, it, model=NULL, D=1)`` for the
    dichotomous case: each item is a Bernoulli draw with probability Pi(th).

    ``theta`` may be a single ability (returns one 0/1 pattern) or a sequence
    of abilities (returns a matrix: list of patterns).
    """
    if rng is None:
        rng = random.random
    is_multi = isinstance(theta, (list, tuple))
    thetas = list(theta) if is_multi else [theta]
    rows = [[1 if rng() < pi(th, item, D).P else 0 for item in items] for th in thetas]
    return rows if is_multi else rows[0]


def simulateRespondents(
    thetas: Sequence[float],
    itemBank: Sequence[Item],
    D: float = 1.0,
    rng: Optional[Rng] = None,
) -> list[list]:
    """
    Simulate response patterns for several respondents. Thin wrapper around
    ``genPattern``; returns a matrix of shape (respondents x items).

    This is our own minimal helper (not a catR function).
    """
    return genPattern(thetas, itemBank, D=D, rng=rng)


def checkStopRule(
    th: float,
    se: float,
    n: int,
    rule: Sequence[str] = ("length",),
    thr: Sequence[float] = (20,),
    alpha: float = 0.05,
    items: Optional[Sequence[Item]] = None,
    D: float = 1.0,
) -> StopRuleResult:
    """
    Stopping rule, mirroring catR's ``checkStopRule(th, se, N, it, stop)``.

    Rules (OR-combined): "length" (n >= thr), "precision" (se <= thr),
    "classification" (CI for th excludes thr), "minInfo" (max item info <= thr).
    """
    decision = False
    triggered = []
    for i, r_ in enumerate(rule):
        if r_ == "length":
            if n >= thr[i]:
                decision = True
                triggered.append(r_)
        elif r_ == "precision":
            if se <= thr[i]:
                decision = True
                triggered.append(r_)
        elif r_ == "classification":
            z = qnorm(1 - alpha / 2.0)
            if th - z * se >= thr[i] or th + z * se <= thr[i]:
                decision = True
                triggered.append(r_)
        elif r_ == "minInfo":
            if items is None:
                raise ValueError('checkStopRule: "minInfo" rule requires `items`')
            maxI = max(ii(th, item, D).Ii for item in items)
            if maxI <= thr[i]:
                decision = True
                triggered.append(r_)
        else:
            raise ValueError("checkStopRule: unknown rule %r" % (r_,))
    return StopRuleResult(decision, triggered)


def randomCAT(
    trueTheta: float,
    itemBank: Sequence[Item],
    method: str = "BM",
    priorDist: str = "norm",
    priorPar: tuple = (0, 1),
    D: float = 1.0,
    range: tuple = (-4, 4),
    parInt: tuple = (-4, 4, 33),
    itemSelect: str = "MFI",
    startTheta: float = 0.0,
    stop: Optional[dict] = None,
    minItems: int = 0,
    maxSteps: Optional[int] = None,
    responses: Optional[Sequence[int]] = None,
    rng: Optional[Rng] = None,
) -> RandomCatResult:
    """
    Minimal CAT runner (catR-inspired). Selects items, (optionally) simulates
    responses, and estimates ability after each step until a stopping rule
    triggers or the item bank / maxSteps is exhausted.

    - ``trueTheta``: true ability (only used when simulating responses)
    - ``itemSelect``: 'MFI' | 'bOpt'
    - ``startTheta``: ability used for the first selection (default 0)
    - ``stop``: dict with keys ``rule``, ``thr``, ``alpha`` passed to
      ``checkStopRule`` (default ``{"rule": ["length"], "thr": [20], "alpha": 0.05}``)
    - ``minItems``: minimum number of items before stopping is considered
    - ``maxSteps``: hard cap on administered items (default bank length)
    - ``responses``: optional fixed 0/1 response sequence to replay instead of
      simulating (for parity testing against catR)
    - ``rng``: injectable RNG for response simulation

    Note: the parameter ``range`` intentionally keeps catR's name.
    """
    if stop is None:
        stop = {"rule": ["length"], "thr": [20], "alpha": 0.05}
    if rng is None:
        rng = random.random

    estOpts = {
        "method": method,
        "priorDist": priorDist,
        "priorPar": priorPar,
        "D": D,
        "range": range,
        "parInt": parInt,
    }

    administered = []
    resp = list(responses) if responses else []
    thetaHist = []
    seHist = []
    selected = []
    infoHist = []

    nSteps = maxSteps if maxSteps is not None else len(itemBank)
    # Never exceed the bank size (catR also stops once all items are used).
    steps = min(len(resp) if responses else nSteps, len(itemBank))

    theta = startTheta
    se = math.inf
    stopRule = None

    s = 0
    while s < steps:
        if len(administered) >= minItems:
            stopRes = checkStopRule(
                theta,
                se,
                len(administered),
                rule=stop.get("rule", ["length"]),
                thr=stop.get("thr", [20]),
                alpha=stop.get("alpha", 0.05),
                items=itemBank,
                D=D,
            )
            if stopRes.decision:
                stopRule = stopRes.rule
                break

        sel = nextItem(itemBank, theta, administered, criterion=itemSelect, D=D)
        selected.append(sel.item)
        infoHist.append(sel.info)

        if responses:
            r = resp[len(administered)]
        else:
            P = pi(trueTheta, itemBank[sel.item], D).P
            r = 1 if rng() < P else 0
            resp.append(r)
        administered.append(sel.item)

        it = [itemBank[i] for i in administered]
        theta = thetaEst(it, resp[: len(administered)], **estOpts)
        se = semTheta(theta, it, resp[: len(administered)], **estOpts)
        thetaHist.append(theta)
        seHist.append(se)
        s += 1

    return RandomCatResult(
        administered=administered,
        responses=resp,
        selected=selected,
        infoHist=infoHist,
        thetaHist=thetaHist,
        seHist=seHist,
        stopRule=stopRule,
        finalTheta=theta,
        finalSe=se,
        nItems=len(administered),
        method=method,
        itemSelect=itemSelect,
    )
