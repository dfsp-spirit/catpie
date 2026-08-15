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
    Simulate a 0/1 response pattern for one or more abilities.

    For each item, the person answers correctly (1) with probability equal to
    the item response function at the given ability (see :func:`pi`), and
    incorrectly (0) otherwise. This is used to generate "fake" participants
    for simulations and validation.

    Args:
        theta: The true ability level(s). A single number returns one pattern;
            a list/tuple of numbers returns one pattern per ability (a list of
            lists).
        items: The items to respond to, as ``(a, b, c, d)`` tuples.
        D: Scale constant (default ``1.0``).
        rng: Optional random source — a callable returning a float in
            ``[0, 1)``. Defaults to ``random.random``. Inject a fixed function
            (e.g. ``lambda: 0.5``) for reproducible tests.

    Returns:
        A list of 0/1 values (one per item) if ``theta`` is a single number,
        or a list of such lists (one per ability) otherwise.
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
    Simulate responses for several respondents on a whole item bank.

    Convenience wrapper around :func:`genPattern`: returns one 0/1 pattern per
    respondent, so you get a matrix of shape (respondents x items). Useful for
    testing item banks offline before deploying an experiment.

    Args:
        thetas: True ability of each respondent.
        itemBank: The item bank, as ``(a, b, c, d)`` tuples.
        D: Scale constant (default ``1.0``).
        rng: Optional random source (see :func:`genPattern`).

    Returns:
        A list (one entry per respondent) of 0/1 lists (one entry per item).
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
    Decide whether an adaptive test should stop.

    After each administered item, call this with the current ability estimate
    ``th``, its standard error ``se`` and the number of items administered
    ``n``. It returns ``True`` (with the rule(s) that triggered) as soon as any
    of the requested stopping rules is satisfied. Rules are combined with OR.

    Available rules (pass a list to ``rule``, thresholds to ``thr``):

    - ``"length"``: stop after ``thr`` items (e.g. ``rule=["length"],
      thr=[20]``).
    - ``"precision"``: stop when the standard error ``se`` drops to ``thr``
      or below (e.g. ``thr=[0.3]``).
    - ``"classification"``: stop when the 95% confidence interval of ``th``
      lies entirely above or below the threshold ``thr`` (e.g. decide whether
      ability is above 0).
    - ``"minInfo"``: stop when the maximum item information in the bank falls
      to ``thr`` or below (i.e. no remaining item can teach much). Requires
      ``items``.

    Args:
        th: Current ability estimate.
        se: Current standard error (output of :func:`semTheta`).
        n: Number of items administered so far.
        rule: List of rule names (``"length"``, ``"precision"``,
            ``"classification"``, ``"minInfo"``).
        thr: List of thresholds, one per rule (same order as ``rule``).
        alpha: Significance level for the ``"classification"`` rule (default
            ``0.05`` -> 95% confidence interval).
        items: The item bank, required by the ``"minInfo"`` rule.
        D: Scale constant (default ``1.0``).

    Returns:
        A :class:`StopRuleResult` with fields ``decision`` (bool: whether to
        stop) and ``rule`` (list of the rule names that triggered).

    Raises:
        ValueError: If a rule name is unknown, or ``"minInfo"`` is used
            without ``items``.
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
    Run a complete (simulated) adaptive test in one call.

    This ties everything together: it repeatedly (1) selects the next item,
    (2) simulates (or replays) the person's response, and (3) re-estimates
    ability and its standard error, until a stopping rule triggers or the item
    bank is exhausted. Use it to validate a question bank and stopping rules
    offline before deploying the real experiment.

    Args:
        trueTheta: The person's *true* ability (only used when simulating
            responses; a number, typically ``[-4, 4]``).
        itemBank: The full item bank, as ``(a, b, c, d)`` tuples.
        method: Estimation method (default ``"BM"``; ``"EAP"`` is the most
            robust).
        priorDist: Prior distribution (``"norm"`` default).
        priorPar: Prior parameters (default ``(0, 1)``).
        D: Scale constant (default ``1.0``).
        range: Search interval for BM/ML/WL estimation (default ``(-4, 4)``).
        parInt: Integration grid for EAP (default ``(-4, 4, 33)``).
        itemSelect: Item selection criterion, ``"MFI"`` (default) or
            ``"bOpt"``.
        startTheta: Ability used for the first item selection (default ``0``).
        stop: A dict with keys ``rule`` (list), ``thr`` (list) and ``alpha``
            passed to :func:`checkStopRule`. Default:
            ``{"rule": ["length"], "thr": [20], "alpha": 0.05}``.
        minItems: Minimum number of items to administer before stopping is
            considered (default 0).
        maxSteps: Hard cap on the number of items (default: the bank size).
        responses: Optional fixed 0/1 response sequence to replay instead of
            simulating (handy for testing against catR).
        rng: Optional random source for simulating responses (see
            :func:`genPattern`).

    Returns:
        A :class:`RandomCatResult` with the full history: ``administered``
        (0-based item indices), ``responses``, ``selected``, ``infoHist``,
        ``thetaHist``, ``seHist``, the triggering ``stopRule`` (or ``None``),
        ``finalTheta``/``finalSe``, ``nItems``, and the ``method``/``itemSelect``
        used.
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
