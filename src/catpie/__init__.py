"""
catpie - Computerized Adaptive Testing (CAT) in pure Python.

A faithful translation of the most relevant subset of the R package ``catR``:
the 4PL item response function, EAP/BM/ML/WL ability estimation with standard
errors, and MFI/bOpt item selection, plus the simulation helpers
``genPattern``, ``checkStopRule`` and a minimal ``randomCAT`` loop.

This is a Python port of the R package ``catR`` (and mirrors the sibling
JavaScript port ``catjs-irt``). It is not written or endorsed by the catR
authors. Numbers are computed with the exact same formulas and grid as catR
(defaults: D=1, priorDist="norm", priorPar=(0,1), parInt=(-4,4,33), trapezoid
integration over 33 points), so results match catR to floating point precision
for the same inputs.

An "item" is a 4-tuple ``(a, b, c, d)`` (discrimination, difficulty, guessing,
inattention). Item indices are 0-indexed throughout the public API.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

from .eap import eapEst, eapSem
from .estimators import semTheta, thetaEst
from .irf import IiResult, Item, JiResult, PiResult, ii, ji, pi
from .math import (
    OptimizeResult,
    dnorm,
    integrateCatR,
    linspace,
    optimizeScalar,
    qnorm,
    uniroot,
)
from .selection import NextItemResult, nextItem
from .simulation import (
    RandomCatResult,
    StopRuleResult,
    checkStopRule,
    genPattern,
    randomCAT,
    simulateRespondents,
)

__version__ = "0.1.0"

__all__ = [
    # IRF
    "pi",
    "ii",
    "ji",
    "PiResult",
    "IiResult",
    "JiResult",
    # estimation
    "eapEst",
    "eapSem",
    "thetaEst",
    "semTheta",
    # selection
    "nextItem",
    "NextItemResult",
    # simulation
    "genPattern",
    "simulateRespondents",
    "checkStopRule",
    "randomCAT",
    "StopRuleResult",
    "RandomCatResult",
    # high-level helpers
    "estimateTheta",
    "selectNextItem",
    "ThetaResult",
    # math helpers (mirror catR/R base functions)
    "dnorm",
    "linspace",
    "integrateCatR",
    "qnorm",
    "uniroot",
    "optimizeScalar",
    "OptimizeResult",
    "__version__",
]


class ThetaResult(NamedTuple):
    """Ability estimate and its standard error (high-level helper)."""

    theta: float
    se: float


def estimateTheta(
    itemBank: Sequence[Item],
    administered: Sequence[int],
    responses: Sequence[int],
    method: str = "EAP",
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    D: float = 1.0,
    range: Tuple[float, float] = (-4, 4),
    parInt: Tuple[float, float, int] = (-4, 4, 33),
    semType: str = "classic",
) -> ThetaResult:
    """
    High-level helper matching the experiment's ``estimate_theta_catr(...)``:
    estimate ability and its standard error from the items administered so far.

    ``itemBank`` is the full item bank, ``administered`` the 0-indexed
    administered item indices, ``responses`` the 0/1 responses for them.

    With no administered items, returns the prior ``(0, Inf)`` (matching the
    experiment's catr.py).

    Note: the parameter ``range`` intentionally keeps catR's name.
    """
    if len(administered) == 0:
        return ThetaResult(0.0, float("inf"))
    it = [itemBank[i] for i in administered]
    theta = thetaEst(
        it,
        responses,
        method=method,
        priorDist=priorDist,
        priorPar=priorPar,
        D=D,
        range=range,
        parInt=parInt,
    )
    se = semTheta(
        theta,
        it,
        responses,
        method=method,
        priorDist=priorDist,
        priorPar=priorPar,
        D=D,
        parInt=parInt,
        semType=semType,
    )
    return ThetaResult(theta, se)


def selectNextItem(
    itemBank: Sequence[Item],
    theta: float,
    administered: Sequence[int],
    criterion: str = "MFI",
    randomesque: int = 1,
    D: float = 1.0,
) -> NextItemResult:
    """
    High-level helper matching the experiment's ``select_next_item_catr(...)``:
    select the next item (default criterion MFI).
    """
    return nextItem(
        itemBank,
        theta,
        administered,
        criterion=criterion,
        randomesque=randomesque,
        D=D,
    )
