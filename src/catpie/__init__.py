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
    Estimate ability and its standard error from the items given so far.

    This is the highest-level estimation helper: hand it the whole item bank,
    the indices of the items already administered, and the person's 0/1
    responses, and you get back ``(theta, se)`` in one call. It is exactly the
    helper the EWM experiment's ``estimate_theta_catr(...)`` uses.

    Args:
        itemBank: The full item bank, as ``(a, b, c, d)`` tuples.
        administered: 0-based indices (into ``itemBank``) of the items
            administered so far.
        responses: 0/1 responses in the same order as ``administered``.
        method: Estimation method (default ``"EAP"``; also ``"BM"``,
            ``"ML"``, ``"WL"``).
        priorDist: Prior distribution (``"norm"`` default).
        priorPar: Prior parameters (default ``(0, 1)``).
        D: Scale constant (default ``1.0``).
        range: Search interval for BM/ML/WL (default ``(-4, 4)``).
        parInt: EAP integration grid (default ``(-4, 4, 33)``).
        semType: ``"classic"`` (default) SE form.

    Returns:
        A :class:`ThetaResult` with fields ``theta`` and ``se``. If no items
        have been administered yet, returns the prior estimate ``(0.0, inf)``.
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
    Select the next item to administer (default criterion: MFI).

    High-level wrapper around :func:`nextItem` that makes the most common call
    shorter. It is exactly the helper the EWM experiment's
    ``select_next_item_catr(...)`` uses.

    Args:
        itemBank: The full item bank, as ``(a, b, c, d)`` tuples.
        theta: Current ability estimate (a float, typically ``[-4, 4]``).
        administered: 0-based indices of the items already administered (they
            will not be selected again).
        criterion: ``"MFI"`` (default) or ``"bOpt"``.
        randomesque: Only the catR default ``1`` is supported.
        D: Scale constant (default ``1.0``).

    Returns:
        A :class:`NextItemResult`; see :func:`nextItem` for its fields.
    """
    return nextItem(
        itemBank,
        theta,
        administered,
        criterion=criterion,
        randomesque=randomesque,
        D=D,
    )
