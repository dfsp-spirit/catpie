"""
Expected A Posteriori (EAP) estimation, mirroring catR's ``eapEst()`` and
``eapSem()`` exactly (including the grid, the prior and the trapezoid
integration via ``integrate.catR``).

Supported priors (catR's ``priorDist``): "norm" (N(priorPar[0], priorPar[1])),
"unif" (uniform over priorPar), "Jeffreys" (weight sqrt(sum(Ii))).

This mirrors catjs-irt's ``src/eap.js``.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple

from .irf import Item, ii, pi
from .math import dnorm, integrateCatR, linspace


def priorWeight(
    s: float,
    priorDist: str,
    priorPar: Tuple[float, float],
    it: Sequence[Item],
    D: float = 1.0,
) -> float:
    """
    Prior density term used in the EAP integrand, matching catR's
    ``switch(...)`` inside ``eapEst``/``eapSem``. For "norm" returns
    ``dnorm(s)``; for "unif" returns ``dunif(s)`` (1/(b-a) inside [a, b], 0
    outside); for "Jeffreys" returns ``sqrt(sum(Ii(s, it)))``.
    """
    if priorDist == "norm":
        return dnorm(s, priorPar[0], priorPar[1])
    if priorDist == "unif":
        a, b = priorPar[0], priorPar[1]
        return 1.0 / (b - a) if a <= s <= b else 0.0
    if priorDist == "Jeffreys":
        return math.sqrt(sum(ii(s, item, D).Ii for item in it))
    raise ValueError("eapEst: priorDist %r not implemented" % (priorDist,))


def makeLikelihood(
    it: Sequence[Item], x: Sequence[int], D: float = 1.0
) -> Callable[[float], float]:
    """
    Build the likelihood L(th) = prod_i P_i^x_i * (1 - P_i)^(1 - x_i),
    exactly as in catR's ``eapEst``:

        L <- function(th, it, x) prod(Pi(th, it, D=D)$Pi^x *
                                     (1-Pi(th, it, D=D)$Pi)^(1-x))
    """
    def likelihood(th: float) -> float:
        res = 1.0
        for i, item in enumerate(it):
            P = pi(th, item, D).P
            res *= P ** x[i] * (1 - P) ** (1 - x[i])
        return res

    return likelihood


def eapEst(
    it: Sequence[Item],
    x: Sequence[int],
    D: float = 1.0,
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    lower: float = -4,
    upper: float = 4,
    nqp: int = 33,
) -> float:
    """
    Ability estimate by Expected A Posteriori (EAP) estimation.

    EAP is a *Bayesian* estimate: it combines what the responses tell us (the
    likelihood) with a *prior* belief about the ability distribution. It is the
    most robust of the implemented estimators and the default choice of this
    package (and of the EWM experiment).

    The estimate is computed by numerical integration over a grid of ability
    values (catR's default: 33 points from ``-4`` to ``4``), so the result is
    always within ``[lower, upper]``.

    Args:
        it: The items that were administered, as a list of ``(a, b, c, d)``
            tuples. Length must equal ``len(x)``.
        x: The responses, as 0/1 integers (1 = correct) in the same order as
            ``it``.
        D: Scale constant (default ``1.0``).
        priorDist: The prior distribution, one of ``"norm"`` (normal),
            ``"unif"`` (uniform) or ``"Jeffreys"``.
        priorPar: Prior parameters. For ``"norm"``: ``(mean, sd)`` — catR
            default ``(0, 1)`` (standard normal). For ``"unif"``: ``(lower,
            upper)`` of the uniform interval.
        lower: Lower bound of the integration grid (default ``-4``).
        upper: Upper bound of the integration grid (default ``4``).
        nqp: Number of quadrature points in the grid (catR default ``33``).

    Returns:
        The estimated ability ``theta`` (a float, always within
        ``[lower, upper]``).
    """
    L = makeLikelihood(it, x, D)
    X = linspace(lower, upper, nqp)

    def w(s: float) -> float:
        return priorWeight(s, priorDist, priorPar, it, D)

    g = [s * w(s) * L(s) for s in X]
    h = [w(s) * L(s) for s in X]
    return integrateCatR(X, g) / integrateCatR(X, h)


def eapSem(
    thEst: float,
    it: Sequence[Item],
    x: Sequence[int],
    D: float = 1.0,
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    lower: float = -4,
    upper: float = 4,
    nqp: int = 33,
) -> float:
    """
    Standard error of an EAP ability estimate.

    The standard error tells you how confident you can be in the ability
    estimate :func:`eapEst`: smaller = more confidence. It is the standard
    deviation of the posterior distribution, computed by the same numerical
    integration over the grid.

    Args:
        thEst: The ability estimate (the output of :func:`eapEst`).
        it: The administered items, as ``(a, b, c, d)`` tuples.
        x: The responses (0/1), same order as ``it``.
        D: Scale constant (default ``1.0``).
        priorDist: Prior distribution, ``"norm"`` / ``"unif"`` / ``"Jeffreys"``.
        priorPar: Prior parameters (see :func:`eapEst`).
        lower: Lower bound of the integration grid (default ``-4``).
        upper: Upper bound of the integration grid (default ``4``).
        nqp: Number of quadrature points (catR default ``33``).

    Returns:
        The standard error (a non-negative float; typically a few tenths, and
        smaller after more informative items are administered).
    """
    L = makeLikelihood(it, x, D)
    X = linspace(lower, upper, nqp)

    def w(s: float) -> float:
        return priorWeight(s, priorDist, priorPar, it, D)

    g = [(s - thEst) ** 2 * w(s) * L(s) for s in X]
    h = [w(s) * L(s) for s in X]
    return math.sqrt(integrateCatR(X, g) / integrateCatR(X, h))
