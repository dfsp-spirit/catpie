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
    EAP ability estimate. Mirrors catR ``eapEst(it, x, ...)``:

        X  <- seq(from = lower, to = upper, length = nqp)
        Y1 <- s * prior(s) * L(s)      (g)
        Y2 <- prior(s) * L(s)          (h)
        RES <- integrate.catR(X, Y1) / integrate.catR(X, Y2)

    where prior(s) = dnorm / dunif / sqrt(sum(Ii)) per ``priorDist``.
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
    Standard error of an EAP estimate. Mirrors catR ``eapSem(thEst, it, x, ...)``:

        Y1 <- (s - thEst)^2 * prior(s) * L(s)
        Y2 <- prior(s) * L(s)
        RES <- sqrt(integrate.catR(X, Y1) / integrate.catR(X, Y2))
    """
    L = makeLikelihood(it, x, D)
    X = linspace(lower, upper, nqp)

    def w(s: float) -> float:
        return priorWeight(s, priorDist, priorPar, it, D)

    g = [(s - thEst) ** 2 * w(s) * L(s) for s in X]
    h = [w(s) * L(s) for s in X]
    return math.sqrt(integrateCatR(X, g) / integrateCatR(X, h))
