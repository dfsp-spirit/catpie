"""
Item selection criteria that are not plain Fisher information.

catR computes these in separate files (``MWI.R``, ``KL.R``); catpie keeps the
same split so the ports stay comparable one-by-one. Everything here works for
dichotomous items only, which is all catpie supports.

Both functions integrate over the same ability grid that the EAP estimator uses
(``parInt``, default ``(-4, 4, 33)``) with catR's trapezoid rule, and both are
weighted by the likelihood of the responses given so far -- that likelihood is
constant across candidate items, so callers that evaluate many candidates
should compute it once and pass it in (``lik=``).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from .irf import Item, ii, pi
from .math import dnorm, integrateCatR, linspace


def _log(x: float) -> float:
    """R's ``log()``: ``0`` maps to ``-inf`` and negatives to ``nan``."""
    if x > 0:
        return math.log(x)
    return -math.inf if x == 0 else math.nan


def _priorDensity(
    x: float, priorDist: str, priorPar: Tuple[float, float]
) -> float:
    """Prior density term used by the posterior-weighted criteria."""
    if priorDist == "norm":
        return dnorm(x, priorPar[0], priorPar[1])
    if priorDist == "unif":
        lo, hi = priorPar
        return 1.0 / (hi - lo) if lo <= x <= hi else 0.0
    raise ValueError("prior-dist must be 'norm' or 'unif', got %r" % (priorDist,))


def likelihood(
    x: Sequence[int],
    itGiven: Sequence[Item],
    grid: Sequence[float],
    D: float = 1.0,
) -> list:
    """
    Likelihood of the response pattern ``x`` at every ability in ``grid``.

    This is ``prod_i P_i^x_i * (1 - P_i)^(1-x_i)`` over the items already
    administered (the "likelihood function" that MLWI, MPWI, KL and KLP weight
    their criteria with). With no items given it is 1 everywhere, like catR.
    """
    values = []
    for theta in grid:
        prod = 1.0
        for resp, item in zip(x, itGiven):
            p = pi(theta, item, D).P
            prod *= p**resp * (1 - p) ** (1 - resp)
        values.append(prod)
    return values


def mwi(
    itemBank: Sequence[Item],
    item: int,
    x: Sequence[int],
    itGiven: Sequence[Item],
    type: str = "MLWI",
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    lower: float = -4.0,
    upper: float = 4.0,
    nqp: int = 33,
    D: float = 1.0,
    lik: Optional[Sequence[float]] = None,
) -> float:
    """
    Weighted information of candidate ``item`` (catR ``MWI()``).

    The criterion is the item information averaged over the plausible ability
    values, weighted by the likelihood of the observed responses
    (``type="MLWI"``) or by the posterior density (``type="MPWI"``). It is more
    forgiving than MFI when the ability estimate is still uncertain, which is
    usually the case early in a test.

    Args:
        itemBank: The full item bank, as ``(a, b, c, d)`` tuples.
        item: 0-based index of the candidate item.
        x: Responses (0/1) for the items in ``itGiven``, same order.
        itGiven: The items administered so far, as ``(a, b, c, d)`` tuples.
        type: ``"MLWI"`` (default) or ``"MPWI"``.
        priorDist: ``"norm"`` (default) or ``"unif"``; used by ``"MPWI"``.
        priorPar: Prior parameters, see :func:`catpie.eapEst`.
        lower: Lower bound of the integration grid (default ``-4``).
        upper: Upper bound of the integration grid (default ``4``).
        nqp: Number of quadrature points (default ``33``).
        D: Scale constant (default ``1.0``).
        lik: Optional precomputed :func:`likelihood` on the grid.

    Returns:
        The integrated criterion value (a float, higher is better).

    Raises:
        ValueError: If ``type`` is not ``"MLWI"``/``"MPWI"``.
    """
    if type not in ("MLWI", "MPWI"):
        raise ValueError("'type' must be either 'MLWI' or 'MPWI'")
    grid = linspace(lower, upper, nqp)
    if lik is None:
        lik = likelihood(x, itGiven, grid, D)
    crit = []
    for theta, weight in zip(grid, lik):
        value = weight * ii(theta, itemBank[item], D).Ii
        if type == "MPWI":
            value *= _priorDensity(theta, priorDist, priorPar)
        crit.append(value)
    return integrateCatR(grid, crit)


def kl(
    itemBank: Sequence[Item],
    item: int,
    x: Sequence[int],
    itGiven: Sequence[Item],
    theta: Optional[float] = None,
    type: str = "KL",
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    lower: float = -4.0,
    upper: float = 4.0,
    nqp: int = 33,
    D: float = 1.0,
    lik: Optional[Sequence[float]] = None,
) -> float:
    """
    Kullback-Leibler information of candidate ``item`` (catR ``KL()``).

    Measures how strongly the candidate would separate the current ability
    estimate from every other ability on the grid, weighted by the likelihood
    of the responses so far (``"KL"``) or by the posterior (``"KLP"``). For
    classification-style goals it is often preferable to Fisher information.

    Args:
        itemBank: The full item bank, as ``(a, b, c, d)`` tuples.
        item: 0-based index of the candidate item.
        x: Responses (0/1) for the items in ``itGiven``, same order.
        itGiven: The items administered so far.
        theta: Current ability estimate; the divergence is computed from it.
            Required by catpie (catR silently falls back to ML estimation).
        type: ``"KL"`` (default) or ``"KLP"``.
        priorDist: ``"norm"`` (default) or ``"unif"``; used by ``"KLP"``.
        priorPar: Prior parameters, see :func:`catpie.eapEst`.
        lower: Lower bound of the integration grid (default ``-4``).
        upper: Upper bound of the integration grid (default ``4``).
        nqp: Number of quadrature points (default ``33``).
        D: Scale constant (default ``1.0``).
        lik: Optional precomputed :func:`likelihood` on the grid.

    Returns:
        The integrated criterion value (a float, higher is better).

    Raises:
        ValueError: If ``type`` is not ``"KL"``/``"KLP"`` or ``theta`` is None.
    """
    if type not in ("KL", "KLP"):
        raise ValueError("'type' must be either 'KL' or 'KLP'")
    if theta is None:
        raise ValueError("kl: `theta` is required")
    grid = linspace(lower, upper, nqp)
    if lik is None:
        lik = likelihood(x, itGiven, grid, D)
    trueP = pi(theta, itemBank[item], D).P
    crit = []
    for xTheta, weight in zip(grid, lik):
        p = pi(xTheta, itemBank[item], D).P
        value = trueP * _log(trueP / p) + (1 - trueP) * _log((1 - trueP) / (1 - p))
        value *= weight
        if type == "KLP":
            value *= _priorDensity(xTheta, priorDist, priorPar)
        crit.append(value)
    return integrateCatR(grid, crit)
