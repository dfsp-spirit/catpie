"""
Ability (theta) estimation and its standard error, mirroring catR's
``thetaEst()`` and ``semTheta()`` for the dichotomous model:

  - method:     "EAP" | "BM" | "ML" | "WL"
  - priorDist:  "norm" | "unif" | "Jeffreys"  (used by EAP and BM)
  - defaults:   D=1, priorPar=(0,1), parInt=(-4,4,33), range=(-4,4)

The non-EAP methods replicate catR's exact algorithm: solve the score equation
T(th) = r0(th) + r(th) = 0 by bisection (R's ``uniroot``) over ``range``, with
catR's ``optimize()``-based fallback when T does not change sign.

This mirrors catjs-irt's ``src/estimators.js``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence, Tuple, Union

from .eap import eapEst, eapSem
from .irf import Item, ii, ji, pi
from .math import OptimizeResult, _sqrt, optimizeScalar, uniroot

Number = Union[int, float]


class SumDersResult(NamedTuple):
    """Summed information and its derivatives over a set of items."""

    Ii: float
    dIi: float
    d2Ii: float


def _is_missing(v: object) -> bool:
    """True for NaN/None, matching catjs-irt's NaN/undefined/null filter."""
    if v is None:
        return True
    try:
        return math.isnan(v)  # type: ignore[arg-type]
    except TypeError:
        return False


def dropMissing(
    it: Sequence[Item], x: Sequence[Number]
) -> Tuple[Sequence[Item], Sequence[Number]]:
    """
    Filter out NaN/None responses and the corresponding items, matching catR's
    ``thetaEst``: ``ind <- which(!is.na(x)); it <- it[ind, ]; x <- x[ind]``.
    """
    idx = [i for i, v in enumerate(x) if not _is_missing(v)]
    return [it[i] for i in idx], [x[i] for i in idx]


def sumIi(th: float, it: Sequence[Item], D: float) -> float:
    """Sum of item information at ``th`` over ``it``."""
    return sum(ii(th, item, D).Ii for item in it)


def sumDers(th: float, it: Sequence[Item], D: float) -> SumDersResult:
    """Sum of item information derivatives at ``th`` over ``it``."""
    Ii = 0.0
    dIi = 0.0
    d2Ii = 0.0
    for item in it:
        res = ii(th, item, D)
        Ii += res.Ii
        dIi += res.dIi
        d2Ii += res.d2Ii
    return SumDersResult(Ii, dIi, d2Ii)


def r0(
    th: float,
    it: Sequence[Item],
    method: str = "BM",
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    D: float = 1.0,
) -> float:
    """catR's ``r0`` term (prior / method correction) in the score equation."""
    if method == "ML":
        return 0.0
    if method == "WL":
        sumJ = sum(ji(th, item, D).Ji for item in it)
        return sumJ / (2 * sumIi(th, it, D))
    # BM
    if priorDist == "norm":
        return (priorPar[0] - th) / priorPar[1] ** 2
    if priorDist == "unif":
        return 0.0
    if priorDist == "Jeffreys":
        ders = sumDers(th, it, D)
        return ders.dIi / (2 * ders.Ii)
    raise ValueError("thetaEst: priorDist %r not implemented" % (priorDist,))


def r(th: float, it: Sequence[Item], x: Sequence[Number], D: float = 1.0) -> float:
    """catR's ``r`` term: sum of dP*(x - P)/(P*Q) over items."""
    res = 0.0
    for i, item in enumerate(it):
        pr = pi(th, item, D)
        P = pr.P
        Q = 1 - P
        dP = pr.dP
        res += (dP * (x[i] - P)) / (P * Q)
    return res


def thetaEst(
    it: Sequence[Item],
    x: Sequence[Number],
    method: str = "EAP",
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    D: float = 1.0,
    range: Tuple[float, float] = (-4, 4),
    parInt: Tuple[float, float, int] = (-4, 4, 33),
) -> float:
    """
    Ability estimate. Mirrors catR ``thetaEst(it, x, method=..., ...)``.
    ``it`` is a sequence of items ``(a, b, c, d)``; ``x`` is the 0/1 response
    vector.

    Note: the parameter ``range`` intentionally keeps catR's name (it shadows
    the builtin ``range``); the function never uses the builtin.
    """
    fit, fx = dropMissing(it, x)

    if method == "EAP":
        return eapEst(
            fit,
            fx,
            D=D,
            priorDist=priorDist,
            priorPar=priorPar,
            lower=parInt[0],
            upper=parInt[1],
            nqp=parInt[2],
        )
    if method not in ("BM", "ML", "WL"):
        raise ValueError("thetaEst: method %r not implemented" % (method,))

    # catR: f is T(th) = r0(th) + r(th), except BM+unif which reduces to ML
    # and searches over the prior interval.
    usePrior = not (method == "BM" and priorDist == "unif")

    def f(th: float) -> float:
        t = r(th, fit, fx, D)
        if usePrior:
            return (
                r0(
                    th,
                    fit,
                    method=method,
                    priorDist=priorDist,
                    priorPar=priorPar,
                    D=D,
                )
                + t
            )
        return t

    RANGE = priorPar if (method == "BM" and priorDist == "unif") else range

    fLo = f(RANGE[0])
    fHi = f(RANGE[1])
    if (fLo < 0 < fHi) or (fLo > 0 > fHi):
        return uniroot(f, RANGE[0], RANGE[1])

    # catR fallback: minimize f; if min > 0 -> upper bound; maximize; if max < 0
    # -> lower bound; else root between the argmax and argmin.
    pr: OptimizeResult = optimizeScalar(f, RANGE[0], RANGE[1])
    if pr.y > 0:
        return RANGE[1]
    pr2: OptimizeResult = optimizeScalar(f, RANGE[0], RANGE[1], maximize=True)
    if pr2.y < 0:
        return RANGE[0]
    lo = min(pr2.x, pr.x)
    hi = max(pr2.x, pr.x)
    return uniroot(f, lo, hi)


def semTheta(
    thEst: float,
    it: Sequence[Item],
    x: Sequence[Number],
    method: str = "EAP",
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    D: float = 1.0,
    parInt: Tuple[float, float, int] = (-4, 4, 33),
    semType: str = "classic",
    range: Tuple[float, float] = (-4, 4),
) -> float:
    """
    Standard error of an ability estimate. Mirrors catR ``semTheta(...)``:

      - EAP -> eapSem
      - ML  -> 1/sqrt(info)
      - WL  -> 1/sqrt(info)                    (classic)
      - BM  -> 1/sqrt(info - dr0)              (classic), dr0 per prior

    Note: the parameter ``range`` intentionally keeps catR's name (catR uses it
    only for ``sem.exact``, which is not implemented here); it is accepted for
    signature parity and otherwise unused.
    """
    fit, fx = dropMissing(it, x)

    if method == "EAP":
        return eapSem(
            thEst,
            fit,
            fx,
            D=D,
            priorDist=priorDist,
            priorPar=priorPar,
            lower=parInt[0],
            upper=parInt[1],
            nqp=parInt[2],
        )
    if method not in ("BM", "ML", "WL"):
        raise ValueError("semTheta: method %r not implemented" % (method,))

    info = sumIi(thEst, fit, D)

    if method == "ML":
        return 1.0 / _sqrt(info)

    if method == "WL":
        # classic: 1/sqrt(info); new: sqrt(info)/abs(info - dr0) -- classic only
        # is implemented (the catR new-type WL needs Ji derivatives).
        if semType == "new":
            dr0wl = 0.0
            return _sqrt(info) / abs(info - dr0wl)
        return 1.0 / _sqrt(info)

    # BM
    if priorDist == "norm":
        dr0 = -1.0 / priorPar[1] ** 2
    elif priorDist == "unif":
        dr0 = 0.0
    elif priorDist == "Jeffreys":
        ders = sumDers(thEst, fit, D)
        dr0 = (ders.d2Ii * ders.Ii - ders.dIi * ders.dIi) / (2 * ders.Ii * ders.Ii)
    else:
        raise ValueError("semTheta: priorDist %r not implemented" % (priorDist,))

    if semType == "classic":
        return 1.0 / _sqrt(info - dr0)
    return _sqrt(info) / abs(info - dr0)
