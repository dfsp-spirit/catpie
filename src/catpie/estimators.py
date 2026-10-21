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

#: Alias so that functions taking a catR-style ``range`` argument can still
#: use the builtin.
_pyRange = range


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


def _isConstantPattern(x: Sequence[Number]) -> bool:
    """True if all responses are wrong or all are correct (catR's check)."""
    values = [v for v in x if not _is_missing(v)]
    return bool(values) and (sum(values) == 0 or sum(values) == len(values))


def _robustScore(
    th: float,
    it: Sequence[Item],
    x: Sequence[Number],
    weight: str,
    tuCo: float,
    D: float,
    withDerivative: bool = False,
) -> tuple:
    """
    catR's robust score function (``T.rob``): the usual score, with per-item
    weights that damp influential items.

    Items are down-weighted when their ``a * (theta - b)`` residual exceeds
    ``tuCo`` (Huber: ``tuCo/|r|``; Tukey: ``(1 - (r/tuCo)^2)^2``, and zero
    beyond ``tuCo``).
    """
    if weight not in ("Huber", "Tukey"):
        raise ValueError("'weight' must be either 'Huber' or 'Tukey'")
    score = 0.0
    factors = []
    for resp, item in zip(x, it):
        pr = pi(th, item, D)
        P, Q, dP = pr.P, 1 - pr.P, pr.dP
        r_i = item[0] * (th - item[1])
        if weight == "Huber":
            w_i = 1.0 if abs(r_i) <= tuCo else tuCo / abs(r_i)
        else:
            w_i = (1.0 - (r_i / tuCo) ** 2) ** 2 if abs(r_i) <= tuCo else 0.0
        score += w_i * dP * (resp - P) / (P * Q)
        factors.append(w_i)
    if withDerivative:
        return score, factors
    return score


def _solveScore(f, RANGE: Tuple[float, float]) -> float:
    """
    catR's root-finding wrapper: ``uniroot`` when the score changes sign,
    otherwise the ``optimize``-based fallback (clamp to a bound, or bracket the
    root between the minimum and maximum of the score).
    """
    fLo = f(RANGE[0])
    fHi = f(RANGE[1])
    if (fLo < 0 < fHi) or (fLo > 0 > fHi):
        return uniroot(f, RANGE[0], RANGE[1])
    pr: OptimizeResult = optimizeScalar(f, RANGE[0], RANGE[1])
    if pr.y > 0:
        return RANGE[1]
    pr2: OptimizeResult = optimizeScalar(f, RANGE[0], RANGE[1], maximize=True)
    if pr2.y < 0:
        return RANGE[0]
    lo = min(pr2.x, pr.x)
    hi = max(pr2.x, pr.x)
    return uniroot(f, lo, hi)


def _robustRoot(
    it: Sequence[Item],
    x: Sequence[Number],
    weight: str,
    tuCo: float,
    D: float,
    range: Tuple[float, float],
) -> float:
    """Ability estimate by catR's robust (Huber/Tukey) estimator."""
    return _solveScore(
        lambda th: _robustScore(th, it, x, weight, tuCo, D), range
    )


def _distributionSE(estimates: Sequence[float], probabilities: Sequence[float]) -> float:
    """Standard deviation of an estimator over its response-pattern distribution."""
    total = sum(probabilities)
    mean = sum(p * e for p, e in zip(probabilities, estimates)) / total
    variance = sum(p * (e - mean) ** 2 for p, e in zip(probabilities, estimates)) / total
    return _sqrt(variance)


def fullDist(
    th: Sequence[float],
    it: Sequence[Item],
    method: str = "BM",
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    weight: str = "Huber",
    tuCo: float = 1.0,
    range: Tuple[float, float] = (-4, 4),
    parInt: Tuple[float, float, int] = (-4, 4, 33),
) -> list:
    """
    Exact posterior distribution of the ability estimate (catR ``fullDist()``).

    Enumerates every possible 0/1 response pattern for the ``n`` administered
    items, estimates ability for each pattern, and records its probability. The
    result is the distribution of the estimator, which is what
    :func:`semTheta` uses for ``semExact=True``.

    Args:
        th: Ability values at which the pattern probabilities are evaluated;
            the first one is the estimate whose SE is wanted.
        it: The administered items, as ``(a, b, c, d)`` tuples.
        method: Ability estimator used for each pattern.
        priorDist: Prior distribution.
        priorPar: Prior parameters.
        weight: Robust weight (``"Huber"``/``"Tukey"``) for ``method="ROB"``.
        tuCo: Tuning constant for the robust weights.
        range: Search interval for the estimator.
        parInt: Integration grid for ``method="EAP"``.

    Returns:
        A list of ``(estimate, probability_at_th[0], probability_at_th[1], ...)``
        tuples, one per response pattern (``2 ** n`` rows).

    Raises:
        ValueError: If more than 20 items are given (``2 ** n`` patterns would
            be impractical, exactly as in catR).
    """
    n = len(it)
    if n > 20:
        raise ValueError(
            "fullDist: too many items (%d); at most 20 are supported" % (n,)
        )
    rows = []
    for pattern in _pyRange(2**n):
        responses = [(pattern >> (n - 1 - i)) & 1 for i in _pyRange(n)]
        estimate = thetaEst(
            it,
            responses,
            method=method,
            priorDist=priorDist,
            priorPar=priorPar,
            parInt=parInt,
            range=range,
            weight=weight,
            tuCo=tuCo,
        )
        probabilities = []
        for theta in th:
            prob = 1.0
            for resp, item in zip(responses, it):
                p = pi(theta, item, 1.0).P
                prob *= p**resp * (1 - p) ** (1 - resp)
            probabilities.append(prob)
        rows.append((estimate, *probabilities))
    return rows


def thetaEst(
    it: Sequence[Item],
    x: Sequence[Number],
    method: str = "EAP",
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    D: float = 1.0,
    range: Tuple[float, float] = (-4, 4),
    parInt: Tuple[float, float, int] = (-4, 4, 33),
    weight: str = "Huber",
    tuCo: float = 1.0,
    constantPatt: Optional[str] = None,
    currentTh: float = 0.0,
    bRange: Tuple[float, float] = (-2, 2),
) -> float:
    """
    Estimate a person's ability (theta) from their responses.

    This is the main estimation function. Give it the items that were
    administered and the person's 0/1 responses, and it returns the best
    estimate of their ability. The higher the ability, the "better" the
    person performed.

    Four estimation methods are available:

    - ``"EAP"`` (default): Bayesian, most robust, uses the prior; the result
      always lies within the integration grid (default ``[-4, 4]``).
    - ``"BM"``: Bayesian modal estimate (posterior maximum), uses the prior.
    - ``"ML"``: maximum likelihood; uses only the responses (no prior). Can
      be extreme (and is unstable on degenerate item banks, exactly like catR).
    - ``"WL"``: weighted likelihood; reduces the bias of ML.

    Missing responses (``NaN``/``None``) and their items are ignored.

    Args:
        it: The administered items as ``(a, b, c, d)`` tuples (same length as
            ``x``).
        x: The responses as 0/1 integers (1 = correct); ``NaN``/``None``
            entries are dropped together with the corresponding items.
        method: One of ``"EAP"`` (default), ``"BM"``, ``"ML"``, ``"WL"``.
        priorDist: Prior distribution for ``EAP``/``BM``: ``"norm"``
            (default), ``"unif"``, ``"Jeffreys"``.
        priorPar: Prior parameters ``(mean, sd)`` for ``"norm"`` (default
            ``(0, 1)``) or ``(lower, upper)`` for ``"unif"``.
        D: Scale constant (default ``1.0``).
        range: The search interval ``(lower, upper)`` for the BM/ML/WL root
            finding (default ``(-4, 4)``). Not used by EAP.
        parInt: ``(lower, upper, npoints)`` of the integration grid for EAP
            (default ``(-4, 4, 33)``).

    Returns:
        The estimated ability as a float (a higher value = higher estimated
        ability). For EAP it lies within the integration grid; BM/ML/WL may
        return a range boundary when no interior root exists (like catR).

    Raises:
        ValueError: If ``method`` is not one of EAP/BM/ML/WL/ROB, if
            ``priorDist`` is not supported, or if ``constantPatt`` is unknown.

    Note:
        ``constantPatt`` implements catR's handling of *degenerate* response
        patterns (all correct or all wrong), where the score equation has no
        interior root. ``"BM"``/``"EAP"``/``"WL"`` switch to that estimator;
        ``"fixed4"``/``"fixed7"`` move ``currentTh`` by 0.4/0.7 logits in the
        direction of the responses; ``"var"`` averages ``currentTh`` with the
        corresponding bound of ``bRange``. ``method="ROB"`` is catR's robust
        estimator, which down-weights items whose ``a * (theta - b)`` residual
        exceeds ``tuCo`` (``weight="Huber"`` or ``"Tukey"``).
    """
    constant = _isConstantPattern(x)
    if constant and constantPatt is not None:
        if constantPatt in ("BM", "EAP", "WL"):
            method = constantPatt
        else:
            fit, fx = dropMissing(it, x)
            allWrong = sum(fx) == 0
            if constantPatt in ("fixed4", "fixed7"):
                step = 0.4 if constantPatt == "fixed4" else 0.7
                return currentTh - step if allWrong else currentTh + step
            if constantPatt == "var":
                bound = bRange[0] if allWrong else bRange[1]
                return 0.5 * (currentTh + bound)
            raise ValueError(
                "thetaEst: constantPatt %r not implemented "
                "(use 'BM', 'EAP', 'WL', 'fixed4', 'fixed7' or 'var')" % (constantPatt,)
            )

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
    if method not in ("BM", "ML", "WL", "ROB"):
        raise ValueError("thetaEst: method %r not implemented" % (method,))

    if method == "ROB":
        return_ = _robustRoot(fit, fx, weight, tuCo, D, range)
        return return_

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
    return _solveScore(f, RANGE)


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
    weight: str = "Huber",
    tuCo: float = 1.0,
    semExact: bool = False,
    trueTh: Optional[float] = None,
) -> Union[float, tuple]:
    """
    Standard error (SE) of an ability estimate.

    The SE quantifies the uncertainty of an ability estimate :func:`thetaEst`:
    small SE = confident estimate, large SE = uncertain. It is used by the
    stopping rules (e.g. stop once ``se`` drops below a threshold) and to build
    confidence intervals for classification.

    The method should match the one used for estimation (``method=...``).

    Args:
        thEst: The ability estimate (output of :func:`thetaEst`).
        it: The administered items as ``(a, b, c, d)`` tuples.
        x: The responses (0/1); ``NaN``/``None`` are dropped with their items.
        method: Estimation method: ``"EAP"`` (default), ``"BM"``, ``"ML"``,
            ``"WL"``.
        priorDist: Prior distribution for EAP/BM (``"norm"`` default,
            ``"unif"``, ``"Jeffreys"``).
        priorPar: Prior parameters (see :func:`thetaEst`).
        D: Scale constant (default ``1.0``).
        parInt: Integration grid ``(lower, upper, npoints)`` for EAP (default
            ``(-4, 4, 33)``).
        semType: ``"classic"`` (default) or ``"new"`` (only implemented for
            the classic form in this port).
        range: Accepted for catR signature parity; not used by the implemented
            methods.

    Returns:
        The standard error as a non-negative float. As a rough guide, values
        below ~0.3 are often considered "precise enough" for adaptive testing.

    Raises:
        ValueError: If ``method`` is not one of EAP/BM/ML/WL/ROB or
            ``priorDist`` is not supported.
    """
    if semExact:
        # Exact SE: the standard deviation of the estimator over all possible
        # response patterns, weighted by their probabilities (catR's
        # `sem.exact`, which calls fullDist()).
        abilities = [thEst] if trueTh is None else [thEst, trueTh]
        dist = fullDist(
            abilities,
            it,
            method=method,
            priorDist=priorDist,
            priorPar=priorPar,
            weight=weight,
            tuCo=tuCo,
            range=range,
            parInt=parInt,
        )
        estimates = [row[0] for row in dist]
        seExact = _distributionSE(estimates, [row[1] for row in dist])
        if trueTh is None:
            return seExact
        return seExact, _distributionSE(estimates, [row[2] for row in dist])

    fit, fx = dropMissing(it, x)

    if method == "ROB":
        _, factors = _robustScore(thEst, fit, fx, weight, tuCo, D, withDerivative=True)
        # catR weights the *per-item* information by the robust weights.
        itemInfo = [ii(thEst, item, D).Ii for item in fit]
        numerator = _sqrt(sum(w * w * i for w, i in zip(factors, itemInfo)))
        denominator = sum(w * i for w, i in zip(factors, itemInfo))
        return numerator / denominator

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
