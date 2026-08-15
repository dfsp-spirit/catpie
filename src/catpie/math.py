"""
Numerical helpers that mirror the R base functions used by catR.

Each function intentionally replicates catR's exact arithmetic so that the
Python port produces (near) bit-identical numbers to R/catR. This mirrors the
sibling JavaScript port ``catjs-irt`` (``src/math.js``) line by line, which was
validated against the real R package.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Sequence, Union

Number = Union[int, float]


class OptimizeResult(NamedTuple):
    """Result of a scalar optimization, analogous to R's ``optimize()``."""

    x: float
    y: float


def dnorm(x: Number, mean: Number = 0, sd: Number = 1) -> float:
    """
    Density of the normal distribution, matching R's ``dnorm``.

    Gives the height of the normal (bell) curve at point ``x``. Used as the
    ``"norm"`` prior in EAP/BM estimation.

    Args:
        x: Point at which to evaluate the density.
        mean: Mean of the normal distribution (default ``0``).
        sd: Standard deviation, must be ``> 0`` (default ``1``).

    Returns:
        The density (a positive float; largest at ``x == mean``).
    """
    z = (x - mean) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def linspace(start: Number, stop: Number, n: int) -> list[float]:
    """
    Return ``n`` evenly spaced numbers from ``start`` to ``stop`` (inclusive).

    Matches R's ``seq(from, to, length.out = n)`` exactly, including how the
    step is computed, so results agree with catR to the last bit. Used to build
    the EAP integration grid (e.g. 33 points from -4 to 4).

    Args:
        start: First value.
        stop: Last value.
        n: Number of points (``>= 2``).

    Returns:
        A list of ``n`` floats, ``start`` first and ``stop`` last.
    """
    by = (stop - start) / (n - 1)
    return [start + i * by for i in range(n)]


def integrateCatR(x: Sequence[Number], y: Sequence[Number]) -> float:
    """
    Integrate ``y`` over ``x`` using the trapezoid rule.

    Approximates the area under the curve by summing trapezoids between
    consecutive points. Matches catR's ``integrate.catR`` exactly. Used to
    compute the EAP estimate and its standard error over the integration grid.

    Args:
        x: The x-coordinates (must be sorted, e.g. from :func:`linspace`).
        y: The function values at those coordinates (same length as ``x``).

    Returns:
        The approximate integral (a float).
    """
    res = 0.0
    for i in range(len(x) - 1):
        hauteur = x[i + 1] - x[i]
        base = (y[i] + y[i + 1]) / 2.0
        res += base * hauteur
    return res


def qnorm(p: Number, mean: Number = 0, sd: Number = 1) -> float:
    """
    Quantile of the normal distribution, matching R's ``qnorm``.

    The inverse of the normal cumulative distribution: for a probability ``p``,
    returns the value ``z`` such that ``P(Z <= z) = p`` for ``Z ~ N(mean, sd)``.
    Used by the ``"classification"`` stopping rule to build confidence
    intervals (e.g. ``qnorm(0.975) ~ 1.96`` for a 95% interval).

    Args:
        p: A probability in ``(0, 1)`` (``0`` -> ``-inf``, ``1`` -> ``inf``).
        mean: Mean of the normal distribution (default ``0``).
        sd: Standard deviation, ``> 0`` (default ``1``).

    Returns:
        The quantile (a float; ``qnorm(0.5) == mean``).
    """
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf

    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            mean
            + sd
            * (
                ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]
            )
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            mean
            + sd
            * (
                q
                * (
                    ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4])
                    * r
                    + a[5]
                )
            )
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return (
        mean
        - sd
        * (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
            + c[5]
        )
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    )


def uniroot(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    tol: float = 1.22e-4,
    maxIter: int = 1000,
) -> float:
    """
    Find a root of ``f`` in ``[lower, upper]`` by bisection.

    ``f`` must change sign over the interval (``f(lower)`` and ``f(upper)``
    must have opposite signs), which guarantees at least one root. Mirrors
    R's ``uniroot`` semantics and is used internally by the BM/ML/WL ability
    estimators. You normally do not need to call it directly.

    Args:
        f: A function of one variable returning a number.
        lower: Lower bound of the search interval.
        upper: Upper bound of the search interval.
        tol: Convergence tolerance on the interval width (default matches
            catR parity: about ``1.22e-4``).
        maxIter: Maximum number of bisection steps (default ``1000``).

    Returns:
        A float ``x`` in ``[lower, upper]`` with ``f(x) ~ 0``.

    Raises:
        ValueError: If ``f`` does not change sign over the interval.
    """
    a = lower
    b = upper
    fa = f(a)
    fb = f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("uniroot: f() at the endpoints must have opposite signs")
    for _ in range(maxIter):
        if (b - a) / 2 < tol:
            return (a + b) / 2
        mid = (a + b) / 2
        fm = f(mid)
        if fm == 0:
            return mid
        if fm * fa < 0:
            b = mid
        else:
            a = mid
            fa = fm
    return (a + b) / 2


def optimizeScalar(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    maximize: bool = False,
    tol: float = 1e-12,
    maxIter: int = 200,
) -> OptimizeResult:
    """
    Find the minimum (or maximum) of ``f`` on ``[lower, upper]``.

    Golden-section search, analogous to R's ``optimize()``. Used internally by
    the BM/ML/WL ability estimators as a fallback when the score equation has
    no sign change; you normally do not need to call it directly.

    Args:
        f: A function of one variable returning a number.
        lower: Lower bound of the search interval.
        upper: Upper bound of the search interval.
        maximize: If ``True``, find the maximum instead of the minimum.
        tol: Convergence tolerance on the interval width (default ``1e-12``).
        maxIter: Maximum number of iterations (default ``200``).

    Returns:
        An :class:`OptimizeResult` with fields ``x`` (the location of the
        optimum) and ``y`` (the function value there, i.e. ``f(x)``).
    """
    g = (lambda x: -f(x)) if maximize else f
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a = lower
    b = upper
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = g(c)
    fd = g(d)
    for _ in range(maxIter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - invphi * (b - a)
            fc = g(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = g(d)
    x = (a + b) / 2.0
    return OptimizeResult(x, f(x))


def _sqrt(v: Number) -> float:
    """
    Square root that returns NaN for negative input instead of raising,
    matching JS ``Math.sqrt`` (and R's ``sqrt``, which returns NaN with a
    warning). Used internally so the port degrades exactly like catR rather
    than crashing.
    """
    if v < 0:
        return float("nan")
    return math.sqrt(v)


def _exp(x: Number) -> float:
    """
    Exponential that returns inf on overflow instead of raising, matching R's
    ``exp`` and JS ``Math.exp`` (which return Inf/Infinity rather than
    throwing an overflow error). Python's ``math.exp`` raises ``OverflowError``,
    so this is required to reproduce catR's overflow behaviour (e.g. ``Pi``
    becomes NaN when ``e`` overflows, exactly like in R).
    """
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf
