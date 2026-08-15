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
    """Standard normal (or normal) density, matching R's ``dnorm(x, mean, sd)``."""
    z = (x - mean) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def linspace(start: Number, stop: Number, n: int) -> list[float]:
    """
    Linearly spaced sequence, matching R's ``seq(from, to, length.out = n)``.

    R computes ``from + (0:(n-1)) * by`` with ``by = (to-from)/(n-1)``; we
    replicate that exactly (do NOT use ``numpy.linspace`` here, it can round
    the endpoints differently).
    """
    by = (stop - start) / (n - 1)
    return [start + i * by for i in range(n)]


def integrateCatR(x: Sequence[Number], y: Sequence[Number]) -> float:
    """
    Trapezoid integration, matching catR's ``integrate.catR(x, y)``:

        hauteur <- x[2:n] - x[1:(n-1)]
        base    <- rowMeans(cbind(y[1:(n-1)], y[2:n]))
        res     <- sum(base * hauteur)
    """
    res = 0.0
    for i in range(len(x) - 1):
        hauteur = x[i + 1] - x[i]
        base = (y[i] + y[i + 1]) / 2.0
        res += base * hauteur
    return res


def qnorm(p: Number, mean: Number = 0, sd: Number = 1) -> float:
    """
    Standard normal quantile function, matching R's ``qnorm(p, 0, 1)``.

    Uses Peter J. Acklam's rational approximation (relative error ~1.15e-9),
    the same accuracy class as R's AS 241 implementation. This is exactly what
    catjs-irt uses.
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
    Root finding by bisection, matching R's ``uniroot`` semantics (f must
    change sign over [lower, upper]). The default tolerance matches the value
    used by catjs-irt for catR parity (catR relies on R's ``uniroot`` default
    accuracy); tighter values are supported.
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
    Scalar minimizer (golden-section), analogous to R's ``optimize()`` used in
    catR's ``thetaEst`` fallback. Returns ``OptimizeResult(x, y)`` with x the
    argmin/argmax and y = f(x).
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
