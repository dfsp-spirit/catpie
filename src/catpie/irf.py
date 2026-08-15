"""
Item Response Function (4PL) and item information, mirroring catR's ``Pi()``,
``Ii()`` and ``Ji()`` for the dichotomous model.

An item is a 4-tuple ``(a, b, c, d)`` where:
  a = discrimination, b = difficulty, c = guessing (lower asymptote),
  d = inattention (upper asymptote).

This is the exact scalar mirror of catjs-irt's ``src/irf.js``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

from .math import _exp

Item = Sequence[float]  # (a, b, c, d)


class PiResult(NamedTuple):
    """4PL probability and its first three derivatives (catR ``Pi()``)."""

    P: float
    dP: float
    d2P: float
    d3P: float


class IiResult(NamedTuple):
    """Fisher information and its first two derivatives (catR ``Ii()``)."""

    Ii: float
    dIi: float
    d2Ii: float


class JiResult(NamedTuple):
    """Weighted-likelihood quantity and its derivative (catR ``Ji()``)."""

    Ji: float
    dJi: float


def pi(th: float, item: Item, D: float = 1.0) -> PiResult:
    """
    4PL probability of a correct response and its first three derivatives.
    Mirrors catR ``Pi(th, it, D)`` for a single item:

        e    <- exp(D * a * (th - b))
        Pi   <- c + (d - c) * e/(1 + e)
        Pi[Pi == 0] <- 1e-10
        Pi[Pi == 1] <- 1 - 1e-10
        dPi  <- D * a * e * (d - c)/(1 + e)^2
        d2Pi <- D^2 * a^2 * e * (1 - e) * (d - c)/(1 + e)^3
        d3Pi <- D^3 * a^3 * e * (d - c) * (e^2 - 4*e + 1)/(1 + e)^4
    """
    a, b, c, d = item
    e = _exp(D * a * (th - b))
    P = c + ((d - c) * e) / (1 + e)
    if P == 0:
        P = 1e-10
    if P == 1:
        P = 1.0 - 1e-10
    # NOTE: explicit multiplication (not ``**``) so that overflow yields
    # +/-inf and then NaN exactly like R/JS (which return Inf/Infinity and
    # NaN), instead of raising OverflowError like Python's ``**`` does.
    e1 = 1 + e
    dP = (D * a * e * (d - c)) / (e1 * e1)
    d2P = (D * D * a * a * e * (1 - e) * (d - c)) / (e1 * e1 * e1)
    d3P = (
        (D * D * D * a * a * a * e * (d - c) * (e * e - 4 * e + 1))
        / (e1 * e1 * e1 * e1)
    )
    return PiResult(P, dP, d2P, d3P)


def ii(th: float, item: Item, D: float = 1.0) -> IiResult:
    """
    Fisher information of one item at ability ``th``, and its first two
    derivatives. Mirrors catR ``Ii(th, it)`` dichotomous branch:

        Q   <- 1 - P
        Ii  <- dP^2/(P * Q)
        dIi <- dP * (2*P*Q*d2P - dP^2*(Q - P))/(P^2 * Q^2)
        d2Ii <- (2*P*Q*(d2P^2 + dP*d3P) - 2*dP^2*d2P*(Q - P))/(P^2*Q^2)
                - (3*P^2*Q*dP^2*d2P - P*dP^4*(2*Q - P))/(P^4*Q^2)
                + (3*P*Q^2*dP^2*d2P - Q*dP^4*(Q - 2*P))/(P^2*Q^4)
    """
    pr = pi(th, item, D)
    P = pr.P
    dP = pr.dP
    d2P = pr.d2P
    d3P = pr.d3P
    Q = 1 - P
    dP2 = dP * dP
    P2 = P * P
    Q2 = Q * Q
    Ii = dP2 / (P * Q)
    dIi = (dP * (2 * P * Q * d2P - dP2 * (Q - P))) / (P2 * Q2)
    d2Ii = (
        (2 * P * Q * (d2P * d2P + dP * d3P) - 2 * dP2 * d2P * (Q - P))
        / (P2 * Q2)
        - (3 * P2 * Q * dP2 * d2P - P * (dP2 * dP2) * (2 * Q - P))
        / ((P2 * P2) * Q2)
        + (3 * P * Q2 * dP2 * d2P - Q * (dP2 * dP2) * (Q - 2 * P))
        / (P2 * (Q2 * Q2))
    )
    return IiResult(Ii, dIi, d2Ii)


def ji(th: float, item: Item, D: float = 1.0) -> JiResult:
    """
    Weighted-likelihood quantity (third-derivative term), mirroring catR's
    ``Ji(th, it)`` dichotomous branch:

        Q   <- 1 - P
        Ji  <- dP * d2P/(P * Q)
        dJi <- (P * Q * (d2P^2 + dP * d3P) - dP^2 * d2P * (Q - P))/(P^2 * Q^2)
    """
    pr = pi(th, item, D)
    P = pr.P
    dP = pr.dP
    d2P = pr.d2P
    d3P = pr.d3P
    Q = 1 - P
    dP2 = dP * dP
    P2 = P * P
    Q2 = Q * Q
    Ji = (dP * d2P) / (P * Q)
    dJi = (P * Q * (d2P * d2P + dP * d3P) - dP2 * d2P * (Q - P)) / (P2 * Q2)
    return JiResult(Ji, dJi)
