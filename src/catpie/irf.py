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
    Probability of a correct answer (and its derivatives) for one item.

    This is the *item response function* (IRF): the chance that a person with
    ability ``th`` answers an item correctly, given the item's 4-parameter
    model. In plain words: the higher the ability relative to the item
    difficulty, the higher the probability.

    Returns the probability ``P`` together with its first, second and third
    derivatives with respect to ability (``dP``, ``d2P``, ``d3P``) — these are
    used internally by the information and estimation routines.

    Args:
        th: Ability level theta. A real number, typically in ``[-4, 4]``
            (outside that range the probability saturates toward its
            asymptotes).
        item: Item parameters as a 4-tuple ``(a, b, c, d)``:

            - ``a`` discrimination, must be ``> 0`` (typically 0.5-2.5)
            - ``b`` difficulty (typically in ``[-3, 3]``)
            - ``c`` guessing / lower asymptote, in ``[0, 1)`` (typically 0-0.3)
            - ``d`` inattention / upper asymptote, in ``(0, 1]`` (typically 0.9-1)
        D: Scale constant (catR default ``1.0``; use ``1.702`` for the
            logistic approximation to the normal ogive).

    Returns:
        A :class:`PiResult` with fields ``P``, ``dP``, ``d2P``, ``d3P``.
        ``P`` lies between the asymptotes ``c`` and ``d``.

    Note:
        ``P`` is clamped exactly like catR: ``P == 0`` becomes ``1e-10`` and
        ``P == 1`` becomes ``1 - 1e-10``. When the exponent overflows (extreme
        ability and/or discrimination), ``P`` becomes ``NaN`` — the same
        behaviour as R/catR.
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
    Fisher information of one item at ability ``th`` (and its derivatives).

    Information measures how much a single item can tell you about a person's
    ability at level ``th``: it is largest where the item discriminates well
    (around its difficulty) and smallest where the answer is almost certain.
    High information = a good next question, which is why item selection uses
    it (see :func:`nextItem` with ``criterion="MFI"``).

    Args:
        th: Ability level (real number, typically ``[-4, 4]``).
        item: Item parameters ``(a, b, c, d)`` — see :func:`pi`.
        D: Scale constant (default ``1.0``).

    Returns:
        An :class:`IiResult` with fields ``Ii``, ``dIi``, ``d2Ii``. ``Ii`` is
        non-negative; it can be ``NaN`` on overflow, exactly like catR.
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
    Weighted-likelihood quantity of one item (and its derivative).

    This helper appears in the maximum-likelihood ("ML") and
    weighted-likelihood ("WL") ability estimators. You normally do not need
    to call it directly; it is provided for completeness and for parity with
    catR.

    Args:
        th: Ability level (real number).
        item: Item parameters ``(a, b, c, d)`` — see :func:`pi`.
        D: Scale constant (default ``1.0``).

    Returns:
        A :class:`JiResult` with fields ``Ji`` and ``dJi``.
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
