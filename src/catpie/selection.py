"""
Item selection, mirroring catR's ``nextItem()`` with the criteria used by
catpie: "MFI" (Maximum Fisher Information) and "bOpt" (difficulty closest to
current ability), for any ``randomesque`` value.

catR's MFI branch (R/nextItem.R):

    items  <- rep(1, nrow(itemBank)); items[OUT] <- 0
    info   <- Ii(theta, itemBank)$Ii
    ranks  <- rank(info)
    nrIt   <- min(c(randomesque, sum(items)))
    keepRank <- sort(ranks[items == 1], decreasing = TRUE)[1:nrIt]
    keep   <- which(ranks == keepRank[i] & items == 1)
    select <- ifelse(length(keep) == 1, keep, sample(keep, 1))

That is: rank the criterion over the *whole* bank, keep the top ``nrIt`` ranks
*of the eligible items*, and let every eligible item whose rank is one of them
be a candidate (so items tied with the ``nrIt``-th best are included, which is
why the candidate set can hold more than ``randomesque`` items). One candidate
is then drawn uniformly at random.

The ``bOpt`` branch is the same with an ascending sort on ``|b - theta|``;
catR additionally assigns rank ``-1`` to administered items and de-duplicates
the kept ranks, which cannot change the candidate set (administered items are
filtered out either way).

Two deliberate deviations from catR, both documented rather than replicated:

* R's ``rank()`` is used with its default ``ties.method = "average"`` and
  ``na.last = TRUE``; we reproduce both, so a ``NaN`` criterion value ranks
  last and can therefore be selected exactly as in catR.
* catR reseeds the global RNG (``set.seed(random.seed)`` before the draw and
  ``set.seed(NULL)`` at function exit), so its selection is only reproducible
  for an isolated call. catpie instead takes ``seed``/``rng`` and never touches
  global state, so a whole adaptive session is reproducible.

Note that catR uses ``randomesque`` differently per criterion: for MFI, bOpt
and thOpt it is the rank-based rule above, for MLWI/MPWI/KL/KLP/GDI/GDIP/MEI/
MEPV it is a threshold on the criterion value over *all* items, and for
``random``, ``progressive`` and ``proportional`` it is ignored. catpie
currently implements MFI and bOpt only, and those two use the rank-based rule.
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple, Optional, Sequence, Tuple, Union

from .irf import Item, ii

#: Either an explicit ``random.Random`` instance or ``None`` for the global RNG.
Rng = Optional[random.Random]


class NextItemResult(NamedTuple):
    """Result of an item selection (catR ``nextItem()``)."""

    item: int  # 0-indexed selection
    par: Item  # the selected item's parameters
    info: float  # criterion value at the selection (Ii for MFI, |b - theta| for bOpt)
    criterion: str
    randomesque: int = 1  # the randomesque value used (catR echoes it back)
    candidates: Tuple[int, ...] = ()  # the eligible set one candidate was drawn from


def _resolveRng(seed: Optional[int], rng: Rng) -> Union[random.Random, "random"]:
    """Return the random source to use: explicit ``rng``, ``seed``, else global."""
    if rng is not None:
        return rng
    if seed is not None:
        return random.Random(seed)
    return random


def _rankAverage(values: Sequence[float]) -> list:
    """
    R's ``rank()`` with default ``ties.method = "average"`` and ``na.last = TRUE``.

    Tied values share the mean of the ranks they span, and ``NaN`` values are
    ranked last (so they are treated as the *largest* values, exactly like
    catR; see the module docstring).
    """
    order = sorted(
        range(len(values)),
        key=lambda i: (math.isnan(values[i]), values[i]),
    )
    ranks = [0.0] * len(values)
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n and (
            values[order[j + 1]] == values[order[i]]
        ):
            j += 1
        average = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def nextItem(
    itemBank: Sequence[Item],
    theta: float,
    out: Sequence[int] = (),
    criterion: str = "MFI",
    randomesque: int = 1,
    D: float = 1.0,
    seed: Optional[int] = None,
    rng: Rng = None,
) -> NextItemResult:
    """
    Select the next (most informative) item to administer.

    This is the heart of adaptive testing: given the current ability estimate
    ``theta`` and the items already used (``out``), it picks the single item
    that will tell you the most about the person's true ability. You then
    administer that item, score the response, re-estimate ability, and repeat.

    Two selection criteria are available:

    - ``"MFI"`` (default): Maximum Fisher Information — picks the item with
      the highest information at the current ability estimate (see :func:`ii`).
      Usually a "middle-difficulty" item around the current theta.
    - ``"bOpt"``: picks the item whose difficulty ``b`` is closest to the
      current ability estimate (simpler, often used early in a test).

    With ``randomesque = 1`` (the catR default) the single best item is
    selected, and items tied with it share the selection probability. With
    ``randomesque = k > 1`` the ``k`` best eligible items (plus anything tied
    with the ``k``-th best) form the candidate set and one of them is drawn
    uniformly at random. That is catR's exposure-control mechanism: it
    prevents every participant from seeing the same items in the same order at
    the cost of a little measurement precision. The candidate set is returned
    in :attr:`NextItemResult.candidates` so callers can log or audit it.

    Args:
        itemBank: The full item bank, a list of ``(a, b, c, d)`` tuples.
        theta: Current ability estimate (a float, typically ``[-4, 4]``).
        out: Indices (0-based) of items that have already been administered;
            they will not be selected again. Default: none.
        criterion: ``"MFI"`` (default) or ``"bOpt"``.
        randomesque: Number of best items to draw uniformly from (``>= 1``,
            default ``1``, matching catR). Must be an integer.
        D: Scale constant (default ``1.0``).
        seed: Optional seed for the draw among the candidates (catR's
            ``random.seed``). Ignored if ``rng`` is given. With ``seed`` set,
            this call is reproducible; without ``seed`` and ``rng`` the global
            ``random`` module is used, like any other Python code.
        rng: Optional :class:`random.Random` instance used for the draw. Pass
            one object through a whole session to make an adaptive test fully
            reproducible (catR cannot do that, see the module docstring).

    Returns:
        A :class:`NextItemResult` with fields ``item`` (the 0-based index of
        the selected item in the bank), ``par`` (its parameters), ``info``
        (the criterion value of the *selected* item: information for MFI,
        ``|b - theta|`` for bOpt), ``criterion``, ``randomesque`` and
        ``candidates`` (the tuple of 0-based indices one candidate was drawn
        from, in bank order).

    Raises:
        ValueError: If ``criterion`` is not ``"MFI"``/``"bOpt"``, if
            ``randomesque`` is not a positive integer, or if no eligible item
            is left. The last case is an error in catpie; catR's behaviour
            there is undefined.
    """
    if criterion not in ("MFI", "bOpt"):
        raise ValueError(
            "nextItem: criterion %r not implemented (only 'MFI', 'bOpt')" % (criterion,)
        )
    if isinstance(randomesque, bool) or not isinstance(randomesque, int):
        raise ValueError(
            "nextItem: randomesque must be a positive integer, got %r" % (randomesque,)
        )
    if randomesque < 1:
        raise ValueError(
            "nextItem: randomesque must be >= 1, got %r" % (randomesque,)
        )

    n = len(itemBank)
    out_set = set(out)
    eligible = [i for i in range(n) if i not in out_set]
    if not eligible:
        raise ValueError(
            "nextItem: no eligible item left (bank has %d items, %d administered)"
            % (n, len(out_set))
        )

    # Criterion value per item: information for MFI, distance for bOpt (catR's
    # bOpt sorts the distance in ascending order).
    if criterion == "MFI":
        values = [ii(theta, item, D).Ii for item in itemBank]
        descending = True
    else:
        values = [abs(item[1] - theta) for item in itemBank]
        descending = False

    ranks = _rankAverage(values)

    # catR: nrIt <- min(c(randomesque, sum(items))) and the top nrIt ranks of
    # the eligible items; every eligible item sharing one of those ranks is a
    # candidate (so ties at the k-th position widen the candidate set).
    nrIt = min(randomesque, len(eligible))
    keepRank = sorted((ranks[i] for i in eligible), reverse=descending)[:nrIt]
    candidates = [i for i in eligible if ranks[i] in keepRank]

    # catR: select <- ifelse(length(keep) == 1, keep, sample(keep, 1))
    if len(candidates) == 1:
        select = candidates[0]
    else:
        select = _resolveRng(seed, rng).choice(candidates)

    return NextItemResult(
        select,
        itemBank[select],
        values[select],
        criterion,
        randomesque,
        tuple(candidates),
    )

