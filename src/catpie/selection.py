"""
Item selection, mirroring catR's ``nextItem()`` with the criteria used by
catpie: "MFI" (Maximum Fisher Information) and "bOpt" (difficulty closest to
current ability), both with ``randomesque = 1`` (catR's default).

catR's MFI branch:

    items  <- rep(1, nrow(itemBank)); items[OUT] <- 0
    info   <- Ii(theta, itemBank)$Ii
    ranks  <- rank(info)
    nrIt   <- min(c(randomesque, sum(items)))          # = 1 for randomesque=1
    keepRank <- sort(ranks[items == 1], decreasing = TRUE)[1:nrIt]
    keep   <- which(ranks == keepRank[i] & items == 1) # all eligible items tied at the top
    select <- ifelse(length(keep) == 1, keep, sample(keep, 1))

Because ``rank()`` is monotone, ties share the same rank, so the eligible
candidates at the optimum are exactly the items tied at the optimum; catR
samples one of them at random. We replicate that without implementing R's
average-rank tie handling (exactly like catjs-irt's ``src/selection.js``).
"""

from __future__ import annotations

import random
from typing import NamedTuple, Optional, Sequence

from .irf import Item, ii


class NextItemResult(NamedTuple):
    """Result of an item selection (catR ``nextItem()``)."""

    item: int  # 0-indexed selection
    par: Item  # the selected item's parameters
    info: float  # criterion value at the selection (Ii for MFI, |b - theta| for bOpt)
    criterion: str


def nextItem(
    itemBank: Sequence[Item],
    theta: float,
    out: Sequence[int] = (),
    criterion: str = "MFI",
    randomesque: int = 1,
    D: float = 1.0,
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

    When several items tie for the best criterion value, one of them is chosen
    at random (catR's ``randomesque = 1`` behaviour).

    Args:
        itemBank: The full item bank, a list of ``(a, b, c, d)`` tuples.
        theta: Current ability estimate (a float, typically ``[-4, 4]``).
        out: Indices (0-based) of items that have already been administered;
            they will not be selected again. Default: none.
        criterion: ``"MFI"`` (default) or ``"bOpt"``.
        randomesque: Only the catR default ``1`` is supported.
        D: Scale constant (default ``1.0``).

    Returns:
        A :class:`NextItemResult` with fields ``item`` (the 0-based index of
        the selected item in the bank), ``par`` (its parameters), ``info``
        (the criterion value: information for MFI, ``|b - theta|`` for bOpt)
        and ``criterion``.

    Raises:
        ValueError: If ``criterion`` is not ``"MFI"``/``"bOpt"`` or
            ``randomesque`` is not 1.
    """
    if criterion not in ("MFI", "bOpt"):
        raise ValueError(
            "nextItem: criterion %r not implemented (only 'MFI', 'bOpt')" % (criterion,)
        )
    if randomesque != 1:
        raise ValueError(
            "nextItem: randomesque=%r not implemented (only the default 1)" % (randomesque,)
        )

    n = len(itemBank)
    out_set = set(out)

    # Criterion value per item (higher is better for MFI, lower is better for
    # bOpt, so we negate for a uniform "maximize" treatment).
    def criterionVal(item: Item, i: int) -> float:
        if criterion == "MFI":
            return ii(theta, item, D).Ii
        return -abs(item[1] - theta)

    eligible = [i for i in range(n) if i not in out_set]

    best = max(criterionVal(itemBank[i], i) for i in eligible)

    # All eligible items tied at the optimum (catR: keep)
    keep = [i for i in eligible if criterionVal(itemBank[i], i) == best]

    # catR: select <- ifelse(length(keep) == 1, keep, sample(keep, 1))
    select = keep[0] if len(keep) == 1 else random.choice(keep)

    info = (
        ii(theta, itemBank[select], D).Ii
        if criterion == "MFI"
        else abs(itemBank[select][1] - theta)
    )
    return NextItemResult(select, itemBank[select], info, criterion)
