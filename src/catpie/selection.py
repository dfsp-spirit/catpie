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

from .criteria import kl, likelihood, mwi
from .irf import Item, ii
from .math import linspace

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


#: Selection criteria implemented by catpie (a subset of catR's fourteen).
_CRITERIA = (
    "MFI",
    "bOpt",
    "thOpt",
    "MLWI",
    "MPWI",
    "KL",
    "KLP",
    "progressive",
    "proportional",
    "random",
)

#: Criteria ranked with the *largest* value first (the others use distance).
_RANK_DESCENDING = {"MFI": True, "bOpt": False, "thOpt": False}


def _thetaOptimum(item: Item) -> float:
    """
    Ability at which a 4PL item reaches its maximum information (catR ``thOpt``).

    Closed-form solution of the cubic that catR solves for the item; under the
    1PL/2PL (``c = 0``, ``d = 1``) it collapses to the item difficulty.
    """
    a, b, c, d = item
    u = -3.0 / 4.0 + (c + d - 2.0 * c * d) / 2.0
    v = (c + d - 1.0) / 4.0
    xstar = 2.0 * math.sqrt(-u / 3.0) * math.cos(
        math.acos(-v * math.sqrt(-27.0 / u**3) / 2.0) / 3.0 + 4.0 * math.pi / 3.0
    ) + 0.5
    return b + math.log((xstar - c) / (d - xstar)) / a


def _rankValues(
    itemBank: Sequence[Item], theta: float, criterion: str, D: float
) -> list:
    """Criterion value per item for the rank-based criteria."""
    if criterion == "MFI":
        return [ii(theta, item, D).Ii for item in itemBank]
    if criterion == "bOpt":
        return [abs(item[1] - theta) for item in itemBank]
    return [abs(_thetaOptimum(item) - theta) for item in itemBank]


def _rankSelect(
    values: Sequence[float],
    available: Sequence[int],
    randomesque: int,
    descending: bool,
    source,
) -> Tuple[int, list]:
    """
    catR's rank-based candidate rule (MFI, bOpt, thOpt).

    The top ``min(randomesque, n_available)`` *ranks* of the available items are
    kept and every available item sharing one of those ranks becomes a
    candidate, so ties widen the candidate set beyond ``randomesque`` items.
    One candidate is then drawn uniformly at random.
    """
    ranks = _rankAverage(values)
    nrIt = min(randomesque, len(available))
    keepRank = sorted((ranks[i] for i in available), reverse=descending)[:nrIt]
    candidates = [i for i in available if ranks[i] in keepRank]
    if len(candidates) == 1:
        return candidates[0], candidates
    return source.choice(candidates), candidates


def _thresholdSelect(
    values: Sequence[float],
    available: Sequence[int],
    randomesque: int,
    source,
) -> Tuple[int, list]:
    """
    catR's threshold rule for MLWI/MPWI/KL/KLP: keep the available items whose
    value reaches the ``randomesque``-th largest value.

    catR builds this candidate set over the *whole* bank, so an administered
    item with a value of exactly 0 can be re-selected; catpie restricts it to
    the available items (documented deviation).
    """
    nrIt = min(randomesque, len(available))
    threshold = sorted((values[i] for i in available), reverse=True)[nrIt - 1]
    candidates = [i for i in available if values[i] >= threshold]
    if len(candidates) == 1:
        return candidates[0], candidates
    return source.choice(candidates), candidates


def _weightedValues(
    itemBank: Sequence[Item],
    theta: float,
    criterion: str,
    out: Sequence[int],
    x: Optional[Sequence[int]],
    available: Sequence[int],
    source,
    priorDist: str,
    priorPar: Tuple[float, float],
    parInt: Tuple[float, float, int],
    D: float,
) -> list:
    """Criterion values for MLWI, MPWI, KL and KLP (non-candidates get 0)."""
    if x is None:
        raise ValueError(
            "nextItem: criterion %r requires the provisional responses `x`" % (criterion,)
        )
    if len(x) != len(out):
        raise ValueError(
            "nextItem: `x` must have one response per administered item "
            "(got %d responses for %d items)" % (len(x), len(out))
        )
    itGiven = [itemBank[i] for i in out]
    grid = linspace(parInt[0], parInt[1], int(parInt[2]))
    lik = likelihood(x, itGiven, grid, D)
    values = [0.0] * len(itemBank)
    for i in available:
        if criterion in ("MLWI", "MPWI"):
            values[i] = mwi(
                itemBank, i, x, itGiven,
                type=criterion, priorDist=priorDist, priorPar=priorPar,
                lower=parInt[0], upper=parInt[1], nqp=int(parInt[2]), D=D, lik=lik,
            )
        else:
            values[i] = kl(
                itemBank, i, x, itGiven, theta=theta,
                type=criterion, priorDist=priorDist, priorPar=priorPar,
                lower=parInt[0], upper=parInt[1], nqp=int(parInt[2]), D=D, lik=lik,
            )
    return values


def _progressiveWeight(
    nAdministered: int,
    rule: str,
    thr: float,
    SETH: Optional[float],
    AP: float,
    maxItems: int,
) -> float:
    """
    catR's acceleration weight ``wq`` for the progressive/proportional methods.

    With ``rule="length"`` it ramps from 0 (first item, purely random) to 1 at
    item ``thr``; with ``rule="precision"`` it follows the ratio of the target
    to the actual test information. ``AP`` is the acceleration parameter.
    """
    if rule == "length":
        if nAdministered <= 0:
            return 0.0
        numerator = sum(i**AP for i in range(1, nAdministered + 1))
        denominator = sum(i**AP for i in range(1, int(thr)))
        return numerator / denominator
    if rule == "precision":
        if nAdministered <= 0:
            return 0.0
        if SETH is None:
            raise ValueError(
                'nextItem: rule="precision" requires SETH (the target standard error)'
            )
        infoStop = (1.0 / thr) ** 2
        cumInfo = (1.0 / SETH) ** 2
        return max(cumInfo / infoStop, nAdministered / (maxItems - 1)) ** AP
    raise ValueError("nextItem: rule must be 'length' or 'precision', got %r" % (rule,))


def _progressiveValues(
    itemBank: Sequence[Item],
    theta: float,
    criterion: str,
    out: Sequence[int],
    available: Sequence[int],
    D: float,
    rule: str,
    thr: float,
    SETH: Optional[float],
    AP: float,
    maxItems: int,
    parInt: Tuple[float, float, int],
    source,
) -> Tuple[list, list]:
    """Information plus the blended progressive criterion (catR's ``funcPR``)."""
    del parInt  # catR ignores the integration grid for this criterion
    values = [ii(theta, item, D).Ii for item in itemBank]
    wq = _progressiveWeight(len(out), rule, thr, SETH, AP, maxItems)
    maxInfo = max(values[i] for i in available)
    randomValues = [source.uniform(0.0, maxInfo) for _ in itemBank]
    funcPR = [0.0] * len(itemBank)
    for i in available:
        funcPR[i] = values[i] * wq + randomValues[i] * (1.0 - wq)
    return values, funcPR


def _proportionalWeights(
    itemBank: Sequence[Item],
    theta: float,
    out: Sequence[int],
    available: Sequence[int],
    D: float,
    rule: str,
    thr: float,
    SETH: Optional[float],
    AP: float,
    maxItems: int,
) -> Tuple[list, list]:
    """
    Selection weights for the proportional method: ``Ii ^ wq``.

    ``wq`` is 0 on the first item (uniform random choice) and approaches the
    exponent that makes the draw concentrate on the most informative item.
    """
    values = [ii(theta, item, D).Ii for item in itemBank]
    base = _progressiveWeight(len(out), rule, thr, SETH, AP, maxItems)
    if rule == "length":
        wq = thr * base
    else:
        wq = (1.0 / thr) ** 2 * base
    weights = [0.0] * len(itemBank)
    for i in available:
        weights[i] = values[i] ** wq
    return values, weights


def _weightedChoice(items: Sequence[int], probabilities: Sequence[float], source) -> int:
    """Draw one item with the given (already normalised) probabilities."""
    target = source.random()
    cumulative = 0.0
    for item, probability in zip(items, probabilities):
        cumulative += probability
        if target <= cumulative:
            return item
    return items[-1]



def nextItem(
    itemBank: Sequence[Item],
    theta: float,
    out: Sequence[int] = (),
    criterion: str = "MFI",
    randomesque: int = 1,
    D: float = 1.0,
    seed: Optional[int] = None,
    rng: Rng = None,
    x: Optional[Sequence[int]] = None,
    priorDist: str = "norm",
    priorPar: Tuple[float, float] = (0, 1),
    parInt: Tuple[float, float, int] = (-4.0, 4.0, 33),
    rule: str = "length",
    thr: float = 20,
    SETH: Optional[float] = None,
    AP: float = 1.0,
    maxItems: int = 50,
    nAvailable: Optional[Sequence[int]] = None,
    cbControl: Optional[dict] = None,
    cbGroup: Optional[Sequence[str]] = None,
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
        (the criterion value of the *selected* item), ``criterion``,
        ``randomesque`` and ``candidates`` (the tuple of 0-based indices one
        candidate was drawn from, in bank order). For the randomised criteria
        (``"progressive"``, ``"proportional"``, ``"random"``) every eligible
        item is a candidate.

    Raises:
        ValueError: If ``criterion`` is unknown, if ``randomesque`` is not a
            positive integer, if a criterion needs ``x``/``theta`` and it is
            missing, or if no eligible item is left. The last case is an error
            in catpie; catR's behaviour there is undefined.
        NotImplementedError: If content balancing (``cbControl``/``cbGroup``)
            is requested, which catpie does not implement.
    """
    if criterion not in _CRITERIA:
        raise ValueError(
            "nextItem: criterion %r not implemented (supported: %s)"
            % (criterion, ", ".join(_CRITERIA))
        )
    if isinstance(randomesque, bool) or not isinstance(randomesque, int):
        raise ValueError(
            "nextItem: randomesque must be a positive integer, got %r" % (randomesque,)
        )
    if randomesque < 1:
        raise ValueError(
            "nextItem: randomesque must be >= 1, got %r" % (randomesque,)
        )
    if cbControl is not None or cbGroup is not None:
        raise NotImplementedError(
            "nextItem: content balancing (cbControl/cbGroup) is not implemented in catpie"
        )

    n = len(itemBank)
    out_list = list(out)
    out_set = set(out_list)
    available = [
        i
        for i in range(n)
        if i not in out_set and (nAvailable is None or nAvailable[i] != 0)
    ]
    if not available:
        raise ValueError(
            "nextItem: no eligible item left (bank has %d items, %d administered)"
            % (n, len(out_set))
        )

    source = _resolveRng(seed, rng)

    if criterion in ("MFI", "bOpt", "thOpt"):
        values = _rankValues(itemBank, theta, criterion, D)
        select, candidates = _rankSelect(
            values, available, randomesque, _RANK_DESCENDING[criterion], source
        )
    elif criterion in ("MLWI", "MPWI", "KL", "KLP"):
        values = _weightedValues(
            itemBank, theta, criterion, out_list, x, available, source,
            priorDist, priorPar, parInt, D,
        )
        select, candidates = _thresholdSelect(values, available, randomesque, source)
    elif criterion == "progressive":
        values, funcPR = _progressiveValues(
            itemBank, theta, criterion, out_list, available, D,
            rule, thr, SETH, AP, maxItems, parInt, source,
        )
        best = max(funcPR[i] for i in available)
        candidates = [i for i in available if funcPR[i] == best]
        select = (
            candidates[0]
            if len(candidates) == 1
            else source.choice(candidates)
        )
    elif criterion == "proportional":
        values, weights = _proportionalWeights(
            itemBank, theta, out_list, available, D, rule, thr, SETH, AP, maxItems
        )
        total = sum(weights[i] for i in available)
        candidates = list(available)
        if total <= 0 or math.isnan(total):
            select = source.choice(candidates)
        else:
            select = _weightedChoice(
                candidates, [weights[i] / total for i in candidates], source
            )
    else:  # "random": catR ignores theta and randomesque here
        values = [math.nan] * n
        candidates = list(available)
        select = source.choice(candidates)

    return NextItemResult(
        select,
        itemBank[select],
        values[select],
        criterion,
        randomesque,
        tuple(candidates),
    )

