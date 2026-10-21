"""
Parity tests for ``randomesque > 1`` item selection against the real catR.

``reference/catr_selection.json`` is produced by
``scripts/generate_reference_selection.R`` with R + catR. For every case it
records catR's criterion vector, the item catR picks for one fixed seed, and --
crucially -- the *support* of catR's selection over many seeds, because catR
never returns the candidate set itself.

Because R and Python use different RNGs, the selected item is only compared
exactly for ``randomesque = 1`` (deterministic). For ``k > 1`` the test compares
what is actually comparable:

* the criterion vector (deterministic, catR vs catpie),
* the candidate set, derived independently from catR's recorded vector and from
  catpie's own vector,
* the support over many seeds (an item that catR can never pick must never be
  picked by catpie either),
* the uniformity of the draw over the candidate set.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from catpie import checkStopRule, ii, nextItem, selectNextItem

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "reference" / "catr_selection.json"

pytestmark = pytest.mark.skipif(
    not REFERENCE.exists(),
    reason="reference/catr_selection.json missing; run scripts/generate_reference_selection.R",
)


@pytest.fixture(scope="module")
def reference() -> dict:
    with REFERENCE.open(encoding="utf-8") as handle:
        return json.load(handle)


def criterion_values(bank, theta, criterion):
    """catpie's criterion vector (Ii for MFI, |b - theta| for bOpt)."""
    if criterion == "MFI":
        return [ii(theta, item).Ii for item in bank]
    return [abs(item[1] - theta) for item in bank]


def ranks_with_ties(values):
    """R's ``rank()``: average ranks for ties, NaN ranked last."""
    order = sorted(range(len(values)), key=lambda i: (math.isnan(values[i]), values[i]))
    ranks = [0.0] * len(values)
    i, n = 0, len(order)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def expected_candidates(values, out, k, descending):
    """Independent implementation of the documented ranking rule."""
    ranks = ranks_with_ties(values)
    drop = set(as_out_list(out))
    eligible = [i for i in range(len(values)) if i not in drop]
    n_rank = min(k, len(eligible))
    top = sorted((ranks[i] for i in eligible), reverse=descending)[:n_rank]
    return [i for i in eligible if ranks[i] in top]


def as_out_list(out):
    """``out`` may arrive as a scalar (jsonlite unboxes length-1 vectors)."""
    if isinstance(out, int):
        return [out]
    return list(out or [])


def test_criterion_vector_matches_catr(reference):
    for case in reference["cases"]:
        bank = [tuple(item) for item in reference["banks"][case["bank"]]]
        ours = criterion_values(bank, case["theta"], case["criterion"])
        assert len(ours) == len(case["info"])
        for a, b in zip(ours, case["info"]):
            assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), (
                case["criterion"],
                case["bank"],
                case["theta"],
                a,
                b,
            )


def test_candidate_set_matches_catr(reference):
    for case in reference["cases"]:
        bank = [tuple(item) for item in reference["banks"][case["bank"]]]
        k = case["randomesque"]
        descending = case["criterion"] == "MFI"

        # Candidate set implied by catR's own criterion vector ...
        from_catr = expected_candidates(case["info"], case["out"], k, descending)
        # ... and by catpie.
        result = nextItem(
            bank,
            case["theta"],
            case["out"],
            criterion=case["criterion"],
            randomesque=k,
            seed=case["seed"],
        )

        assert list(result.candidates) == expected_candidates(
            criterion_values(bank, case["theta"], case["criterion"]),
            case["out"],
            k,
            descending,
        ), (case["criterion"], case["bank"], case["theta"], case["out"], k)
        assert set(result.candidates) == set(from_catr)
        assert result.item in result.candidates
        assert result.randomesque == k

        # catR's empirical support over many seeds must be reproducible.
        support = {i for i, count in enumerate(case["counts"]) if count > 0}
        assert set(result.candidates) == support, (
            "catpie candidates differ from catR's observed support",
            case["criterion"],
            case["bank"],
            case["theta"],
            case["out"],
            k,
            sorted(set(result.candidates)),
            sorted(support),
        )


def test_selected_item_matches_catr_when_deterministic(reference):
    """With one candidate only, catR and catpie must agree exactly."""
    checked = 0
    for case in reference["cases"]:
        bank = [tuple(item) for item in reference["banks"][case["bank"]]]
        candidates = expected_candidates(
            case["info"], case["out"], case["randomesque"], case["criterion"] == "MFI"
        )
        if len(candidates) != 1:
            # catR draws at random among ties / the top-k items, and R and
            # Python use different RNGs, so only single-candidate cases are
            # comparable item by item.
            continue
        result = nextItem(
            bank,
            case["theta"],
            as_out_list(case["out"]),
            criterion=case["criterion"],
            randomesque=case["randomesque"],
            seed=case["seed"],
        )
        assert result.item == case["item"], (
            case["criterion"],
            case["bank"],
            case["theta"],
            case["out"],
        )
        checked += 1
    assert checked > 0


def test_draw_is_uniform_over_candidates(reference):
    """For k > 1 each candidate must be reachable and drawn with prob ~ 1/k."""
    checked = 0
    n_draws = 3000
    for case in reference["cases"]:
        k = case["randomesque"]
        bank = [tuple(item) for item in reference["banks"][case["bank"]]]
        candidates = expected_candidates(
            case["info"], case["out"], k, case["criterion"] == "MFI"
        )
        if k < 2 or len(candidates) < 2:
            continue

        rng = random.Random(20240101)
        tally = {}
        for _ in range(n_draws):
            item = nextItem(
                bank,
                case["theta"],
                as_out_list(case["out"]),
                criterion=case["criterion"],
                randomesque=k,
                rng=rng,
            ).item
            tally[item] = tally.get(item, 0) + 1

        assert set(tally) == set(candidates)
        for item, count in tally.items():
            assert abs(count / n_draws - 1.0 / len(candidates)) < 0.05, (
                case["criterion"],
                case["bank"],
                k,
                item,
                count,
            )
        checked += 1
    assert checked > 0


def test_ties_at_the_k_th_position_widen_the_candidate_set():
    # info(item0) > info(item1) == info(item2) > info(item3). With k = 2 the
    # second slot is a tie, so both tied items become candidates at k = 2.
    bank = [
        (1.5, 0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.5, 3.0, 0.0, 1.0),
    ]
    assert list(nextItem(bank, 0.0, [], randomesque=1, seed=1).candidates) == [0]
    assert list(nextItem(bank, 0.0, [], randomesque=2, seed=1).candidates) == [0, 1, 2]
    # A tie consumes one slot per tied item, so the third rank is still 2.5 and
    # item 3 only joins the candidate set once the tie no longer fills it.
    assert list(nextItem(bank, 0.0, [], randomesque=3, seed=1).candidates) == [0, 1, 2]
    assert list(nextItem(bank, 0.0, [], randomesque=4, seed=1).candidates) == [0, 1, 2, 3]


def test_identical_items_are_both_candidates_at_k_1():
    # Two identical items: both are candidates even for randomesque = 1.
    bank = [
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 2.0, 0.0, 1.0),
    ]
    result = nextItem(bank, 0.0, [], criterion="MFI", seed=1)
    assert list(result.candidates) == [0, 1]
    picks = {nextItem(bank, 0.0, [], criterion="MFI", seed=s).item for s in range(50)}
    assert picks == {0, 1}

    # k = 2 fills both slots with the tied pair, so nothing else qualifies.
    result = nextItem(bank, 0.0, [], criterion="MFI", randomesque=2, seed=1)
    assert list(result.candidates) == [0, 1]


def test_bopt_ties_are_candidates():
    # Items equidistant from theta: |b - theta| ties exactly.
    bank = [(1.0, -1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0, 3.0, 0.0, 1.0)]
    result = nextItem(bank, 0.0, [], criterion="bOpt", seed=1)
    assert list(result.candidates) == [0, 1]


def test_randomesque_beyond_available_items_is_clamped():
    bank = [(1.0, 0.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0, 2.0, 0.0, 1.0)]
    result = nextItem(bank, 0.0, [1], criterion="MFI", randomesque=99, seed=1)
    assert list(result.candidates) == [0, 2]
    assert result.item in (0, 2)


def test_seed_makes_selection_reproducible():
    bank = [(1.0, b, 0.0, 1.0) for b in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    first = [nextItem(bank, 0.0, [], randomesque=3, seed=7).item for _ in range(5)]
    second = [nextItem(bank, 0.0, [], randomesque=3, seed=7).item for _ in range(5)]
    assert first == second

    # A shared rng object gives a reproducible *stream* (unlike catR, which
    # reseeds the global RNG after every call).
    def stream():
        rng = random.Random(7)
        return [nextItem(bank, 0.0, [], randomesque=3, rng=rng).item for _ in range(5)]

    assert stream() == stream()


def test_select_next_item_forwards_randomesque_and_seed():
    bank = [(1.0, b, 0.0, 1.0) for b in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    direct = nextItem(bank, 0.0, [0], randomesque=2, seed=3)
    wrapped = selectNextItem(bank, 0.0, [0], randomesque=2, seed=3)
    assert direct.item == wrapped.item
    assert direct.candidates == wrapped.candidates


def test_invalid_randomesque_is_rejected():
    bank = [(1.0, 0.0, 0.0, 1.0)]
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="randomesque"):
            nextItem(bank, 0.0, [], randomesque=bad)


def test_unknown_criterion_is_rejected():
    with pytest.raises(ValueError, match="not implemented"):
        nextItem([(1.0, 0.0, 0.0, 1.0)], 0.0, [], criterion="KL")


def test_exhausted_bank_raises():
    bank = [(1.0, 0.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)]
    with pytest.raises(ValueError, match="no eligible item"):
        nextItem(bank, 0.0, [0, 1])


def test_minInfo_stop_rule_accounts_for_administered_items():
    # One informative item and one uninformative one; once the informative item
    # has been administered the rule must fire.
    bank = [(3.0, 0.0, 0.0, 1.0), (0.10, 20.0, 0.0, 1.0)]
    kwargs = dict(rule=["minInfo"], thr=[0.05], items=bank)

    assert checkStopRule(0.0, 1.0, 1, out=[], **kwargs).decision is False
    assert checkStopRule(0.0, 1.0, 1, out=[0], **kwargs).decision is True

    # Without `out` the administered item keeps the rule from firing: this is
    # the catR behaviour that motivated adding `out`.
    assert checkStopRule(0.0, 1.0, 1, **kwargs).decision is False
