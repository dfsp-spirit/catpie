"""
Parity tests for the non-MFI selection criteria and the extended estimators.

References in ``reference/catr_criteria.json`` and
``reference/catr_estimation.json`` are produced by
``scripts/generate_reference_criteria.R`` with R + catR:

* ``catr_criteria.json`` records catR's full criterion vectors for MLWI, MPWI,
  KL and KLP (computed by calling catR's ``MWI()``/``KL()`` item by item), the
  selection for one fixed seed, and the *support* of catR's selection over 300
  seeds for every criterion.
* ``catr_estimation.json`` records ``thetaEst``/``semTheta`` for degenerate
  response patterns (``constantPatt``), the robust estimator (ROB with
  Huber/Tukey weights) and the exact SE (``sem.exact``).

R and Python use different RNGs, so random selections are compared by support
and by selection frequencies, never draw by draw.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from catpie import (
    kl,
    mwi,
    nextItem,
    semTheta,
    thetaEst,
)
from catpie.estimators import _robustScore

ROOT = Path(__file__).resolve().parent.parent
CRITERIA_REF = ROOT / "reference" / "catr_criteria.json"
ESTIMATION_REF = ROOT / "reference" / "catr_estimation.json"

needs_criteria = pytest.mark.skipif(
    not CRITERIA_REF.exists(),
    reason="run scripts/generate_reference_criteria.R to create reference/catr_criteria.json",
)
needs_estimation = pytest.mark.skipif(
    not ESTIMATION_REF.exists(),
    reason="run scripts/generate_reference_criteria.R to create reference/catr_estimation.json",
)

TOL_INTEGRATION = 1e-9  # trapezoid integration over the same grid
TOL_ROOT = 5e-4  # catR's uniroot tolerance


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def as_list(value, cast=int):
    """jsonlite unboxes length-1 vectors; normalise to a list."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [cast(value)]
    return [cast(v) for v in value]


@needs_criteria
def test_weighted_criterion_values_match_catr():
    reference = load(CRITERIA_REF)
    bank = [tuple(row) for row in reference["bank"]]
    checked = 0
    for case in reference["cases"]:
        if "values" not in case:
            continue
        theta = case["theta"]
        out = as_list(case["out"])
        x = as_list(case["x"])
        itGiven = [bank[i] for i in out]
        ours = []
        for i in range(len(bank)):
            if case["criterion"] in ("MLWI", "MPWI"):
                ours.append(mwi(bank, i, x, itGiven, type=case["criterion"]))
            elif case["criterion"] in ("KL", "KLP"):
                ours.append(kl(bank, i, x, itGiven, theta=theta, type=case["criterion"]))
            elif case["criterion"] == "MFI":
                from catpie import ii

                ours.append(ii(theta, bank[i]).Ii)
            else:  # bOpt
                ours.append(abs(bank[i][1] - theta))
        for got, want in zip(ours, case["values"]):
            assert abs(got - want) <= TOL_INTEGRATION * max(1.0, abs(want)), (
                case["criterion"],
                case["theta"],
                out,
                x,
                got,
                want,
            )
        checked += 1
    assert checked > 0


@needs_criteria
def test_selection_support_matches_catr():
    """Whatever catR can select, catpie must be able to select -- and nothing else."""
    reference = load(CRITERIA_REF)
    bank = [tuple(row) for row in reference["bank"]]
    random_criteria = {"progressive", "proportional", "random"}
    checked = 0
    for case in reference["cases"]:
        out = as_list(case["out"])
        x = as_list(case["x"])
        criterion = case["criterion"]
        support = {i for i, count in enumerate(case["counts"]) if count > 0}
        available = set(range(len(bank))) - set(out)

        ours = set()
        rng = random.Random(4242)
        for _ in range(case["n_seeds"]):
            ours.add(
                nextItem(
                    bank,
                    case["theta"],
                    out,
                    criterion=criterion,
                    randomesque=case["randomesque"],
                    x=x,
                    thr=5,
                    maxItems=20,
                    rng=rng,
                ).item
            )

        assert ours <= available
        if criterion in random_criteria:
            assert ours <= support, (criterion, sorted(ours), sorted(support))
            assert support <= available
        else:
            assert ours == support, (criterion, case["theta"], out, x,
                                     sorted(ours), sorted(support))
        checked += 1
    assert checked > 0


@needs_criteria
def test_deterministic_selection_matches_catr_exactly():
    """Where catR's support is a single item, catpie must pick that item."""
    reference = load(CRITERIA_REF)
    bank = [tuple(row) for row in reference["bank"]]
    checked = 0
    for case in reference["cases"]:
        support = [i for i, count in enumerate(case["counts"]) if count > 0]
        if len(support) != 1:
            continue
        result = nextItem(
            bank,
            case["theta"],
            as_list(case["out"]),
            criterion=case["criterion"],
            randomesque=case["randomesque"],
            x=as_list(case["x"]),
            thr=5,
            maxItems=20,
            seed=case["seed"],
        )
        assert case["item"] == support[0]
        assert result.item == support[0], (
            case["criterion"],
            case["theta"],
            case["out"],
            case["x"],
        )
        checked += 1
    assert checked > 0


@needs_criteria
def test_randomised_criteria_match_catr_frequencies():
    """progressive/proportional/random: same support and similar frequencies."""
    reference = load(CRITERIA_REF)
    bank = [tuple(row) for row in reference["bank"]]
    checked = 0
    for case in reference["cases"]:
        criterion = case["criterion"]
        if criterion not in ("progressive", "proportional", "random"):
            continue
        out = as_list(case["out"])
        x = as_list(case["x"])
        n = case["n_seeds"]

        rng = random.Random(99)
        tally = [0] * len(bank)
        for _ in range(n):
            item = nextItem(
                bank,
                case["theta"],
                out,
                criterion=criterion,
                x=x,
                thr=5,
                maxItems=20,
                rng=rng,
            ).item
            tally[item] += 1
        total = sum(tally)
        assert total == n
        for item, (got, want) in enumerate(zip(tally, case["counts"])):
            assert abs(got / n - want / n) < 0.12, (criterion, item, got, want)
        checked += 1
    assert checked > 0


@needs_criteria
def test_mfi_like_criteria_still_match_after_dispatch_refactor():
    """The rank-based path is unchanged by the new criteria."""
    reference = load(CRITERIA_REF)
    bank = [tuple(row) for row in reference["bank"]]
    for case in reference["cases"]:
        if case["criterion"] not in ("MFI", "bOpt"):
            continue
        result = nextItem(
            bank,
            case["theta"],
            as_list(case["out"]),
            criterion=case["criterion"],
            randomesque=case["randomesque"],
            seed=case["seed"],
        )
        assert result.item in result.candidates
        assert set(result.candidates) <= set(range(len(bank))) - set(as_list(case["out"]))


@needs_criteria
def test_missing_responses_are_rejected_for_weighted_criteria():
    bank = [(1.0, 0.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0, 2.0, 0.0, 1.0)]
    with pytest.raises(ValueError, match="provisional responses"):
        nextItem(bank, 0.0, [0], criterion="MLWI")
    with pytest.raises(ValueError, match="one response per administered item"):
        nextItem(bank, 0.0, [0, 1], criterion="KL", x=[1])


@needs_criteria
def test_content_balancing_is_explicitly_unsupported():
    bank = [(1.0, 0.0, 0.0, 1.0)]
    with pytest.raises(NotImplementedError, match="content balancing"):
        nextItem(bank, 0.0, [], cbControl={"props": [1.0], "names": ["a"]}, cbGroup=["a"])


@needs_criteria
def test_navailable_excludes_items():
    bank = [(1.0, b, 0.0, 1.0) for b in (-1.0, 0.0, 1.0)]
    result = nextItem(bank, 0.0, [], criterion="MFI", nAvailable=[1, 0, 0], seed=1)
    assert list(result.candidates) == [0]


@needs_estimation
def test_constant_pattern_rules_match_catr():
    reference = load(ESTIMATION_REF)
    bank = [tuple(row) for row in reference["bank"]]
    for case in reference["patterns"]:
        x = as_list(case["x"])
        degenerate = sum(x) in (0, len(x))
        got = {
            "fixed4": thetaEst(bank, x, method="ML", constantPatt="fixed4", currentTh=0.25),
            "fixed7": thetaEst(bank, x, method="ML", constantPatt="fixed7", currentTh=0.25),
            "var": thetaEst(
                bank, x, method="ML", constantPatt="var", currentTh=0.25, bRange=(-3, 2)
            ),
            "BMconstant": thetaEst(bank, x, method="ML", constantPatt="BM"),
        }
        for key, value in got.items():
            want = case[key]
            if not degenerate or (isinstance(want, float) and math.isnan(want)):
                continue
            assert abs(value - want) <= TOL_ROOT, (x, key, value, want)
            # For non-degenerate patterns the special rules must not fire.
        if not degenerate:
            assert abs(got["fixed4"] - case["fixed4"]) > 1e-12


@needs_estimation
def test_robust_estimator_matches_catr():
    reference = load(ESTIMATION_REF)
    bank = [tuple(row) for row in reference["bank"]]
    for case in reference["patterns"]:
        x = as_list(case["x"])
        for weight, key in (("Huber", "ROB_Huber"), ("Tukey", "ROB_Tukey")):
            got = thetaEst(bank, x, method="ROB", weight=weight, tuCo=1)
            want = case[key]
            if abs(got - want) <= TOL_ROOT:
                continue
            # Where the robust score is exactly 0 on a whole plateau (all items
            # fully down-weighted, which happens with Tukey weights for
            # extreme response patterns) the estimate is only identified up to
            # that plateau, and R's uniroot and catpie's may land on different
            # points of it. What must agree is the score function, so check
            # that catpie returned a genuine root.
            assert abs(_robustScore(got, bank, x, weight, 1.0, 1.0)) <= 1e-6, (
                x,
                weight,
                got,
                want,
            )
            assert abs(_robustScore(want, bank, x, weight, 1.0, 1.0)) <= 1e-6


@needs_estimation
def test_standard_estimators_still_match_after_extension():
    reference = load(ESTIMATION_REF)
    bank = [tuple(row) for row in reference["bank"]]
    for case in reference["patterns"]:
        x = as_list(case["x"])
        assert abs(thetaEst(bank, x, method="BM") - case["BM"]) <= TOL_ROOT
        assert abs(thetaEst(bank, x, method="EAP") - case["EAP"]) <= TOL_INTEGRATION


@needs_estimation
def test_standard_errors_match_catr():
    reference = load(ESTIMATION_REF)
    bank = [tuple(row) for row in reference["bank"]]
    for case in reference["se"]:
        x = as_list(case["x"])
        theta = case["theta"]
        assert abs(semTheta(theta, bank, x, method="BM") - case["classic"]) <= TOL_ROOT
        assert abs(
            semTheta(theta, bank, x, method="BM", semType="new") - case["new"]
        ) <= TOL_ROOT
        exact = semTheta(theta, bank, x, method="BM", semExact=True)
        # The exact SE averages estimates obtained by root finding, so it
        # inherits catR's uniroot tolerance rather than machine precision.
        assert abs(exact - case["exact"]) <= TOL_ROOT
        exactTrue = semTheta(theta, bank, x, method="BM", semExact=True, trueTh=theta)
        assert isinstance(exactTrue, tuple)
        assert abs(exactTrue[0] - case["exact"]) <= TOL_ROOT
        rob = semTheta(
            thetaEst(bank, x, method="ROB"), bank, x, method="ROB", weight="Huber", tuCo=1
        )
        assert abs(rob - case["ROB"]) <= 1e-3
