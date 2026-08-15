"""
Unit tests for the core catR port (mirrors catjs-irt's ``test/catr.test.js``).

Reference values are computed from the catR source formulas (eapEst.R,
eapSem.R, Pi.R, Ii.R, nextItem.R, integrate.catR.R). For true end-to-end
parity against the real R package, run ``uv run python scripts/validate.py``,
which replays committed catR ground-truth references.
"""

from __future__ import annotations

import math

import pytest

from catpie import (
    dnorm,
    eapEst,
    eapSem,
    estimateTheta,
    ii,
    integrateCatR,
    linspace,
    nextItem,
    pi,
    selectNextItem,
    semTheta,
    thetaEst,
)

# A couple of items with normal parameters (a, b, c, d)
easy = (1.0, -1.0, 0.2, 0.95)
hard = (1.5, 1.0, 0.1, 0.98)


def test_dnorm_matches_standard_normal_density():
    assert abs(dnorm(0) - 0.3989422804014327) < 1e-15
    assert abs(dnorm(1) - 0.24197072451914337) < 1e-15


def test_linspace_matches_r_seq():
    assert linspace(0, 1, 5) == [0, 0.25, 0.5, 0.75, 1]
    x = linspace(-4, 4, 33)
    assert len(x) == 33
    assert x[0] == -4
    assert x[32] == 4
    assert x[16] == 0


def test_integrateCatR_integrates_a_constant():
    x = [0, 1, 2, 3]
    y = [2, 2, 2, 2]
    assert integrateCatR(x, y) == 6


def test_pi_matches_4pl_formula_with_catr_clamping():
    pr = pi(0, easy)
    # e = exp(1 * 1.0 * (0 - (-1))) = e^1
    e = math.exp(1)
    P_expect = 0.2 + (0.95 - 0.2) * (e / (1 + e))
    dP_expect = (1 * 1.0 * e * (0.95 - 0.2)) / (1 + e) ** 2
    assert abs(pr.P - P_expect) < 1e-15
    assert abs(pr.dP - dP_expect) < 1e-15
    # monotonic in theta
    assert pi(-5, hard).P < pi(0, hard).P < pi(5, hard).P
    # bounds
    p_lo = pi(-100, hard).P
    p_hi = pi(100, hard).P
    assert p_lo >= hard[2] and p_hi <= hard[3]


def test_pi_clamps_P0_like_catr_and_matches_overflow_behavior():
    # Item with c=0 and theta far below b: exp underflows to 0 -> P=0 exactly,
    # which catR clamps to 1e-10.
    item = (1, 0, 0, 1)
    far_low = pi(-1000, item).P
    assert far_low == 1e-10
    # theta far above b: exp overflows to Inf -> Inf/(1+Inf) = NaN, exactly as
    # in R/catR (the Pi==1 clamp is effectively unreachable for the logistic).
    far_high = pi(1000, item).P
    assert math.isnan(far_high), "catR gives NaN on logistic overflow"


def test_ii_matches_dP2_over_PQ():
    pr = pi(0.5, hard)
    expect = (pr.dP * pr.dP) / (pr.P * (1 - pr.P))
    assert abs(ii(0.5, hard).Ii - expect) < 1e-15


def test_eapEst_returns_prior_mean_with_single_weak_response():
    # One easy item, correct answer: EAP should be > 0 but small
    th = eapEst([easy], [1])
    assert 0 < th < 2, f"expected small positive, got {th}"
    # all-correct vs all-incorrect on the same item must move in the right direction
    up = eapEst([easy], [1])
    down = eapEst([easy], [0])
    assert up > down, f"expected up({up}) > down({down})"


def test_eapSem_is_finite_and_positive_for_a_real_pattern():
    it = [easy, hard]
    x = [1, 0]
    th = eapEst(it, x)
    se = eapSem(th, it, x)
    assert math.isfinite(se) and se > 0


def test_thetaEst_semTheta_mirror_eapEst_eapSem():
    it = [easy, hard]
    x = [1, 0]
    assert thetaEst(it, x, method="EAP") == eapEst(it, x)
    th = thetaEst(it, x)
    assert semTheta(th, it, x, method="EAP") == eapSem(th, it, x)


def test_thetaEst_rejects_unimplemented_methods():
    # EAP/BM/ML/WL are implemented; ROB and others are not.
    with pytest.raises(ValueError, match="not implemented"):
        thetaEst([easy], [1], method="ROB")
    for method in ("ML", "BM", "WL"):
        thetaEst([easy], [1], method=method)  # must not raise


def test_nextItem_picks_the_item_with_maximum_information():
    # At theta very negative, easy (-1) has more info than hard (+1)
    bank = [hard, easy]
    sel = nextItem(bank, -5, [], criterion="MFI")
    assert sel.item == 1  # easy item
    sel2 = nextItem(bank, 5, [], criterion="MFI")
    assert sel2.item == 0  # hard item


def test_nextItem_never_reselects_an_administered_item():
    bank = [easy, hard]
    sel = nextItem(bank, 0, [1], criterion="MFI")
    assert sel.item != 1


def test_estimateTheta_with_no_items_returns_the_prior():
    res = estimateTheta([], [], [])
    assert res.theta == 0.0
    assert res.se == float("inf")


def test_estimateTheta_matches_thetaEst_semTheta_on_a_pattern():
    bank = [easy, hard]
    res = estimateTheta(bank, [0, 1], [1, 0])
    th = thetaEst([easy, hard], [1, 0])
    se = semTheta(th, [easy, hard], [1, 0])
    assert res.theta == th
    assert res.se == se


def test_selectNextItem_wraps_nextItem_with_MFI():
    bank = [hard, easy]
    sel = selectNextItem(bank, -5, [])
    assert sel.item == 1
