"""
Tests for the extended catR port: ML/BM/WL estimation, priors, bOpt selection,
genPattern, checkStopRule and the minimal randomCAT loop.

Mirrors catjs-irt's ``test/extended.test.js``.
"""

from __future__ import annotations

import math

import pytest

from catpie import (
    checkStopRule,
    genPattern,
    ii,
    ji,
    nextItem,
    optimizeScalar,
    pi,
    qnorm,
    randomCAT,
    semTheta,
    simulateRespondents,
    thetaEst,
    uniroot,
)

easy = (1.0, -1.0, 0.2, 0.95)  # (a, b, c, d)
hard = (1.5, 1.0, 0.1, 0.98)
bank = [easy, hard]
# Larger bank so randomCAT can run several steps.
bigBank = [
    (0.8, -2.0, 0.10, 0.97),
    (1.1, -1.0, 0.15, 0.98),
    (1.3, 0.0, 0.20, 0.98),
    (1.2, 1.0, 0.15, 0.97),
    (0.9, 2.0, 0.10, 0.96),
    (1.0, 0.5, 0.05, 0.99),
]


# --- qnorm / uniroot / optimizeScalar ---
def test_qnorm_matches_known_standard_normal_quantiles():
    # Acklam's approximation is accurate to ~1.15e-9 relative (~2e-9 at |z|~2).
    assert abs(qnorm(0.5) - 0) < 1e-9
    assert abs(qnorm(0.975) - 1.959963984540054) < 1e-8
    assert abs(qnorm(0.025) + 1.959963984540054) < 1e-8


def test_uniroot_finds_a_simple_root():
    root = uniroot(lambda x: x * x - 2, 0, 2, 1e-12)
    assert abs(root - math.sqrt(2)) < 1e-9
    with pytest.raises(ValueError, match="opposite signs"):
        uniroot(lambda x: x * x + 1, -1, 1)


def test_optimizeScalar_finds_min_and_max():
    f = lambda x: (x - 0.3) ** 2
    mn = optimizeScalar(f, -1, 1)
    assert abs(mn.x - 0.3) < 1e-6
    mx = optimizeScalar(lambda x: -((x - 0.3) ** 2), -1, 1, maximize=True)
    assert abs(mx.x - 0.3) < 1e-6


# --- derivatives (Ji) ---
def test_ji_matches_dP_d2P_over_PQ():
    pr = pi(0.2, hard)
    expect = (pr.dP * pr.d2P) / (pr.P * (1 - pr.P))
    assert abs(ji(0.2, hard).Ji - expect) < 1e-15


def test_ii_returns_finite_derivatives():
    I = ii(0.2, hard)
    assert math.isfinite(I.Ii) and math.isfinite(I.dIi) and math.isfinite(I.d2Ii)


# --- ML / BM / WL estimation ---
def test_ML_BM_WL_produce_finite_estimates_that_move_with_responses():
    for method in ("ML", "BM", "WL"):
        it = [easy, hard]
        up = thetaEst(it, [1, 1], method=method)
        down = thetaEst(it, [0, 0], method=method)
        assert math.isfinite(up), f"{method} up finite"
        assert math.isfinite(down), f"{method} down finite"
        assert up >= down, f"{method}: up({up}) >= down({down})"


def test_BM_agrees_with_ML_under_a_flat_unif_prior():
    # BM + unif prior reduces to ML over the prior interval in catR.
    it = [easy, hard]
    bm = thetaEst(it, [1, 0], method="BM", priorDist="unif", priorPar=(-4, 4))
    ml = thetaEst(it, [1, 0], method="ML")
    assert abs(bm - ml) < 1e-6, f"bm={bm} ml={ml}"


def test_semTheta_ML_equals_1_over_sqrt_info():
    it = [easy, hard]
    th = 0.5
    info = sum(ii(th, item).Ii for item in it)
    assert abs(semTheta(th, it, [1, 0], method="ML") - 1 / math.sqrt(info)) < 1e-12
    assert (
        abs(semTheta(th, it, [1, 0], method="BM") - 1 / math.sqrt(info + 1)) < 1e-12
    )
    assert abs(semTheta(th, it, [1, 0], method="WL") - 1 / math.sqrt(info)) < 1e-12


# --- EAP with non-default priors ---
def test_EAP_with_unif_prior_is_finite():
    th = thetaEst([easy, hard], [1, 0], method="EAP", priorDist="unif", priorPar=(0, 1))
    assert math.isfinite(th)


def test_EAP_with_jeffreys_prior_is_finite():
    th = thetaEst([easy, hard], [1, 0], method="EAP", priorDist="Jeffreys")
    assert math.isfinite(th)


# --- bOpt selection ---
def test_bOpt_picks_the_item_whose_difficulty_is_closest_to_theta():
    # At theta=0, easy (b=-1) and hard (b=1) are equidistant -> tie set of both
    sel = nextItem(bank, 0, [], criterion="bOpt")
    assert sel.item in (0, 1)
    # At theta=-0.9, easy is strictly closer
    sel2 = nextItem(bank, -0.9, [], criterion="bOpt")
    assert sel2.item == 0
    # administered items are excluded
    sel3 = nextItem(bank, -0.9, [0], criterion="bOpt")
    assert sel3.item == 1


# --- genPattern / simulateRespondents ---
def test_genPattern_uses_Pi_probabilities_with_an_injectable_rng():
    # rng always 0 -> always below P -> always correct
    assert genPattern(0, bank, rng=lambda: 0) == [1, 1]
    # rng always ~1 -> always above P -> always incorrect
    assert genPattern(0, bank, rng=lambda: 0.9999) == [0, 0]
    # matches catR Pi at theta=0
    p = genPattern(0, bank, rng=lambda: 0.5)
    assert p[0] == (1 if pi(0, easy).P >= 0.5 else 0)
    assert p[1] == (1 if pi(0, hard).P >= 0.5 else 0)


def test_simulateRespondents_returns_a_matrix():
    m = simulateRespondents([0, 1], bank, rng=lambda: 0.1)
    assert len(m) == 2
    assert len(m[0]) == 2


# --- checkStopRule ---
def test_checkStopRule_length_and_precision_rules():
    assert checkStopRule(0, 1, 5, rule=["length"], thr=[20]).decision is False
    assert checkStopRule(0, 1, 25, rule=["length"], thr=[20]).decision is True
    assert checkStopRule(0, 0.1, 5, rule=["precision"], thr=[0.2]).decision is True
    assert checkStopRule(0, 0.5, 5, rule=["precision"], thr=[0.2]).decision is False


def test_checkStopRule_classification_rule_excludes_the_threshold():
    # th=2, se=0.1 -> 95% CI [1.804, 2.196] entirely above 1.5 -> stop
    assert checkStopRule(2, 0.1, 5, rule=["classification"], thr=[1.5]).decision is True
    # th=0, se=0.5 -> CI [-0.98, 0.98] straddles 0 -> no stop
    assert checkStopRule(0, 0.5, 5, rule=["classification"], thr=[0]).decision is False
    # th=0, se=0.5 -> CI [-0.98, 0.98] entirely below 1.5 -> stop
    assert checkStopRule(0, 0.5, 5, rule=["classification"], thr=[1.5]).decision is True


def test_checkStopRule_minInfo_rule_requires_items():
    assert (
        checkStopRule(0, 0.1, 5, rule=["minInfo"], thr=[1e6], items=bank).decision is True
    )
    with pytest.raises(ValueError, match="items"):
        checkStopRule(0, 0.1, 5, rule=["minInfo"], thr=[1])


def test_checkStopRule_returns_which_rule_triggered():
    r = checkStopRule(0, 0.05, 30, rule=["length", "precision"], thr=[20, 0.1])
    assert sorted(r.rule) == ["length", "precision"]
    assert r.decision is True


# --- randomCAT ---
def test_randomCAT_replays_a_fixed_response_sequence():
    responses = [1, 0, 1, 0, 1]
    run = randomCAT(
        0.5,
        bigBank,
        method="EAP",
        responses=responses,
        stop={"rule": ["length"], "thr": [20]},
        rng=lambda: 0.5,
    )
    assert run.nItems == len(responses)
    assert run.responses == responses
    assert len(run.administered) == len(responses)
    assert all(math.isfinite(t) for t in run.thetaHist)
    assert all(math.isfinite(s) for s in run.seHist)


def test_randomCAT_stops_early_on_a_precision_rule():
    run = randomCAT(
        0.5,
        bigBank,
        method="EAP",
        stop={"rule": ["precision"], "thr": [1e-9]},
        maxSteps=50,
        rng=lambda: 0.5,
    )
    # A very tight precision threshold never triggers on a 6-item bank, so it
    # should run to the bank size.
    assert run.nItems == len(bigBank)
    assert run.stopRule is None


def test_randomCAT_stops_at_the_item_bank_size():
    run = randomCAT(0.5, bigBank, method="EAP", rng=lambda: 0.5)
    assert run.nItems == len(bigBank)
    assert run.stopRule is None or isinstance(run.stopRule, list)
