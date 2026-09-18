"""
Parity test against catR on one recorded EWM session.

Background
----------
This session is a real participant's 35 trials from the `psynet-ewmadults-adaptive`
experiment. The participant answered 33 of 35 items correctly, yet the EAP ability
estimate stays at ~0.0009 and the administered items never get harder. That looks
like a bug in catpie's item selection or estimation.

It is not. This test replays the *exact* item bank, the exact sequence of
administered items and the exact responses through catpie and compares against
the output of the REAL R catR package for the same session. catR and catpie agree
to machine precision, which pins the behaviour on the item bank's 4PL
discrimination parameters (20% of items have a < 0, 52% have |a| > 5, range
-65.97 .. 69.30) and not on the port.

The fixture and the reference are both self-contained (the item bank is embedded),
so these tests do not depend on the experiment repository and do not change when
that repository's item bank is revised.

  reference/ewm_session_input.json      inputs (bank + administered + responses)
  reference/catr_ewm_session.json       catR ground truth
                                        (scripts/generate_ewm_session_reference.R)

To re-check against R, run: `python scripts/validate_ewm_session.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from catpie import estimateTheta, selectNextItem

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "reference/ewm_session_input.json"
REFERENCE_PATH = ROOT / "reference/catr_ewm_session.json"

# catR's EAP is the same numerical integration over the same grid, so agreement
# should be at machine precision. The tolerances leave a little room for
# platform-dependent floating point differences.
TOL_THETA = 1e-9
TOL_SE = 1e-8


def _load():
    with INPUT_PATH.open(encoding="utf-8") as handle:
        fixture = json.load(handle)
    with REFERENCE_PATH.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    return fixture, reference


def _replay(fixture):
    """Replay the logged session with catpie, as the experiment does."""
    bank = [tuple(item) for item in fixture["item_bank"]]
    administered = list(fixture["administered"])
    responses = list(fixture["responses"])

    selected, thetas, ses = [], [], []
    adm: list[int] = []
    resp: list[int] = []
    theta = 0.0
    for index, response in zip(administered, responses):
        picked = selectNextItem(bank, theta, adm, criterion="MFI")
        selected.append(picked.item)
        adm.append(index)
        resp.append(response)
        result = estimateTheta(bank, adm, resp, method="EAP")
        theta = result.theta
        thetas.append(result.theta)
        ses.append(result.se)
    return selected, thetas, ses


@pytest.fixture(scope="module")
def session():
    fixture, reference = _load()
    selected, thetas, ses = _replay(fixture)
    return fixture, reference, selected, thetas, ses


def test_session_reproduces_catr_item_selection(session):
    """MFI must pick the same item as catR at every step of the session."""
    _, reference, selected, _, _ = session
    expected = [step["administered_item_0based"] for step in reference["steps"]]
    assert len(selected) == len(expected)
    assert selected == expected


def test_session_reproduces_catr_ability_estimates(session):
    """EAP ability estimates must match catR to machine precision."""
    _, reference, _, thetas, _ = session
    expected = [step["theta"] for step in reference["steps"]]
    differences = [abs(a - b) for a, b in zip(thetas, expected)]
    assert max(differences) <= TOL_THETA, (
        f"max |dtheta| = {max(differences):.3e} exceeds {TOL_THETA:.0e}"
    )


def test_session_reproduces_catr_standard_errors(session):
    """EAP standard errors must match catR to machine precision."""
    _, reference, _, _, ses = session
    expected = [step["se"] for step in reference["steps"]]
    differences = [abs(a - b) for a, b in zip(ses, expected)]
    assert max(differences) <= TOL_SE, (
        f"max |dse| = {max(differences):.3e} exceeds {TOL_SE:.0e}"
    )


def test_session_selection_info_matches_catr(session):
    """The information values behind the MFI choice must match catR too."""
    fixture, reference, _, _, _ = session
    bank = [tuple(item) for item in fixture["item_bank"]]
    administered = list(fixture["administered"])

    adm: list[int] = []
    theta = 0.0
    for step in reference["steps"]:
        picked = selectNextItem(bank, theta, adm, criterion="MFI")
        assert abs(picked.info - step["info"]) <= 1e-6
        adm.append(administered[step["trial"] - 1])
        theta = estimateTheta(bank, adm, list(fixture["responses"])[: len(adm)]).theta


def test_session_is_a_catr_reproduced_pathology(session):
    """
    Documents *why* this fixture exists.

    A participant with 33/35 correct responses should end up with a high ability
    estimate and progressively harder items. Instead the estimate crawls from
    -0.774 to 0.000865 and the administered difficulty barely moves, because the
    bank's discrimination parameters make Fisher information - and hence MFI's
    choice - depend on |a| rather than on how well an item targets theta.

    catR produces the same numbers (see the parity tests above), so this is an
    item-bank problem. If this test ever fails, the fixture or the bank changed:
    re-read the accompanying analysis before relaxing it.
    """
    fixture, reference, _, thetas, ses = session

    correct = sum(fixture["responses"])
    assert correct / fixture["n_trials"] > 0.9, "fixture should be a near-perfect session"

    # The estimate stays pinned near the prior mean ...
    assert abs(thetas[-1]) < 0.01
    # ... while catR agrees, so it is not a porting artefact.
    assert abs(thetas[-1] - reference["steps"][-1]["theta"]) <= TOL_THETA

    # ... and the standard error grows instead of shrinking for most of the test.
    assert ses[-1] > ses[5]

    # The bank is the reason: it is full of extreme discrimination values.
    a_values = [item[0] for item in fixture["item_bank"]]
    assert sum(1 for a in a_values if a < 0) == 29
    assert sum(1 for a in a_values if abs(a) > 5) == 76
