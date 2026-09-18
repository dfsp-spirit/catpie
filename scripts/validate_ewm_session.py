#!/usr/bin/env python3
"""
Validate catpie against catR on one recorded EWM session.

This is the "is it the port or is it the data?" check. catpie produces a
surprising result on this session (a participant answers 33 of 35 items
correctly, yet the ability estimate stays near 0 and the next items never get
harder). This script replays the exact same item bank, the exact same sequence
of administered items and the exact same responses with catpie, and compares
against the output of the REAL R catR package for the same session
(reference/catr_ewm_session.json, produced by
scripts/generate_ewm_session_reference.R).

If catR and catpie agree here, the surprising behaviour is a property of the
item parameters, not a porting artefact.

Usage:
  python scripts/validate_ewm_session.py [reference.json] [input.json]

Exit code 0 = agreement within tolerance, 1 = mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from catpie import estimateTheta, selectNextItem

ROOT = Path(__file__).resolve().parent.parent

# Tolerances: EAP is pure numerical integration over the same grid, so it should
# agree to near machine precision. The committed catR reference has 16
# significant digits.
TOL_THETA = 1e-9
TOL_SE = 1e-8
TOL_INFO = 1e-6


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv) -> int:
    reference_path = Path(argv[1]) if len(argv) > 1 else ROOT / "reference/catr_ewm_session.json"
    input_path = Path(argv[2]) if len(argv) > 2 else ROOT / "reference/ewm_session_input.json"

    for path in (reference_path, input_path):
        if not path.is_file():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2

    fixture = load(input_path)
    reference = load(reference_path)

    bank = [tuple(item) for item in fixture["item_bank"]]
    administered = list(fixture["administered"])
    responses = list(fixture["responses"])
    steps = reference["steps"]

    print("EWM session parity: catR vs catpie")
    print(f"  input      : {input_path}")
    print(f"  reference  : {reference_path} "
          f"(catR {reference.get('catr_version')}, R {reference.get('r_version')})")
    print(f"  session    : {len(administered)} trials, {sum(responses)} correct, "
          f"{len(bank)}-item bank")
    print(f"  settings   : selection={reference['selection']}, "
          f"estimation={reference['estimation']['method']}, "
          f"parInt={reference['estimation']['parInt']}")
    print()

    header = (f"  {'trial':>5} {'MFI catR':>9} {'MFI catpie':>11} {'sel':>4} "
              f"{'theta catR':>18} {'theta catpie':>18} {'|dtheta|':>10} "
              f"{'se catR':>15} {'se catpie':>15} {'|dse|':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    max_dtheta = max_dse = max_dinfo = 0.0
    sel_mismatch = 0
    adm: list[int] = []
    resp: list[int] = []
    theta = 0.0

    for step in steps:
        k = step["trial"]

        # Same MFI call the experiment makes, at the estimate from the previous trial.
        picked = selectNextItem(bank, theta, adm, criterion="MFI")
        d_info = abs(picked.info - step["info"])
        max_dinfo = max(max_dinfo, d_info)
        ok_sel = picked.item == step["administered_item_0based"]
        sel_mismatch += not ok_sel

        # Administer the *logged* item and response, then re-estimate.
        adm.append(administered[k - 1])
        resp.append(responses[k - 1])
        result = estimateTheta(bank, adm, resp, method="EAP")

        d_theta = abs(result.theta - step["theta"])
        d_se = abs(result.se - step["se"])
        max_dtheta = max(max_dtheta, d_theta)
        max_dse = max(max_dse, d_se)

        # Only print every trial when there is something to see, else sample.
        if ok_sel is False or d_theta > TOL_THETA or d_se > TOL_SE or k <= 3 or k % 10 == 0:
            print(f"  {k:>5} {step['administered_item_0based']:>9} {picked.item:>11} "
                  f"{'ok' if ok_sel else 'BAD':>4} "
                  f"{step['theta']:>18.12f} {result.theta:>18.12f} {d_theta:>10.2e} "
                  f"{step['se']:>15.12f} {result.se:>15.12f} {d_se:>9.2e}")

        theta = result.theta

    print()
    print("  summary")
    print(f"    selection mismatches : {sel_mismatch}/{len(steps)} "
          f"(max |dinfo| = {max_dinfo:.2e}, tolerance {TOL_INFO:.0e})")
    print(f"    max |dtheta|         : {max_dtheta:.3e} (tolerance {TOL_THETA:.0e})")
    print(f"    max |dse|            : {max_dse:.3e} (tolerance {TOL_SE:.0e})")
    print()

    ok = (sel_mismatch == 0 and max_dtheta <= TOL_THETA
          and max_dse <= TOL_SE and max_dinfo <= TOL_INFO)

    if ok:
        print("  RESULT: PASS - catR and catpie agree on this session.")
        print()
        print(f"  Note: the session ends at theta = {theta:.6f} despite "
              f"{sum(responses)}/{len(responses)} correct responses.")
        print("  catR produces the same frozen estimate, so this is a property of the")
        print("  item bank (implausible 4PL discrimination values), not of catpie.")
        return 0

    print("  RESULT: FAIL - catpie does not reproduce catR on this session.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
