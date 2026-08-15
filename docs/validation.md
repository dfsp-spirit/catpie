# Parity with catR

catpie is a **faithful translation** of the R package
[catR](https://cran.r-project.org/package=catR). "Faithful" here is not just a
promise — it is **verified automatically** by a validation harness that
compares catpie's output against the *real* R `catR` package, on every change,
in continuous integration.

## How it works

1. **Ground truth.** R scripts (`scripts/generate_reference.R`,
   `scripts/generate_reference_goodbank.R`,
   `scripts/generate_reference_derivatives.R`) run the real `catR` package to
   produce reference JSON files (ability estimates, standard errors, item
   information, item selections, derivatives) for simulated participants.
   These files are **committed** to the repository under `reference/`.
2. **Replay.** The script `scripts/validate.py` replays every simulated step
   with catpie and reports the maximum absolute differences.
3. **Continuous integration.** The [`Validate` workflow](https://github.com/dfsp-spirit/catpie/actions/workflows/validate.yml)
   runs the replay on every push to `main` and on every pull request, and fails
   if the differences exceed the documented tolerances.

You can run the checks locally (no R needed — the references are already
committed):

```shell
uv run python scripts/validate.py
uv run python scripts/validate.py reference/catr_reference_goodbank.json
```

> **Tie-breaking note.** catR samples *randomly* among items that tie for the
> best selection criterion. The validation therefore checks the criterion
> values (`Ii`, `|b − θ|`) exactly, and checks item *selection* for
> tie-consistency (i.e. that the chosen item belongs to the tie set), rather
> than demanding a bit-identical random pick.

## Measured results

The references were generated with **catR 3.17 / R 4.6.1** on this machine.

### Real EWM item bank (145 items) — the experiment's exact path (EAP + MFI + bOpt)

| Quantity | Comparisons | Max \|diff\| |
|----------|-------------|--------------|
| ability estimate `theta` (EAP) | 500 | 8.9e-16 |
| standard error `se` (EAP) | 500 | 2.2e-16 |
| item information `Ii` | 72,500 | 0.0 (exact) |
| MFI selection | 500 | 0 mismatches |
| bOpt selection | 500 | 0 mismatches |
| IRF derivatives (`Pi`/`Ii`/`Ji`) | 9,132 | 5.8e-11 |

### Well-behaved synthetic bank (40 items) — proves the BM/ML/WL estimation port

| Quantity | Comparisons | Max \|diff\| |
|----------|-------------|--------------|
| EAP `theta` / `se` | 240 | 8.9e-16 |
| BM / ML / WL `theta` | 720 | 8.2e-5 |
| BM / ML / WL `se` | 720 | 7.6e-5 |
| MFI / bOpt selection | 480 | 0 mismatches |

!!! note "About the BM/ML/WL tolerance"
    BM/ML/WL estimation solves an equation numerically; catR's own root-finder
    (`uniroot`) has a tolerance of ~1.2e-4, so the 8.2e-5 difference is *catR's
    own numerical precision*, not a catpie error. On the real (degenerate) EWM
    bank, BM/ML/WL are numerically unstable **in catR itself** (catR clamps
    ~1/3 of ML estimates to ±4) — only EAP is robust there, which is what the
    EWM experiment uses. That is why the BM/ML/WL check is only a hard
    pass/fail on the well-behaved bank.

## Regenerating the references

You only need R if you want to regenerate the ground-truth references from
scratch (e.g. for a different item bank):

```shell
# needs R with the catR and jsonlite packages installed
Rscript scripts/generate_reference.R /path/to/your/itembank.csv
Rscript scripts/generate_reference_goodbank.R
Rscript scripts/generate_reference_derivatives.R /path/to/your/itembank.csv
```

The item-bank CSV must have columns `discrimination`, `difficulty`, `guessing`,
`inattention`.
