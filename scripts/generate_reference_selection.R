#!/usr/bin/env Rscript
# Generate a JSON reference for catR's *item selection* with randomesque > 1,
# so the Python port (catpie) can be validated against the real R package.
#
# What is recorded per case:
#   - the full criterion vector (Ii for MFI, |b - theta| for bOpt) so catpie can
#     be checked against catR's numbers, not just its decision;
#   - the item catR selects for one fixed seed;
#   - the *support* and frequency table over many seeds. The candidate set is
#     not returned by catR, so we recover it empirically: with k <= 5 every
#     candidate is drawn with probability >= 1/5 per call, hence it shows up
#     in a few hundred draws with overwhelming probability.
#
# Usage:
#   Rscript scripts/generate_reference_selection.R
#
# Requirements: R with the `catR` and `jsonlite` packages installed.

suppressMessages(library(catR))
suppressMessages(library(jsonlite))

out_json <- "reference/catr_selection.json"
n_seeds <- 300L
fixed_seed <- 1L

# Bank A: distinct information values, includes c > 0 and d < 1 (real 4PL).
bankA <- rbind(
  c(1.00, -1.00, 0.20, 0.95),
  c(1.50,  1.00, 0.10, 0.98),
  c(0.80,  0.00, 0.00, 1.00),
  c(1.20,  0.50, 0.00, 1.00),
  c(2.00, -0.50, 0.25, 0.90),
  c(0.60,  1.60, 0.00, 1.00),
  c(1.80,  0.20, 0.15, 0.85),
  c(0.90,  2.00, 0.00, 1.00)
)

# Bank B: exact duplicates, to force ties in the criterion values.
bankB <- rbind(
  c(1.00,  0.00, 0.00, 1.00),
  c(1.00,  0.00, 0.00, 1.00),
  c(1.00,  1.00, 0.00, 1.00),
  c(1.00,  1.00, 0.00, 1.00),
  c(1.50,  0.25, 0.00, 1.00)
)

cases <- list()
# NOTE ON INDEXING: item indices in the JSON are 0-based (catpie convention),
# but catR's `out` and the returned `item` are 1-based, so they are converted
# at the boundary here.
add_case <- function(bank, bank_name, theta, out, criterion, randomesque) {
  out1 <- as.integer(out) + 1L
  info <- if (criterion == "MFI") Ii(theta, bank)$Ii else abs(bank[, 2] - theta)

  set.seed(fixed_seed)
  one <- nextItem(bank, theta = theta, out = out1, criterion = criterion,
                  randomesque = randomesque, random.seed = fixed_seed)$item

  counts <- integer(nrow(bank))
  for (s in seq_len(n_seeds)) {
    set.seed(s)
    it <- nextItem(bank, theta = theta, out = out1, criterion = criterion,
                   randomesque = randomesque, random.seed = s)$item
    counts[it] <- counts[it] + 1L
  }

  cases[[length(cases) + 1L]] <<- list(
    bank = bank_name,
    theta = theta,
    out = I(as.integer(out)),  # I() keeps it an array even when length 1
    criterion = criterion,
    randomesque = randomesque,
    seed = fixed_seed,
    info = as.numeric(info),
    item = as.integer(one) - 1L,
    n_seeds = n_seeds,
    support = as.integer(which(counts > 0L) - 1L),
    counts = as.integer(counts)
  )
}

for (theta in c(-1.5, 0.0, 1.0)) {
  for (criterion in c("MFI", "bOpt")) {
    for (k in c(1L, 2L, 3L, 5L, 10L)) {
      add_case(bankA, "A", theta, integer(0), criterion, k)
    }
    add_case(bankA, "A", theta, c(0L, 2L), criterion, 3L)
    add_case(bankA, "A", theta, 0L:5L, criterion, 3L)
  }
}

# Tie bank: k = 1 must stay on the tie group, k = 2 must widen it.
for (criterion in c("MFI", "bOpt")) {
  for (k in c(1L, 2L, 3L)) {
    add_case(bankB, "B", 0.0, integer(0), criterion, k)
    add_case(bankB, "B", 0.1, c(4L), criterion, k)
  }
}

payload <- list(
  meta = list(
    generated_by = "scripts/generate_reference_selection.R",
    catR_version = as.character(packageVersion("catR")),
    R_version = R.version.string,
    n_seeds = n_seeds,
    fixed_seed = fixed_seed
  ),
  banks = list(
    A = lapply(seq_len(nrow(bankA)), function(i) as.numeric(bankA[i, ])),
    B = lapply(seq_len(nrow(bankB)), function(i) as.numeric(bankB[i, ]))
  ),
  cases = cases
)

write_json(payload, out_json, auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("Wrote", length(cases), "cases to", out_json, "\n")
