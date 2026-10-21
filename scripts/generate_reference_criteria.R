#!/usr/bin/env Rscript
# Generate JSON references for catR's remaining selection criteria (MLWI, MPWI,
# KL, KLP, thOpt, progressive, proportional, random) and for the estimator
# features catpie gained last (constantPatt, ROB, exact SE).
#
# Two files are written:
#   reference/catr_criteria.json    -- criterion values, selections and the
#                                      selection support over many seeds
#   reference/catr_estimation.json  -- constantPatt / ROB / sem.exact values
#
# Usage:
#   Rscript scripts/generate_reference_criteria.R
#
# Requirements: R with the `catR` and `jsonlite` packages.

suppressMessages(library(catR))
suppressMessages(library(jsonlite))

n_seeds <- 300L
fixed_seed <- 1L

bank <- rbind(
  c(1.00, -1.00, 0.20, 0.95),
  c(1.50,  1.00, 0.10, 0.98),
  c(0.80,  0.00, 0.00, 1.00),
  c(1.20,  0.50, 0.00, 1.00),
  c(2.00, -0.50, 0.25, 0.90),
  c(0.60,  1.60, 0.00, 1.00)
)

# ---------------------------------------------------------------- criteria ---
cases <- list()

add_case <- function(theta, out, x, criterion, randomesque, extra = list()) {
  out1 <- as.integer(out) + 1L
  itGiven <- if (length(out1) == 0) bank[integer(0), ] else rbind(bank[out1, ])

  values <- NULL
  if (criterion %in% c("MLWI", "MPWI")) {
    values <- sapply(seq_len(nrow(bank)), function(i)
      MWI(bank, i, x = x, it.given = itGiven, type = criterion))
  } else if (criterion %in% c("KL", "KLP")) {
    values <- sapply(seq_len(nrow(bank)), function(i)
      KL(bank, i, x = x, it.given = itGiven, theta = theta, type = criterion))
  } else if (criterion %in% c("MFI", "bOpt")) {
    values <- if (criterion == "MFI") Ii(theta, bank)$Ii else abs(bank[, 2] - theta)
  }

  set.seed(fixed_seed)
  one <- nextItem(bank, theta = theta, out = out1, x = x, criterion = criterion,
                  randomesque = randomesque, random.seed = fixed_seed,
                  thr = 5, maxItems = 20)$item

  counts <- integer(nrow(bank))
  for (s in seq_len(n_seeds)) {
    set.seed(s)
    it <- nextItem(bank, theta = theta, out = out1, x = x, criterion = criterion,
                   randomesque = randomesque, random.seed = s,
                   thr = 5, maxItems = 20)$item
    counts[it] <- counts[it] + 1L
  }

  case <- c(list(
    theta = theta,
    out = I(as.integer(out)),
    x = I(as.integer(x)),
    criterion = criterion,
    randomesque = randomesque,
    seed = fixed_seed,
    values = if (is.null(values)) NULL else as.numeric(values),
    item = as.integer(one) - 1L,
    n_seeds = n_seeds,
    support = as.integer(which(counts > 0L) - 1L),
    counts = as.integer(counts)
  ), extra)
  cases[[length(cases) + 1L]] <<- case
}

states <- list(
  list(theta = 0.0, out = integer(0), x = integer(0)),
  list(theta = 0.4, out = c(0L, 3L), x = c(1L, 0L)),
  list(theta = -0.8, out = c(1L, 4L, 5L), x = c(0L, 1L, 1L))
)

for (st in states) {
  for (criterion in c("MFI", "bOpt", "MLWI", "MPWI", "KL", "KLP")) {
    add_case(st$theta, st$out, st$x, criterion, 1L)
    add_case(st$theta, st$out, st$x, criterion, 3L)
  }
  for (criterion in c("thOpt", "progressive", "proportional", "random")) {
    add_case(st$theta, st$out, st$x, criterion, 1L)
  }
}

criteria_payload <- list(
  meta = list(
    generated_by = "scripts/generate_reference_criteria.R",
    catR_version = as.character(packageVersion("catR")),
    R_version = R.version.string,
    n_seeds = n_seeds,
    note = "item indices are 0-based; catR's 1-based `out`/`item` are converted"
  ),
  bank = lapply(seq_len(nrow(bank)), function(i) as.numeric(bank[i, ])),
  cases = cases
)
write_json(criteria_payload, "reference/catr_criteria.json", auto_unbox = TRUE,
           digits = NA, pretty = TRUE)

# -------------------------------------------------------------- estimation ---
it <- rbind(bank[1, ], bank[3, ], bank[4, ])
patterns <- list(
  c(1L, 1L, 1L),  # all correct -> degenerate
  c(0L, 0L, 0L),  # all wrong   -> degenerate
  c(1L, 0L, 1L),
  c(0L, 1L, 0L)
)

estimation <- list()
for (p in patterns) {
  entry <- list(
    x = I(as.integer(p)),
    ML = thetaEst(it, p, method = "ML"),
    BM = thetaEst(it, p, method = "BM"),
    EAP = thetaEst(it, p, method = "EAP"),
    ROB_Huber = thetaEst(it, p, method = "ROB", weight = "Huber", tuCo = 1),
    ROB_Tukey = thetaEst(it, p, method = "ROB", weight = "Tukey", tuCo = 1),
    fixed4 = tryCatch(thetaEst(it, p, method = "ML", constantPatt = "fixed4",
                               current.th = 0.25), error = function(e) NA),
    fixed7 = tryCatch(thetaEst(it, p, method = "ML", constantPatt = "fixed7",
                               current.th = 0.25), error = function(e) NA),
    var = tryCatch(thetaEst(it, p, method = "ML", constantPatt = "var",
                            current.th = 0.25, bRange = c(-3, 2)),
                   error = function(e) NA),
    BMconstant = tryCatch(thetaEst(it, p, method = "ML", constantPatt = "BM"),
                          error = function(e) NA)
  )
  estimation[[length(estimation) + 1L]] <- entry
}

seReference <- list()
for (p in patterns[3:4]) {
  th <- thetaEst(it, p, method = "BM")
  seReference[[length(seReference) + 1L]] <- list(
    x = I(as.integer(p)),
    theta = th,
    classic = semTheta(th, it, p, method = "BM", sem.type = "classic"),
    new = semTheta(th, it, p, method = "BM", sem.type = "new"),
    exact = semTheta(th, it, p, method = "BM", sem.exact = TRUE)[1],
    exact_true = semTheta(th, it, p, method = "BM", sem.exact = TRUE,
                          trueTh = th)[1],
    ROB = semTheta(thetaEst(it, p, method = "ROB"), it, p, method = "ROB",
                   weight = "Huber", tuCo = 1)
  )
}

estimation_payload <- list(
  meta = list(
    generated_by = "scripts/generate_reference_criteria.R",
    catR_version = as.character(packageVersion("catR")),
    R_version = R.version.string
  ),
  bank = lapply(seq_len(nrow(it)), function(i) as.numeric(it[i, ])),
  patterns = estimation,
  se = seReference
)
write_json(estimation_payload, "reference/catr_estimation.json",
           auto_unbox = TRUE, digits = NA, pretty = TRUE)

cat("Wrote", length(cases), "criteria cases and",
    length(estimation), "estimation cases\n")
