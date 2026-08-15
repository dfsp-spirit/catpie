#!/usr/bin/env Rscript
# Generate a catR reference on a WELL-BEHAVED synthetic item bank (realistic
# discrimination values) to verify the BM/ML/WL estimation and bOpt selection
# ports in isolation from the degenerate real bank. This script is fully
# self-contained (it generates its own item bank with a fixed seed).
#
# Usage:
#   Rscript scripts/generate_reference_goodbank.R
#
# Requirements: R with the `catR` and `jsonlite` packages installed.

suppressMessages(library(catR))
suppressMessages(library(jsonlite))

out_json <- "reference/catr_reference_goodbank.json"

set.seed(1234)
n_items <- 40
bank <- cbind(
  a = runif(n_items, 0.5, 2.0),
  b = runif(n_items, -2.5, 2.5),
  c = runif(n_items, 0.0, 0.3),
  d = runif(n_items, 0.90, 1.0)
)
colnames(bank) <- c("a", "b", "c", "d")

p_correct <- function(th, par) Pi(th, rbind(par), D = 1)$Pi

simulate_run <- function(true_theta, seed, max_items = 12) {
  set.seed(seed)
  administered <- integer(0)
  responses <- integer(0)
  steps <- list()
  for (k in 1:max_items) {
    theta_used <- if (k == 1) 0.0 else steps[[k - 1]]$theta
    nxt <- nextItem(itemBank = bank, theta = theta_used, out = administered,
                    x = responses, criterion = "MFI")
    selected <- nxt$item
    info_vec <- as.numeric(Ii(theta_used, bank, D = 1)$Ii)
    p <- p_correct(true_theta, bank[selected, ])
    r <- rbinom(1, 1, p)
    administered <- c(administered, selected)
    responses <- c(responses, r)

    est_one <- function(m, pd = "norm") {
      th <- thetaEst(bank[administered, , drop = FALSE], responses, method = m, priorDist = pd)
      s <- semTheta(thEst = th, it = bank[administered, , drop = FALSE],
                    x = responses, method = m, priorDist = pd)
      list(theta = as.numeric(th), se = as.numeric(s))
    }
    estimates <- list(
      EAP_norm = est_one("EAP"), BM_norm = est_one("BM"), ML_norm = est_one("ML"),
      WL_norm = est_one("WL")
    )
    nxt_bopt <- nextItem(itemBank = bank, theta = theta_used, out = administered,
                         x = responses, criterion = "bOpt")
    steps[[k]] <- list(
      step = k, true_theta = true_theta,
      administered0 = as.integer(administered) - 1L,
      responses = as.integer(responses),
      theta_used = theta_used,
      selected0 = as.integer(selected) - 1L,
      info = info_vec,
      bopt0 = as.integer(nxt_bopt$item) - 1L,
      theta = as.numeric(est_one("EAP")$theta),
      se = as.numeric(est_one("EAP")$se),
      estimates = estimates,
      p_true = p
    )
  }
  steps
}

true_thetas <- c(-2.0, -1.0, 0.0, 1.0, 2.0)
seeds <- 201:204   # 4 seeds per theta level
max_items <- 12

runs <- list()
for (tt in true_thetas) {
  for (seed in seeds) {
    runs[[length(runs) + 1]] <- list(true_theta = tt, seed = seed,
                                     steps = simulate_run(tt, seed, max_items))
  }
}

cat(sprintf("Simulated %d runs x %d steps on a %d-item synthetic bank.\n",
            length(runs), length(runs[[1]]$steps), n_items))

ref <- list(
  r_version     = paste(R.version$major, R.version$minor, sep = "."),
  catr_version  = as.character(packageVersion("catR")),
  n_items       = n_items,
  itembank      = unname(bank),
  runs          = runs
)

dir.create(dirname(out_json), showWarnings = FALSE, recursive = TRUE)
write_json(ref, out_json, digits = 16, auto_unbox = TRUE)
cat("Wrote reference to:", out_json, "\n")
