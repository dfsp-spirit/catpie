#!/usr/bin/env Rscript
# Generate a compact reference of catR's derivative quantities (Pi / Ii / Ji
# with all derivatives) at several abilities, to validate the Python `pi`,
# `ii` and `ji` functions exactly.
#
# Usage:
#   Rscript scripts/generate_reference_derivatives.R /path/to/itembank.csv
#
# Requirements: R with the `catR` and `jsonlite` packages installed.

suppressMessages(library(catR))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
itembank_csv <- if (length(args) >= 1) args[1] else "assets/csv/itembank.csv"
out_json     <- "reference/catr_derivatives.json"

stopifnot(file.exists(itembank_csv))
cat("Reading item bank from:", itembank_csv, "\n")

df <- read.csv(itembank_csv)
bank <- as.matrix(df[, c("discrimination", "difficulty", "guessing", "inattention")])
colnames(bank) <- c("a", "b", "c", "d")

thetas <- c(-3, -2, -1, 0, 1, 2, 3)

rows <- list()
for (th in thetas) {
  p <- Pi(th, bank, D = 1)
  I <- Ii(th, bank, D = 1)
  J <- Ji(th, bank, D = 1)
  for (i in seq_len(nrow(bank))) {
    rows[[length(rows) + 1]] <- list(
      theta = th,
      item  = i - 1L,                     # 0-indexed
      P     = p$Pi[i],  dP = p$dPi[i],  d2P = p$d2Pi[i],  d3P = p$d3Pi[i],
      Ii    = I$Ii[i], dIi = I$dIi[i], d2Ii = I$d2Ii[i],
      Ji    = J$Ji[i], dJi = J$dJi[i]
    )
  }
}

ref <- list(
  r_version    = paste(R.version$major, R.version$minor, sep = "."),
  catr_version = as.character(packageVersion("catR")),
  n_items      = nrow(bank),
  n_rows       = length(rows),
  rows         = rows
)

dir.create(dirname(out_json), showWarnings = FALSE, recursive = TRUE)
write_json(ref, out_json, digits = 16, auto_unbox = TRUE)
cat("Wrote", length(rows), "derivative rows to:", out_json, "\n")
