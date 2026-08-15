"""Minimal demo of catpie, mirroring catjs-irt's examples/demo.mjs."""

from catpie import estimateTheta, genPattern, randomCAT, selectNextItem

# Item bank: (a, b, c, d) = (discrimination, difficulty, guessing, inattention)
bank = [
    (1.0, -1.0, 0.20, 0.95),
    (1.5, 1.0, 0.10, 0.98),
    (0.9, 0.0, 0.15, 0.97),
    (1.2, 0.5, 0.05, 0.99),
    (1.1, -0.5, 0.25, 0.96),
]

administered = []
responses = []

# Select the first item (theta starts at 0)
sel = selectNextItem(bank, 0.0, administered)
administered.append(sel.item)
print(f"First item selected: #{sel.item} (criterion={sel.criterion}, info={sel.info:.4f})")

# ... run the trial, score it (0/1) ...
r = genPattern(0.7, [bank[sel.item]], rng=lambda: 0.37)[0]
responses.append(r)

# Estimate ability + SE (EAP by default; BM/ML/WL via method=...)
res = estimateTheta(bank, administered, responses)
print(f"After 1 item: theta={res.theta:.4f}, se={res.se:.4f}")

# Full simulation (selection + response + estimation + stopping)
run = randomCAT(
    0.7,
    bank,
    method="EAP",
    stop={"rule": ["precision", "length"], "thr": [0.3, 10]},
    rng=lambda: 0.37,
)
print(f"randomCAT: administered {run.nItems} items, final theta={run.finalTheta:.4f}, "
      f"se={run.finalSe:.4f}, stopRule={run.stopRule}")
